import numpy as np
import mujoco
import pandas as pd
import time
from pathlib import Path
from typing import List
import robosuite
import robosuite.utils.transform_utils as T
from gr00t.eval.websocket.eef_policy_adapter import EEFPolicyAdapter, remap_qpos

URDF_OR_MJCF_PATH = "/home/asus/project/any4lerobot/gr1/robot.xml"
PARQUET_PATH = "/home/asus/project/any4lerobot/gr1/test.parquet"
EEF_SITES = ["left_eef_site", "right_eef_site"]


def load_qpos(path: str) -> np.ndarray:
    frame = pd.read_parquet(path)
    return np.asarray(frame["observation.state"][0], dtype=np.float64)


def joint_qpos_dim(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def dump_qpos_layout(model: mujoco.MjModel) -> None:
    import pandas as pd

    rows = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        start = int(model.jnt_qposadr[j])
        width = joint_qpos_dim(model.jnt_type[j])
        rows.append((start, start + width, width, name, int(model.jnt_type[j])))
    header = ["start", "end", "width", "joint_name", "joint_type"]
    print(pd.DataFrame(rows, columns=header).sort_values("start"))


def mat_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    R = np.clip(R, -1.0, 1.0)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw])

def rpy2mat(rpy):
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

def get_gr1_robot_config():
    robot_config = {
        "joint_names": [
            "torso_waist_yaw",
            "torso_waist_pitch",
            "torso_waist_roll",
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow_pitch",
            "l_wrist_yaw",
            "l_wrist_roll",
            "l_wrist_pitch",
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_shoulder_yaw",
            "r_elbow_pitch",
            "r_wrist_yaw",
            "r_wrist_roll",
            "r_wrist_pitch",
        ],
        "end_effector_sites": [
            "left_eef_site",
            "right_eef_site",
        ],
        "site_transforms": {
            "left_eef_site": np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
            "right_eef_site": np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
        }
    }
    robot_config["nullspace_gains"] = np.ones(len(robot_config["joint_names"]))
    return robot_config


def test_batch_fk_ik_interfaces():
    print("=" * 60)
    print("Testing batch FK/IK interfaces ((B,N,D) = (1,1,D))")
    print("=" * 60)

    robot_xml_path = Path(robosuite.__file__).parent / "models" / "assets" / "robots" / "gr1" / "robot.xml"
    if not robot_xml_path.exists():
        raise FileNotFoundError(f"Robot XML not found at {robot_xml_path}")

    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    data = mujoco.MjData(model)

    robot_config = get_gr1_robot_config()
    robot_config["ik_soft_limit_margin"] = 0.2

    class DummyPolicy:
        def __init__(self):
            self.modality_config = {}

    adapter = EEFPolicyAdapter(
        policy=DummyPolicy(),
        model=model,
        data=data,
        robot_config=robot_config,
    )

    # raw qpos (observation layout)
    raw_qpos = load_qpos(PARQUET_PATH)  # (44,)
    raw_qpos_bn = raw_qpos.reshape(1, 1, -1)  # (1,1,44)

    # 1) batch FK
    eef_bn = adapter.forward_kinematics_batch(raw_qpos_bn)
    print(eef_bn)

    # 2) batch IK (use same raw as current_qpos)
    q_des_bn = adapter.inverse_kinematics_batch(eef_bn, current_qpos_bn=raw_qpos_bn)
    
    # 3) Write IK result back to MuJoCo and compute reached pose
    remapped = remap_qpos(raw_qpos, model)
    print(remapped)
    data.qpos[:] = remapped
    mujoco.mj_forward(model, data)

    qfull = data.qpos.copy()
    qfull[np.asarray(adapter._qpos_adr, dtype=int)] = q_des_bn[0, 0]
    data.qpos[:] = qfull
    mujoco.mj_forward(model, data)

if __name__ == "__main__":
    test_batch_fk_ik_interfaces()
