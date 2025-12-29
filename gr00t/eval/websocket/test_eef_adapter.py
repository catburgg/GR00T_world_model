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
    return np.asarray(frame["observation.state"][400], dtype=np.float64)


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
    }
    robot_config["nullspace_gains"] = np.ones(len(robot_config["joint_names"]))
    return robot_config


def _pose_errors_from_data(
    *,
    data: mujoco.MjData,
    site_ids: List[int],
    site_names: List[str],
    target_eef_poses: dict,
) -> dict:
    out = {}
    for name, sid in zip(site_names, site_ids):
        arm = "left" if "left" in name.lower() else "right"

        pos_key = f"{arm}_eef_pos"
        p_t = np.asarray(target_eef_poses[pos_key], dtype=float)
        p_i = data.site_xpos[sid].copy()
        out[f"{arm}_pos_err"] = float(np.linalg.norm(p_t - p_i))

        quat_key = f"{arm}_eef_quat_wxyz"
        q_t_wxyz = np.asarray(target_eef_poses[quat_key], dtype=float)
        q_t_xyzw = np.roll(q_t_wxyz, -1)
        R_t = T.quat2mat(q_t_xyzw)

        R_i = data.site_xmat[sid].reshape(3, 3).copy()
        R_err = R_t.T @ R_i
        cosang = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        out[f"{arm}_rot_err"] = float(np.arccos(cosang))

    return out


def test_batch_fk_ik_interfaces():
    """只测 batch 接口：构造 (1,1,D) 输入，FK->IK->写回->FK 误差，并打印耗时。"""
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

    # Print qpos before FK (in model layout)
    qpos_before = remap_qpos(raw_qpos, model)
    print("qpos before FK (model layout slices):")
    print(f"  waist   qpos[7:10]   = {qpos_before[7:10]}")
    print(f"  right   qpos[13:20]  = {qpos_before[13:20]}")
    print(f"  left    qpos[20:27]  = {qpos_before[20:27]}")

    # 1) batch FK
    t0 = time.perf_counter()
    eef_bn = adapter.forward_kinematics_batch(raw_qpos_bn)
    t1 = time.perf_counter()

    # 2) batch IK (use same raw as current_qpos)
    t2 = time.perf_counter()
    q_des_bn = adapter.inverse_kinematics_batch(eef_bn, current_qpos_bn=raw_qpos_bn)
    t3 = time.perf_counter()

    print(f"FK batch time: {(t1 - t0) * 1e3:.3f} ms")
    print(f"IK batch time: {(t3 - t2) * 1e3:.3f} ms")
    print(f"q_des_bn shape: {q_des_bn.shape} (expect (1,1,{len(adapter._qpos_adr)}))")

    # 3) Write IK result back to MuJoCo and compute reached pose
    remapped = remap_qpos(raw_qpos, model)
    data.qpos[:] = remapped
    mujoco.mj_forward(model, data)

    qfull = data.qpos.copy()
    qfull[np.asarray(adapter._qpos_adr, dtype=int)] = q_des_bn[0, 0]
    data.qpos[:] = qfull
    mujoco.mj_forward(model, data)

    # Print qpos after IK writeback (model layout slices)
    print("qpos after IK (model layout slices):")
    print(f"  waist   qpos[7:10]   = {data.qpos[7:10].copy()}")
    print(f"  right   qpos[13:20]  = {data.qpos[13:20].copy()}")
    print(f"  left    qpos[20:27]  = {data.qpos[20:27].copy()}")

    target_poses_single = {
        "left_eef_pos": eef_bn["left_eef_pos"][0, 0],
        "right_eef_pos": eef_bn["right_eef_pos"][0, 0],
        "left_eef_quat_wxyz": eef_bn["left_eef_quat_wxyz"][0, 0],
        "right_eef_quat_wxyz": eef_bn["right_eef_quat_wxyz"][0, 0],
    }

    errs = _pose_errors_from_data(
        data=data,
        site_ids=adapter.site_ids,
        site_names=adapter.site_names,
        target_eef_poses=target_poses_single,
    )

    print("Final EEF errors (single step, not iterative):")
    print(f"  left_pos_err:  {errs['left_pos_err']:.6f} m")
    print(f"  right_pos_err: {errs['right_pos_err']:.6f} m")
    print(f"  left_rot_err:  {errs['left_rot_err']:.6f} rad")
    print(f"  right_rot_err: {errs['right_rot_err']:.6f} rad")


if __name__ == "__main__":
    test_batch_fk_ik_interfaces()
