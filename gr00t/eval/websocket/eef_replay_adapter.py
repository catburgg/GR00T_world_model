from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import pandas as pd
import robosuite

from gr00t.eval.websocket.eef_policy_adapter import EEFPolicyAdapter, remap_qpos


def _default_gr1_robot_xml_path() -> Path:
    return (
        Path(robosuite.__file__).parent
        / "models"
        / "assets"
        / "robots"
        / "gr1"
        / "robot.xml"
    )


def _default_gr1_robot_config() -> Dict[str, Any]:
    # Copied from gr00t/eval/websocket/test_eef_adapter.py (intentionally hard-coded).
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
            "left_eef_site": np.array(
                [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
                dtype=float,
            ),
            "right_eef_site": np.array(
                [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
                dtype=float,
            ),
        },
    }
    robot_config["nullspace_gains"] = np.ones(len(robot_config["joint_names"]))
    robot_config["ik_soft_limit_margin"] = 0.2
    return robot_config


class _DummyPolicy:
    # EEFPolicyAdapter expects policy.modality_config to exist.
    modality_config: dict = {}


class EEFReplayPolicyAdapter:
    """
    WebSocket server policy interface: `infer(obs_list) -> Dict[str, np.ndarray]`
    """

    # Hard-coded parquet column names
    COL_STATE = "observation.state"
    COL_ACTION = "action"
    COL_EEF_LEFT = "eef.left.wrist"
    COL_EEF_RIGHT = "eef.right.wrist"

    # We will always output single-step chunks to match `--n_action_steps 1`
    ACTION_HORIZON = 1

    def __init__(
        self,
        parquet_path: str,
        modality_json_path: str,
        *,
        robot_xml_path: Optional[str] = None,
        # If True, we try to read target eef directly from parquet columns (COL_EEF_LEFT/RIGHT).
        prefer_parquet_eef: bool = True,
        # Whether the eef target comes from row t (0) or row t+1 (1). Default 0.
        eef_target_offset: int = 0,
    ) -> None:
        self.df = pd.read_parquet(parquet_path)
        with open(modality_json_path, "r") as f:
            modality = json.load(f)
        
        # Slice map for parquet action vector -> action.* keys
        self.action_slices: dict[str, Tuple[int, int]] = {
            f"action.{k}": (int(v["start"]), int(v["end"])) for k, v in modality["action"].items()
        }

        # Build MuJoCo model + IK adapter
        xml_path = Path(robot_xml_path) if robot_xml_path is not None else _default_gr1_robot_xml_path()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.robot_config = _default_gr1_robot_config()
        self.eef = EEFPolicyAdapter(
            policy=_DummyPolicy(),
            model=self.model,
            data=self.data,
            robot_config=self.robot_config,
        )

        # Index pointer into parquet (shared across envs)
        self.current_idx = 0
        self.eef_target_offset = int(eef_target_offset)

        # Precompute mapping from IK q_des order -> our action keys
        # robot_config["joint_names"] = [waist(3), left_arm(7), right_arm(7)]
        self._ik_waist_slice = slice(0, 3)
        self._ik_left_arm_slice = slice(3, 10)
        self._ik_right_arm_slice = slice(10, 17)

    def _vec(self, x: Any) -> np.ndarray:
        return (x if isinstance(x, np.ndarray) else np.asarray(x))

    def _parse_eef_pose_vec(self, v: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse one EEF pose vector from parquet.
        Hard-coded format (as requested):
        - len==6: [x, y, z, roll, pitch, yaw]
        Returns: (pos(3,), rpy(3,))
        """
        arr = self._vec(v).reshape(-1)
        if arr.shape[0] != 6:
            raise ValueError(f"EEF pose must be 6D [x,y,z,roll,pitch,yaw], got shape {arr.shape}")
        pos = np.asarray(arr[:3], dtype=float)
        rpy = np.asarray(arr[3:6], dtype=float)
        return pos, rpy

    def _get_target_eef_from_parquet(self, row_idx: int) -> Dict[str, np.ndarray]:
        """
        Read target EEF poses from parquet columns, returning dict in EEFPolicyAdapter format.
        """
        left_raw = self.df[self.COL_EEF_LEFT].iloc[row_idx]
        right_raw = self.df[self.COL_EEF_RIGHT].iloc[row_idx]
        l_pos, l_rpy = self._parse_eef_pose_vec(left_raw)
        r_pos, r_rpy = self._parse_eef_pose_vec(right_raw)
        return {
            "left_eef_pos": l_pos,
            "right_eef_pos": r_pos,
            "left_eef_rpy": l_rpy,
            "right_eef_rpy": r_rpy,
        }

    def _infer_batch_size(self, obs_list: List[Dict[str, Any]]) -> int:
        """
        Infer B from any numpy observation with a leading batch dim.
        Falls back to 1.
        """
        if not obs_list:
            return 1
        obs = obs_list[-1]
        for v in obs.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 1:
                return int(v.shape[0])
        return 1

    def _row_vec(self, col: str, row_idx: int) -> np.ndarray:
        raw: Any = self.df[col].iloc[row_idx]
        vec = raw if isinstance(raw, np.ndarray) else np.asarray(raw)
        return vec.astype(np.float64, copy=False)

    def _slice_action_from_vec(self, action_vec: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split one-step action vector into dict of (1, dim) float32 arrays.
        """
        out: dict[str, np.ndarray] = {}
        for k, (s, e) in self.action_slices.items():
            out[k] = action_vec[s:e].astype(np.float32, copy=False)[None, :]
        return out

    def infer(self, obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        B = self._infer_batch_size(obs_list)
        n_rows = len(self.df)

        # Collect per-env current/next qpos in model-qpos layout
        cur_qpos_bn = np.zeros((B, 1, self.model.nq), dtype=np.float64)
        nxt_qpos_bn = np.zeros((B, 1, self.model.nq), dtype=np.float64)
        base_indices: list[int] = []

        for b in range(B):
            t = self.current_idx + b
            if t >= n_rows:
                t = n_rows - 1
            t_next = min(t + 1, n_rows - 1)
            base_indices.append(t)

            cur_raw = self._row_vec(self.COL_STATE, t)
            nxt_raw = self._row_vec(self.COL_STATE, t_next)
            cur_qpos_bn[b, 0] = remap_qpos(cur_raw, self.model)
            nxt_qpos_bn[b, 0] = remap_qpos(nxt_raw, self.model)

        target_eef_bn = {
            "left_eef_pos": np.zeros((B, 1, 3), dtype=float),
            "right_eef_pos": np.zeros((B, 1, 3), dtype=float),
            "left_eef_rpy": np.zeros((B, 1, 3), dtype=float),
            "right_eef_rpy": np.zeros((B, 1, 3), dtype=float),
        }
        for b in range(B):
            t = base_indices[b]
            t_tgt = min(max(t + self.eef_target_offset, 0), n_rows - 1)
            eef_t = self._get_target_eef_from_parquet(t_tgt)
            target_eef_bn["left_eef_pos"][b, 0] = eef_t["left_eef_pos"]
            target_eef_bn["right_eef_pos"][b, 0] = eef_t["right_eef_pos"]
            target_eef_bn["left_eef_rpy"][b, 0] = eef_t["left_eef_rpy"]
            target_eef_bn["right_eef_rpy"][b, 0] = eef_t["right_eef_rpy"]

        # IK(target, current) -> q_des for ctrl joints
        q_des_bn = self.eef.inverse_kinematics_batch(target_eef_bn, current_qpos_bn=cur_qpos_bn)  # (B,1,17)

        # Build output dict: start from parquet action slices, then override waist/arms
        out_batched: dict[str, np.ndarray] = {}

        for b in range(B):
            t = base_indices[b]
            action_vec = self._row_vec(self.COL_ACTION, t).astype(np.float32, copy=False)
            one = self._slice_action_from_vec(action_vec)  # each key -> (1,dim)

            # Override waist/arms with IK result (q_des order is hard-coded)
            q = q_des_bn[b, 0].astype(np.float32, copy=False)
            one["action.waist"] = q[self._ik_waist_slice][None, :]
            one["action.left_arm"] = q[self._ik_left_arm_slice][None, :]
            one["action.right_arm"] = q[self._ik_right_arm_slice][None, :]

            # Merge into batched dict: stack on batch dim -> (B,1,dim)
            for k, v in one.items():
                if k not in out_batched:
                    out_batched[k] = np.zeros((B, *v.shape), dtype=v.dtype)  # (B,1,dim)
                out_batched[k][b] = v

        # Advance index by B (so parallel envs consume different rows)
        self.current_idx = min(self.current_idx + B, n_rows - 1)
        return out_batched

    def reset(self) -> None:
        self.current_idx = 0


