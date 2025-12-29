"""
EEF (End-Effector) Policy Adapter that converts between joint positions (qpos) and end-effector poses.
"""
from typing import Dict, List, Optional

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T

from gr00t.policy.gr00t_policy import Gr00tPolicy

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

def remap_qpos(raw: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Remap qpos from observation format to model format."""
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[:7] = model.qpos0[:7]  # base

    src = {
        "waist": slice(41, 44),
        "neck": slice(19, 22),
        "right_arm": slice(22, 29),
        "left_arm": slice(0, 7),
        "left_leg": slice(13, 19),
        "right_leg": slice(35, 41),
    }
    dst_order = [
        ("waist", slice(7, 10)),
        ("neck", slice(10, 13)),
        ("right_arm", slice(13, 20)),
        ("left_arm", slice(20, 27)),
        ("left_leg", slice(27, 33)),
        ("right_leg", slice(33, 39)),
    ]
    for name, target in dst_order:
        qpos[target] = raw[src[name]]
    return qpos


class EEFPolicyAdapter:
    """FK/IK adapter for GR1 EEF-space control (assume batch inputs are (B,N,D))."""

    def __init__(
        self,
        policy: Gr00tPolicy,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_config: Dict,
    ):
        self.policy = policy
        self.modality_config = getattr(policy, "modality_config", {})
        self.model = model
        self.data = data
        self.robot_config = robot_config

        # IK-only soft-limit relaxation
        self.site_transforms = robot_config["site_transforms"]
        self.ik_soft_limit_margin = robot_config["ik_soft_limit_margin"]

        # Sites
        self.site_names = robot_config["end_effector_sites"]
        self.site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.site_names
        ]

        # IK solver
        from robosuite.utils.ik_utils import IKSolver

        self.ik_solver = IKSolver(
            model=model,
            data=data,
            robot_config=robot_config,
            damping=0.01,
            integration_dt=0.1,
            max_dq=1.0,
            input_type="keyboard",
            input_action_repr="absolute",
            input_rotation_repr="quat_wxyz",
        )

        # ---- Fix robosuite IKSolver indexing assumptions for this model ----
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ik_solver.joint_names
        ]
        self._dof_adr = np.array([int(model.jnt_dofadr[jid]) for jid in self._joint_ids], dtype=int)
        self._qpos_adr = np.array([int(model.jnt_qposadr[jid]) for jid in self._joint_ids], dtype=int)
        self._jnt_range_ids = np.array(self._joint_ids, dtype=int)

        # Jacobian columns are nv indices
        self.ik_solver.dof_ids = self._dof_adr

        # qpos is nq indices
        self.ik_solver.qpos_ids = self._qpos_adr
        self.ik_solver.jnt_range_ids = self._jnt_range_ids

        # torso indices are local indices into dq/q_des
        self.ik_solver.torso_dof_local_ids = np.array(
            [i for i, n in enumerate(self.ik_solver.joint_names) if "torso" in n], dtype=int
        )

        # Monkey-patch solve() to use the corrected indexing for nullspace, torso limit, and joint-range clipping.
        # We keep the original math but replace the wrong array indexing.
        _orig_solve = self.ik_solver.solve

        def _solve_patched(target_action: np.ndarray, Kpos: float = 0.95, Kori: float = 0.95):
            # Call original solve first to compute dq based on Jacobian / twist.
            # But we cannot trust its nullspace / torso / clip steps, so we re-run those parts safely here.
            target_action_reshaped = target_action.reshape(len(self.ik_solver.site_names), -1)
            target_pos = target_action_reshaped[:, : self.ik_solver.pos_dim]
            target_ori = target_action_reshaped[:, self.ik_solver.pos_dim :]

            # Build target_quat_wxyz exactly like the original
            if self.ik_solver.input_rotation_repr == "axis_angle":
                target_quat_wxyz = np.array(
                    [np.roll(T.axisangle2quat(target_ori[i]), 1) for i in range(len(target_ori))]
                )
            elif self.ik_solver.input_rotation_repr == "mat":
                target_quat_wxyz = np.array([np.roll(T.mat2quat(target_ori[i])) for i in range(len(target_ori))])
            elif self.ik_solver.input_rotation_repr == "quat_wxyz":
                target_quat_wxyz = target_ori
            else:
                raise ValueError(f"Unsupported rotation repr: {self.ik_solver.input_rotation_repr}")

            # Handle relative / relative_pose the same way as upstream
            if "relative" in self.ik_solver.input_action_repr:
                cur_pos = np.array(
                    [self.ik_solver.full_model_data.site(sid).xpos for sid in self.ik_solver.site_ids]
                )
                cur_ori = np.array(
                    [self.ik_solver.full_model_data.site(sid).xmat for sid in self.ik_solver.site_ids]
                )

            if self.ik_solver.input_action_repr == "relative":
                target_pos = target_pos + cur_pos
                target_quat_xyzw = np.array(
                    [
                        T.quat_multiply(
                            T.mat2quat(cur_ori[i].reshape(3, 3)), np.roll(target_quat_wxyz[i], -1)
                        )
                        for i in range(len(self.ik_solver.site_ids))
                    ]
                )
                target_quat_wxyz = np.array(
                    [np.roll(target_quat_xyzw[i], shift=1) for i in range(len(self.ik_solver.site_ids))]
                )
            elif self.ik_solver.input_action_repr == "relative_pose":
                cur_poses = np.zeros((len(self.ik_solver.site_ids), 4, 4))
                for i in range(len(self.ik_solver.site_ids)):
                    cur_poses[i, :3, :3] = cur_ori[i].reshape(3, 3)
                    cur_poses[i, :3, 3] = cur_pos[i]
                    cur_poses[i, 3, :] = [0, 0, 0, 1]

                target_poses = np.zeros_like(cur_poses)
                for i in range(len(self.ik_solver.site_ids)):
                    target_poses[i, :3, :3] = T.quat2mat(target_quat_wxyz[i])
                    target_poses[i, :3, 3] = target_pos[i]
                    target_poses[i, 3, :] = [0, 0, 0, 1]

                new_target_poses = np.array(
                    [np.dot(cur_poses[i], target_poses[i]) for i in range(len(self.ik_solver.site_ids))]
                )
                target_pos = new_target_poses[:, :3, 3]
                target_quat_wxyz = np.array(
                    [
                        np.roll(T.mat2quat(new_target_poses[i, :3, :3]), shift=1)
                        for i in range(len(self.ik_solver.site_ids))
                    ]
                )

            # Jacobian (already correct because we overwrote dof_ids -> nv indices)
            jac = self.ik_solver._compute_jacobian(self.ik_solver.full_model, self.ik_solver.full_model_data)

            for i in range(len(self.ik_solver.site_ids)):
                dx = target_pos[i] - self.ik_solver.full_model_data.site(self.ik_solver.site_ids[i]).xpos
                self.ik_solver.twists[i][:3] = Kpos * dx / self.ik_solver.integration_dt
                mujoco.mju_mat2Quat(
                    self.ik_solver.site_quats[i],
                    self.ik_solver.full_model_data.site(self.ik_solver.site_ids[i]).xmat,
                )
                mujoco.mju_negQuat(self.ik_solver.site_quat_conjs[i], self.ik_solver.site_quats[i])
                mujoco.mju_mulQuat(
                    self.ik_solver.error_quats[i], target_quat_wxyz[i], self.ik_solver.site_quat_conjs[i]
                )
                mujoco.mju_quat2Vel(self.ik_solver.twists[i][3:], self.ik_solver.error_quats[i], 1.0)
                self.ik_solver.twists[i][3:] *= Kori / self.ik_solver.integration_dt

            twist = np.hstack(self.ik_solver.twists)
            diag = (self.ik_solver.damping**2) * np.eye(len(twist))
            eye = np.eye(len(self.ik_solver.dof_ids))

            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            # Correct nullspace: use qpos_ids (nq indices) instead of dof_ids (nv indices)
            q_cur = self.ik_solver.full_model_data.qpos[self.ik_solver.qpos_ids]
            dq_null = (eye - np.linalg.pinv(jac) @ jac) @ (self.ik_solver.Kn * (self.ik_solver.q0 - q_cur))
            dq = dq + dq_null

            # Global max_dq
            if self.ik_solver.max_dq > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > self.ik_solver.max_dq:
                    dq = dq * (self.ik_solver.max_dq / dq_abs_max)

            # Torso max dq: indices are local into dq (NOT joint ids)
            torso_local = getattr(self.ik_solver, "torso_dof_local_ids", np.array([], dtype=int))
            if torso_local.size > 0 and self.ik_solver.max_dq_torso > 0:
                dq_torso = dq[torso_local]
                dq_torso_abs_max = np.abs(dq_torso).max()
                if dq_torso_abs_max > self.ik_solver.max_dq_torso:
                    dq_torso = dq_torso * (self.ik_solver.max_dq_torso / dq_torso_abs_max)
                dq[torso_local] = dq_torso

            # Integrate in joint space (qpos for those joints)
            q_des = q_cur + dq * self.ik_solver.integration_dt

            # Clip with correct joint ranges (indexed by joint id)
            jnt_ranges = self.ik_solver.full_model.jnt_range[self.ik_solver.jnt_range_ids]
            lower = jnt_ranges[:, 0].copy()
            upper = jnt_ranges[:, 1].copy()
            margin = float(getattr(self, "ik_soft_limit_margin", 0.0))
            if margin > 0:
                lower = lower - margin
                upper = upper + margin
            np.clip(q_des, lower, upper, out=q_des)

            return q_des

        self.ik_solver.solve = _solve_patched

    def forward_kinematics(self, raw_qpos: np.ndarray) -> Dict[str, np.ndarray]:
        """Single-step FK: raw qpos (obs layout, 44) -> eef pose dict."""
        qpos = remap_qpos(raw_qpos, self.model)
        print(qpos)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        eef_poses: Dict[str, np.ndarray] = {}
        for name, sid in zip(self.site_names, self.site_ids):
            pos = self.data.site_xpos[sid].copy()
            rot = self.data.site_xmat[sid].reshape(3, 3).copy()
            rot = rot @ self.site_transforms[name][:3, :3]

            R = np.clip(rot, -1.0, 1.0)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
            rpy = np.array([roll, pitch, yaw], dtype=float)

            arm_name = "left" if "left" in name.lower() else "right"
            eef_poses[f"{arm_name}_eef_pos"] = pos
            eef_poses[f"{arm_name}_eef_rpy"] = rpy

        return eef_poses

    def forward_kinematics_batch(self, raw_qpos_bn: np.ndarray) -> Dict[str, np.ndarray]:
        """FK for raw_qpos shaped (B,N,44)."""
        raw = np.asarray(raw_qpos_bn)
        if raw.ndim != 3:
            raise ValueError(f"raw_qpos must be (B,N,D), got {raw.shape}")

        B, N, _ = raw.shape
        out = {
            "left_eef_pos": np.zeros((B, N, 3), dtype=float),
            "right_eef_pos": np.zeros((B, N, 3), dtype=float),
            "left_eef_rpy": np.zeros((B, N, 3), dtype=float),
            "right_eef_rpy": np.zeros((B, N, 3), dtype=float),
        }

        for b in range(B):
            for n in range(N):
                eef = self.forward_kinematics(raw[b, n])
                out["left_eef_pos"][b, n] = eef["left_eef_pos"]
                out["right_eef_pos"][b, n] = eef["right_eef_pos"]
                out["left_eef_rpy"][b, n] = eef["left_eef_rpy"]
                out["right_eef_rpy"][b, n] = eef["right_eef_rpy"]

        return out

    def inverse_kinematics_batch(
        self,
        target_eef_poses_bn: Dict[str, np.ndarray],
        current_qpos_bn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """IK for target poses shaped (B,N,*). Returns q_des shaped (B,N,n_ctrl)."""
        p = np.asarray(target_eef_poses_bn["left_eef_pos"])
        if p.ndim != 3:
            raise ValueError(f"target_eef_poses must be (B,N,*), got left_eef_pos {p.shape}")
        B, N, _ = p.shape

        q_out = np.zeros((B, N, len(self._qpos_adr)), dtype=float)
        cur = None if current_qpos_bn is None else np.asarray(current_qpos_bn)
        if cur is not None and cur.ndim != 3:
            raise ValueError(f"current_qpos must be (B,N,D), got {cur.shape}")

        for b in range(B):
            for n in range(N):
                poses = {
                    "left_eef_pos": np.asarray(target_eef_poses_bn["left_eef_pos"][b, n], dtype=float),
                    "right_eef_pos": np.asarray(target_eef_poses_bn["right_eef_pos"][b, n], dtype=float),
                    "left_eef_rpy": np.asarray(target_eef_poses_bn["left_eef_rpy"][b, n], dtype=float),
                    "right_eef_rpy": np.asarray(target_eef_poses_bn["right_eef_rpy"][b, n], dtype=float),
                }

                cur1 = None if cur is None else cur[b, n]
                q_out[b, n] = self.inverse_kinematics_iter(poses, current_qpos=cur1)

        return q_out

    def _build_target_action_from_rpy(self, target_eef_poses: Dict[str, np.ndarray]) -> np.ndarray:
        """Build solver target_action (pos + quat_wxyz per site) from dict that stores rpy."""
        target_action: List[float] = []

        for site_name in self.site_names:
            arm_name = "left" if "left" in site_name.lower() else "right"

            target_pos = np.asarray(target_eef_poses[f"{arm_name}_eef_pos"], dtype=float)
            target_rpy = np.asarray(target_eef_poses[f"{arm_name}_eef_rpy"], dtype=float)

            # Map adjusted-frame rotation back to MuJoCo site rotation:
            # R_site = R_adj @ inv(R_Tadj)
            R_adj = rpy2mat(target_rpy)
            R_Tadj = np.asarray(self.site_transforms[site_name][:3, :3], dtype=float)
            R_site = R_adj @ np.linalg.inv(R_Tadj)

            # robosuite T.mat2quat expects 3x3 and returns xyzw
            q_xyzw = T.mat2quat(R_site)
            q_wxyz = np.roll(q_xyzw, 1)

            target_action.extend(target_pos.tolist())
            target_action.extend(q_wxyz.tolist())

        return np.asarray(target_action, dtype=float)

    def _eef_errors_current(self, target_eef_poses: Dict[str, np.ndarray]) -> tuple[float, float]:
        """Return (pos_err_max, rot_err_max) for current self.data state vs target dict (rpy)."""
        pos_err_max = 0.0
        rot_err_max = 0.0

        for site_name, sid in zip(self.site_names, self.site_ids):
            arm_name = "left" if "left" in site_name.lower() else "right"

            # position error
            p_t = np.asarray(target_eef_poses[f"{arm_name}_eef_pos"], dtype=float)
            p_c = self.data.site_xpos[sid].copy()
            pos_err_max = max(pos_err_max, float(np.linalg.norm(p_t - p_c)))

            # rotation error (in adjusted frame)
            rpy_t = np.asarray(target_eef_poses[f"{arm_name}_eef_rpy"], dtype=float)
            R_t = rpy2mat(rpy_t)

            R_site = self.data.site_xmat[sid].reshape(3, 3).copy()
            R_adj = np.asarray(self.site_transforms[site_name][:3, :3], dtype=float)
            R_c = R_site @ R_adj

            R_err = R_t.T @ R_c
            cosang = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            rot_err = float(np.arccos(cosang))
            rot_err_max = max(rot_err_max, rot_err)

        return pos_err_max, rot_err_max

    def inverse_kinematics_iter(
        self,
        target_eef_poses: Dict[str, np.ndarray],
        current_qpos: Optional[np.ndarray] = None,
        *,
        max_iters: int = 50,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        verbose: bool = True,
    ) -> np.ndarray:
        if current_qpos is None:
            raise ValueError("inverse_kinematics_iter requires current_qpos (obs layout).")

        # init state
        qfull = remap_qpos(current_qpos, self.model)
        self.data.qpos[:] = qfull
        mujoco.mj_forward(self.model, self.data)
        self.ik_solver.q0 = qfull[self._qpos_adr].copy()

        target_action_np = self._build_target_action_from_rpy(target_eef_poses)

        q_des = qfull[self._qpos_adr].copy()
        for it in range(int(max_iters)):
            # make sure solver sees latest state
            mujoco.mj_forward(self.model, self.data)

            q_des = self.ik_solver.solve(target_action_np, Kpos=0.95, Kori=0.95)

            qfull = self.data.qpos.copy()
            qfull[self._qpos_adr] = q_des
            self.data.qpos[:] = qfull
            mujoco.mj_forward(self.model, self.data)

            pos_err, rot_err = self._eef_errors_current(target_eef_poses)
            if verbose:
                print(f"[IK it={it:02d}] pos_err={pos_err:.6e} rot_err={rot_err:.6e}")
            if pos_err <= pos_tol and rot_err <= rot_tol:
                break

        return q_des
