"""
Adapter to make GR00T policy compatible with WebSocket server/client interface.
"""
from typing import Any, Dict, List
import numpy as np
import torch
from PIL import Image


class Gr00tPolicyAdapter:
    """Adapts Gr00tPolicy to work with the WebSocket server interface."""
    
    OBS_KEY_MAPPING = {
        # 视频观察：客户端的长键名 -> GR00T 的短键名
        "video.ego_view_pad_res256_freq20": "video.ego_view",
        "video.ego_view_bg_crop_pad_res256_freq20": "video.ego_view",
        
        # 状态观察：保持不变
        "state.left_arm": "state.left_arm",
        "state.right_arm": "state.right_arm",
        "state.left_hand": "state.left_hand",
        "state.right_hand": "state.right_hand",
        "state.waist": "state.waist",
        
        "annotation.human.action.task_description": "annotation.human.coarse_action",
        "annotation.human.coarse_action": "annotation.human.coarse_action",
    }
    
    ACTION_KEY_MAPPING = {
        "action.left_arm": "action.left_arm",
        "action.right_arm": "action.right_arm",
        "action.left_hand": "action.left_hand",
        "action.right_hand": "action.right_hand",
        "action.waist": "action.waist",
    }
    
    def __init__(self, policy: Any):
        self.policy = policy
        self.modality_config = policy.modality_config
    
    def _convert_to_numpy(self, value):
        """Convert various types to numpy array."""
        if isinstance(value, Image.Image):
            return np.array(value)
        elif isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], Image.Image):
                return np.array([np.array(img) for img in value])
            return np.array(value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            return np.array(value)
    
    def _map_observation_keys(self, obs: Dict) -> Dict:
        """Map observation keys from client format to GR00T format."""
        mapped_obs = {}
        
        for client_key, value in obs.items():
            groot_key = self.OBS_KEY_MAPPING.get(client_key, client_key)
            
            # Convert to numpy arrays
            converted_value = self._convert_to_numpy(value)
            if groot_key in mapped_obs and groot_key != client_key:
                continue
            
            mapped_obs[groot_key] = converted_value
        
        return mapped_obs
    
    def _map_action_keys(self, action: Dict) -> Dict:
        """Map action keys from GR00T format to client format."""
        mapped_action = {}
        
        for groot_key, value in action.items():
            client_key = next(
                (k for k, v in self.ACTION_KEY_MAPPING.items() if v == groot_key),
                groot_key
            )
            mapped_action[client_key] = value
        
        return mapped_action
        
    def infer(self, obs_list: List[Dict]) -> Dict:
        """
        Interface expected by WebSocket server.
        
        Args:
            obs_list: List of observation dictionaries
            
        Returns:
            Action dictionary
        """
        if not obs_list:
            raise ValueError("obs_list is empty")
            
        # Get the latest observation
        obs = obs_list[-1] if isinstance(obs_list, list) else obs_list
        mapped_obs = self._map_observation_keys(obs)
        for key, value in mapped_obs.items():
            if not isinstance(value, np.ndarray):
                mapped_obs[key] = self._convert_to_numpy(value)
        action = self.policy.get_action(mapped_obs)
        mapped_action = self._map_action_keys(action)

        return mapped_action

    def reset(self):
        """Reset the policy if needed."""
        if hasattr(self.policy, 'reset'):
            self.policy.reset()


class GalbotPolicyAdapter:
    """Adapts Gr00tPolicy to the Galbot client observation/action format.

    Training keys (from your modality.json):
    - video.front_head_left  -> observation.images.front_head_left
    - state.qpos             -> observation.state[0:30]
    - annotation.language.action_text (we treat it as free-form string at inference)

    Output:
    - left_arm, left_gripper, right_arm, right_gripper sliced from action.qpos
    """

    def __init__(self, policy: Any, *, default_task: str = "No task."):
        self.policy = policy
        self.default_task = default_task

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, Image.Image):
            return np.array(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _build_groot_obs(self, obs: Dict) -> Dict[str, np.ndarray]:
        video = self._to_numpy(obs["pixels"]["head_left_rgb"])
        if video.ndim == 3:
            video = video[None, ...]  # (1,H,W,C) as T=1

        # ---- language ----
        lang = obs.get("task_description", None)
        lang = [str(lang)]

        # ---- state.qpos ----
        waist_head = self._to_numpy(obs["state"]["waist_head"])
        left_arm = self._to_numpy(obs["state"]["left_arm"])
        left_gripper = self._to_numpy(obs["state"]["left_gripper"])
        right_arm = self._to_numpy(obs["state"]["right_arm"])
        right_gripper = self._to_numpy(obs["state"]["right_gripper"])
        odom = self._to_numpy(obs["state"]["odom"])
        qpos = np.concatenate([waist_head.reshape(-1), left_arm.reshape(-1), left_gripper.reshape(-1), right_arm.reshape(-1), right_gripper.reshape(-1), odom.reshape(-1)], axis=-1)
        if qpos.ndim == 1:
            qpos = qpos[None, :]  # (1,30) as T=1

        return {
            "video.front_head_left": video.astype(np.uint8, copy=False),
            "state.qpos": qpos.astype(np.float32, copy=False),
            "annotation.language.action_text": np.array(lang, dtype=object),
        }

    @staticmethod
    def _slice_action_qpos(action_qpos: np.ndarray) -> Dict[str, np.ndarray]:
        # action_qpos: (H, D) or (B, H, D)
        if action_qpos.ndim == 2:
            horizon, dim = action_qpos.shape
            sl = slice(1, horizon)
            chunk = action_qpos[sl]  # (n, D)
            return {
                "left_arm": chunk[:, 7:14],
                "left_gripper": chunk[:, 14],
                "right_arm": chunk[:, 15:22],
                "right_gripper": chunk[:, 22],
            }
        if action_qpos.ndim == 3:
            b, horizon, dim = action_qpos.shape
            sl = slice(1, horizon)
            chunk = action_qpos[:, sl]  # (B, n, D)
            return {
                "left_arm": chunk[:, :, 7:14],
                "left_gripper": chunk[:, :, 14],
                "right_arm": chunk[:, :, 15:22],
                "right_gripper": chunk[:, :, 22],
            }
        raise ValueError(f"Unexpected action.qpos shape: {action_qpos.shape}")

    def infer(self, obs_list: List[Dict]) -> Dict[str, np.ndarray]:
        if not obs_list:
            raise ValueError("obs_list is empty")

        first = obs_list[0]
        if isinstance(first, dict) and first.get("mode") == "reset":
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            return {}

        obs = obs_list[-1]
        groot_obs = self._build_groot_obs(obs)

        action = self.policy.get_action(groot_obs)
        if "action.qpos" not in action:
            raise KeyError(f"Expected 'action.qpos' in policy output, got keys={list(action.keys())}")

        action_qpos = self._to_numpy(action["action.qpos"])
        return self._slice_action_qpos(action_qpos)
