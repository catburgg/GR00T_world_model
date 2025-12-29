"""
Adapter to make Gr00tPolicy compatible with WebSocket server/client interface.
"""
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image
import mujoco

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.eval.websocket.eef_policy_adapter import remap_qpos


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
    
    def __init__(self, policy: Gr00tPolicy):
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
