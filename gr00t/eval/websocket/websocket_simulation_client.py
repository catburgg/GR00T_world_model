"""
WebSocket-based simulation client that wraps WebsocketClientPolicy.
"""
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.simulation import SimulationConfig, _create_single_env
from gr00t.eval.websocket.client import WebsocketClientPolicy
from gr00t.model.policy import BasePolicy


class WebSocketSimulationClient(BasePolicy):
    """WebSocket-based simulation client that mimics SimulationInferenceClient."""
    
    def __init__(self, host: str = "localhost", port: int = 8864):
        """Initialize WebSocket client.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        self.ws_client = WebsocketClientPolicy(host=host, port=port)
        self.env = None
    
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the WebSocket server based on observations.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Dictionary containing actions
        """
        # Hot fix: 处理观察键名映射
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        
        # 调用 WebSocket 客户端
        return self.ws_client.infer([observations])
    
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from WebSocket server.
        
        Returns:
            Dictionary of modality configurations
        """
        metadata = self.ws_client.get_server_metadata()
        return metadata.get('modality_config', {})
    
    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Configured vectorized environment
        """
        # Create environment functions for each parallel environment
        env_fns = [
            partial(_create_single_env, config=config, idx=i) 
            for i in range(config.n_envs)
        ]
        
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )
    
    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Tuple of (environment_name, list_of_episode_successes)
        """
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} "
            f"with {config.n_envs} environments"
        )
        
        # Set up the environment
        self.env = self.setup_environment(config)
        
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        
        # Initial environment reset
        obs, _ = self.env.reset()
        
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the server
            actions = self._get_actions_from_server(obs)
            
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    
                    print(
                        f"Episode {completed_episodes}/{config.n_episodes} "
                        f"{'succeeded' if current_successes[env_idx] else 'failed'} "
                        f"(length: {current_lengths[env_idx]})"
                    )
                    
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            
            obs = next_obs
        
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        
        elapsed_time = time.time() - start_time
        print(f"Collecting {config.n_episodes} episodes took {elapsed_time:.2f} seconds")
        print(f"Success rate: {np.mean(episode_successes):.2%}")
        
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        
        return config.env_name, episode_successes
    
    def _get_actions_from_server(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the WebSocket server.
        
        Args:
            observations: Dictionary of observations from the environment
            
        Returns:
            Dictionary of actions
        """
        # Get actions from the server
        action_dict = self.get_action(observations)
        
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        
        return actions
    
    def close(self):
        """Close the WebSocket connection and clean up resources."""
        if self.env is not None:
            self.env.close()
            self.env = None
        
        if hasattr(self, 'ws_client'):
            self.ws_client.close()