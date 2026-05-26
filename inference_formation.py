#!/usr/bin/env python
"""
Formation Flight Inference API
Simple API for running inference on trained formation flight model
"""
import os
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

from envs.JSBSim.envs import FormationEnv
from algorithms.mappo.ppo_actor import PPOActor

logger = logging.getLogger(__name__)


class FormationPolicy:
    """Wrapper for formation flight policy"""
    
    def __init__(
        self,
        model_dir: str,
        scenario_name: str = "formation/wedge",
        device: Optional[torch.device] = None,
        deterministic: bool = True
    ):
        """
        Initialize formation policy
        
        Args:
            model_dir: Path to trained model
            scenario_name: Scenario configuration
            device: Torch device (auto-detect if None)
            deterministic: Use deterministic policy
        """
        self.model_dir = model_dir
        self.deterministic = deterministic
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Create environment
        self.env = FormationEnv(scenario_name)
        self.num_agents = self.env.num_agents
        
        # Load policy
        self.policy = self._load_policy()
        
        # Initialize RNN states
        self.rnn_states = np.zeros((self.num_agents, 1, 128), dtype=np.float32)
        self.masks = np.ones((self.num_agents, 1), dtype=np.float32)
    
    def _load_policy(self) -> PPOActor:
        """Load trained policy"""
        logger.info(f"Loading policy from {self.model_dir}")
        
        # Configuration
        class PolicyConfig:
            gain = 0.01
            hidden_size = '128 128'
            act_hidden_size = '128 128'
            activation_id = 1
            use_feature_normalization = False
            use_recurrent_policy = True
            recurrent_hidden_size = 128
            recurrent_hidden_layers = 1
            tpdv = dict(dtype=torch.float32, device=self.device)
        
        config = PolicyConfig()
        config.tpdv = dict(dtype=torch.float32, device=self.device)
        
        # Create policy
        policy = PPOActor(config, self.env.observation_space, self.env.action_space, device=self.device)
        policy.eval()
        
        # Load weights
        policy_path = os.path.join(self.model_dir, "actor.pt")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        checkpoint = torch.load(policy_path, map_location=self.device)
        policy.load_state_dict(checkpoint)
        
        logger.info("✓ Policy loaded successfully")
        return policy
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        obs, info = self.env.reset()
        self.rnn_states = np.zeros((self.num_agents, 1, 128), dtype=np.float32)
        self.masks = np.ones((self.num_agents, 1), dtype=np.float32)
        return obs, info
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action from policy
        
        Args:
            obs: Observations from environment
        
        Returns:
            Actions for each agent
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(self.device)
            rnn_tensor = torch.from_numpy(self.rnn_states).to(self.device)
            masks_tensor = torch.from_numpy(self.masks).to(self.device)
            
            actions, _, rnn_states_out = self.policy(
                obs_tensor,
                rnn_tensor,
                masks_tensor,
                deterministic=self.deterministic
            )
            
            actions = actions.detach().cpu().numpy()
            self.rnn_states = rnn_states_out.detach().cpu().numpy()
        
        return actions
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute one environment step
        
        Args:
            actions: Actions for each agent
        
        Returns:
            obs, rewards, dones, info
        """
        obs, _, rewards, dones, info = self.env.step(actions)
        
        # Update masks for terminated agents
        for i, done_flag in enumerate(dones):
            if done_flag:
                self.rnn_states[i] = 0
                self.masks[i] = 0
        
        return obs, rewards, dones, info
    
    def run_episode(
        self,
        max_steps: Optional[int] = None,
        render: bool = False,
        render_file: Optional[str] = None
    ) -> Dict:
        """
        Run single episode
        
        Args:
            max_steps: Max steps per episode
            render: Whether to render
            render_file: File for trajectory recording
        
        Returns:
            Episode statistics
        """
        obs, _ = self.reset()
        
        if render and render_file:
            self.env.render(mode='txt', filepath=render_file)
        
        step_rewards = np.zeros(self.num_agents)
        step = 0
        done = False
        
        while not done:
            # Get action
            actions = self.get_action(obs)
            
            # Step environment
            obs, rewards, dones, info = self.step(actions)
            step_rewards += rewards.squeeze()
            step += 1
            
            # Render
            if render and render_file:
                self.env.render(mode='txt', filepath=render_file)
            
            # Check termination
            if dones.all() or (max_steps and step >= max_steps):
                done = True
        
        stats = {
            'total_reward': step_rewards.sum(),
            'avg_reward': step_rewards.mean(),
            'steps': step,
            'agent_rewards': step_rewards
        }
        
        return stats
    
    def close(self):
        """Close environment"""
        self.env.close()


def run_inference(
    model_dir: str,
    num_episodes: int = 1,
    max_steps: Optional[int] = None,
    render: bool = True,
    scenario_name: str = "formation/wedge"
) -> Dict:
    """
    Run inference on trained policy
    
    Args:
        model_dir: Path to trained model
        num_episodes: Number of episodes
        max_steps: Max steps per episode
        render: Whether to render
        scenario_name: Scenario configuration
    
    Returns:
        Episode statistics
    """
    logger.info(f"Initializing formation policy from {model_dir}")
    
    policy = FormationPolicy(
        model_dir=model_dir,
        scenario_name=scenario_name,
        deterministic=True
    )
    
    all_stats = []
    
    for episode in range(num_episodes):
        render_file = f"formation_ep{episode}.txt.acmi" if render else None
        
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        stats = policy.run_episode(
            max_steps=max_steps,
            render=render,
            render_file=render_file
        )
        
        all_stats.append(stats)
        logger.info(f"  Total reward: {stats['total_reward']:.4f}")
        logger.info(f"  Avg reward: {stats['avg_reward']:.4f}")
        logger.info(f"  Steps: {stats['steps']}")
    
    policy.close()
    
    # Summary
    summary = {
        'num_episodes': num_episodes,
        'avg_total_reward': np.mean([s['total_reward'] for s in all_stats]),
        'avg_episode_length': np.mean([s['steps'] for s in all_stats]),
        'episodes': all_stats
    }
    
    logger.info(f"\nInference Summary:")
    logger.info(f"  Avg total reward: {summary['avg_total_reward']:.4f}")
    logger.info(f"  Avg episode length: {summary['avg_episode_length']:.1f}")
    
    return summary


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_formation.py <model_dir> [num_episodes] [max_steps]")
        print("Example: python inference_formation.py results/Formation/formation/wedge/mappo/v1/run1 2 1000")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    logging.basicConfig(level=logging.INFO)
    
    run_inference(
        model_dir=model_dir,
        num_episodes=num_episodes,
        max_steps=max_steps,
        render=True
    )
