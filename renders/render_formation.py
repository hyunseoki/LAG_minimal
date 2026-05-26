#!/usr/bin/env python
"""
Render Formation Flight Task
Loads a trained MAPPO policy and renders the formation flight
"""
import sys
import os
import torch
import numpy as np
import logging
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.JSBSim.envs import FormationEnv
from algorithms.mappo.ppo_actor import PPOActor
from algorithms.mappo.ppo_critic import PPOCritic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _t2n(x):
    """Tensor to numpy"""
    return x.detach().cpu().numpy()


class FormationRenderConfig:
    """Configuration for rendering"""
    def __init__(self):
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))


def load_policy(policy_path, obs_space, act_space, device):
    """Load trained policy"""
    logger.info(f"Loading policy from {policy_path}")
    
    config = FormationRenderConfig()
    config.tpdv = dict(dtype=torch.float32, device=device)
    
    policy = PPOActor(config, obs_space, act_space, device=device)
    policy.eval()
    
    checkpoint = torch.load(policy_path, map_location=device)
    policy.load_state_dict(checkpoint)
    
    logger.info(f"✓ Policy loaded successfully")
    return policy


def render_formation(
    model_dir,
    env_name="Formation",
    scenario_name="formation/wedge",
    algorithm_name="mappo",
    experiment_name="v1",
    seed=5,
    num_episodes=1,
    max_steps=None,
    use_gpu=True,
    output_file=None,
    deterministic=True
):
    """
    Render trained formation flight policy
    
    Args:
        model_dir: Path to trained model directory
        env_name: Environment name
        scenario_name: Scenario name
        algorithm_name: Algorithm name
        experiment_name: Experiment name
        seed: Random seed
        num_episodes: Number of episodes to render
        max_steps: Max steps per episode (None = use env default)
        use_gpu: Use GPU for inference
        output_file: Output file for trajectory recording
        deterministic: Use deterministic policy
    """
    
    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using GPU for inference")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference")
    
    # Create environment
    logger.info(f"Creating environment: {env_name}/{scenario_name}")
    env = FormationEnv(scenario_name)
    env.seed(seed)
    num_agents = env.num_agents
    logger.info(f"Number of agents: {num_agents}")
    
    # Load policy
    if algorithm_name == "mappo":
        policy_file = "actor.pt"
    else:
        policy_file = "policy.pt"
    
    policy_path = os.path.join(model_dir, policy_file)
    policy = load_policy(policy_path, env.observation_space, env.action_space, device)
    
    # Render settings
    render_mode = 'txt'
    if output_file is None:
        output_file = f"{algorithm_name}_{experiment_name}_seed{seed}.txt.acmi"
    
    # Render episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"{'='*60}")
        
        obs, _ = env.reset()
        if render_mode == 'txt':
            env.render(mode=render_mode, filepath=output_file)
        
        rnn_states = np.zeros((num_agents, 1, 128), dtype=np.float32)
        masks = np.ones((num_agents, 1), dtype=np.float32)
        
        step_rewards = np.zeros(num_agents)
        step_count = 0
        done = False
        
        while not done:
            # Get actions from policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(device)
                rnn_tensor = torch.from_numpy(rnn_states).to(device)
                masks_tensor = torch.from_numpy(masks).to(device)
                
                actions, _, rnn_states_out = policy(
                    obs_tensor,
                    rnn_tensor,
                    masks_tensor,
                    deterministic=deterministic
                )
                
                actions = _t2n(actions)
                rnn_states = _t2n(rnn_states_out)
            
            # Step environment
            obs, _, rewards, dones, infos = env.step(actions)
            
            step_rewards += rewards.squeeze()
            step_count += 1
            
            # Render
            if render_mode == 'txt':
                env.render(mode=render_mode, filepath=output_file)
            
            # Print status
            if step_count % 50 == 0:
                agent_positions = []
                for agent_id in env.agents.keys():
                    state = env.agents[agent_id].get_property_values([
                        env.task.state_var[0],  # lon
                        env.task.state_var[1],  # lat
                        env.task.state_var[2],  # alt
                    ])
                    agent_positions.append(f"{agent_id}: {state}")
                
                logger.info(f"Step {step_count}: rewards={step_rewards}")
                
            # Check termination
            if dones.all() or (max_steps and step_count >= max_steps):
                done = True
            
            # Reset RNN states for terminated agents
            for i, done_flag in enumerate(dones):
                if done_flag:
                    rnn_states[i] = 0
                    masks[i] = 0
        
        episode_rewards.append(step_rewards.mean())
        logger.info(f"Episode reward: {step_rewards.mean():.4f}")
        logger.info(f"Episode steps: {step_count}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Rendering completed!")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Average episode reward: {np.mean(episode_rewards):.4f}")
    logger.info(f"{'='*60}\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Render Formation Flight")
    
    # Required arguments
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--env-name',
        type=str,
        default='Formation',
        help='Environment name'
    )
    parser.add_argument(
        '--scenario-name',
        type=str,
        default='formation/wedge',
        help='Scenario name'
    )
    parser.add_argument(
        '--algorithm-name',
        type=str,
        default='mappo',
        help='Algorithm name'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='v1',
        help='Experiment name'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=5,
        help='Random seed'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='Number of episodes to render'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Max steps per episode'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file for trajectory'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU inference'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic policy (default: deterministic)'
    )
    
    args = parser.parse_args()
    
    # Render
    render_formation(
        model_dir=args.model_dir,
        env_name=args.env_name,
        scenario_name=args.scenario_name,
        algorithm_name=args.algorithm_name,
        experiment_name=args.experiment_name,
        seed=args.seed,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        use_gpu=not args.cpu,
        output_file=args.output_file,
        deterministic=not args.stochastic
    )


if __name__ == '__main__':
    main()
