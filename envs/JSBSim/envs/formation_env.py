import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.formation_task import FormationTask


class FormationEnv(BaseEnv):
    """
    FormationEnv is a multi-agent cooperative environment for learning formation flight.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self._create_records = False

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'formation':
            self.task = FormationTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname for FormationEnv: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics.

        Args:
            action (np.ndarray): the agents' actions

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        
        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        # Compute rewards
        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        # Compute dones
        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Get all agents' observations."""
        obs = {}
        for agent_id in self.agents.keys():
            obs[agent_id] = self.task.get_obs(self, agent_id)
        return obs

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get all agents' state (for centralized critic if needed)."""
        state = {}
        for agent_id in self.agents.keys():
            state[agent_id] = self.task.get_obs(self, agent_id)
        return state

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack dict of agent data into array for vectorized env wrapper."""
        agent_ids = sorted(data.keys())
        return np.array([data[agent_id] for agent_id in agent_ids])

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack array back to dict of agent data."""
        agent_ids = sorted(self.agents.keys())
        return {agent_id: data[i] for i, agent_id in enumerate(agent_ids)}
