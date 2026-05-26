import numpy as np
from gymnasium import spaces
from typing import Tuple

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, PostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Timeout
from ..utils.utils import LLA2NEU


class FormationTask(SingleCombatTask):
    """
    Formation Flight Task:
    - Leader follows a predefined ㄱ-shaped trajectory (or other path)
    - Followers learn to maintain wedge formation while following the leader
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
        ]
        
        self.termination_conditions = [
            ExtremeState(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        
        # Formation parameters
        self.formation_type = getattr(config, 'formation_type', 'wedge')  # wedge, line, echelon
        self.num_followers = getattr(config, 'num_followers', 3)
        self.total_agents = 1 + self.num_followers  # 1 leader + followers
        
        # Wedge formation reference distances (meters)
        # For a 3-follower wedge: [-d, 0] (left), [0, d] (center), [-d, 0] (right)
        self.formation_spacing = getattr(config, 'formation_spacing', 100)  # meters
        self.formation_lateral = getattr(config, 'formation_lateral', 80)    # meters
        
        # Leader trajectory waypoints (ㄱ shape)
        # [lon, lat, alt] - initial state
        self.leader_waypoints = None
        self.leader_wp_idx = 0
        self.leader_current_wp = None
        
    @property
    def num_agents(self) -> int:
        return self.total_agents
    
    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. longitude
            c.position_lat_geod_deg,            # 1. latitude
            c.position_h_sl_m,                  # 2. altitude
            c.attitude_roll_rad,                # 3. roll
            c.attitude_pitch_rad,               # 4. pitch
            c.attitude_heading_true_rad,        # 5. yaw
            c.velocities_v_north_mps,           # 6. v_north
            c.velocities_v_east_mps,            # 7. v_east
            c.velocities_v_down_mps,            # 8. v_down
            c.velocities_u_mps,                 # 9. v_body_x
            c.velocities_v_mps,                 # 10. v_body_y
            c.velocities_w_mps,                 # 11. v_body_z
            c.velocities_vc_mps,                # 12. vc
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]
    
    def load_observation_space(self):
        # Observation for each agent:
        # - Ego state (9): alt, roll_sin/cos, pitch_sin/cos, v_x/y/z, vc
        # - Relative to leader (6): dpos_ned, dvel_ned  
        # - Relative to each follower (6 each): dpos_ned, dvel_ned
        obs_per_agent = 9 + 6 * self.num_agents  # ego (9) + leader (6) + others
        self.obs_length = obs_per_agent
        self.observation_space = spaces.Box(
            low=-10, high=10., shape=(obs_per_agent,), dtype=np.float32
        )
        self.share_observation_space = spaces.Box(
            low=-10, high=10., 
            shape=(self.num_agents * obs_per_agent,), 
            dtype=np.float32
        )
    
    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])
    
    def reset(self, env):
        super().reset(env)
        self.leader_wp_idx = 0
        self._generate_leader_waypoints(env)
    
    def step(self, env):
        """Called once per environment step to update leader trajectory."""
        super().step(env)
        # Update leader waypoint target (if needed)
        # This could transition between waypoints
    
    def _generate_leader_waypoints(self, env):
        """
        Generate L-shaped (ㄱ) trajectory for leader.
        Phase 1: Fly straight ahead
        Phase 2: Turn 90 degrees and fly again
        """
        # Get initial leader position
        leader_state = np.array(env.agents[0].get_property_values(self.state_var[:3]))
        lon0, lat0, alt0 = leader_state
        
        # Convert to NED for easier planning
        ned0 = LLA2NEU(lon0, lat0, alt0, env.center_lon, env.center_lat, env.center_alt)
        north0, east0, down0 = ned0
        
        # L-shaped waypoints (in NED, relative to initial position)
        # Phase 1: fly 2000m north
        wp1_ned = np.array([north0 + 2000, east0, down0])
        # Phase 2: fly 2000m east from there
        wp2_ned = np.array([north0 + 2000, east0 + 2000, down0])
        
        self.leader_waypoints = [wp1_ned, wp2_ned]
        self.leader_current_wp = wp1_ned
    
    def get_obs(self, env, agent_id):
        """Get observation for agent_id.
        
        Observation structure:
        [0-8]: Ego state
        [9:]: Relative observations to all other agents
        """
        norm_obs = np.zeros(self.obs_length, dtype=np.float32)
        
        # Get ego state
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        
        # Ego observation (normalized)
        norm_obs[0] = ego_state[2] / 5000            # altitude (5km)
        norm_obs[1] = np.sin(ego_state[3])           # roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # pitch_cos
        norm_obs[5] = ego_state[9] / 340             # v_body_x (mach)
        norm_obs[6] = ego_state[10] / 340            # v_body_y (mach)
        norm_obs[7] = ego_state[11] / 340            # v_body_z (mach)
        norm_obs[8] = ego_state[12] / 340            # vc (mach)
        
        # Relative observations to all other agents (leader first if not leader)
        offset = 9
        
        # If this is a follower, observe leader
        if agent_id != 0:
            leader_state = np.array(env.agents[0].get_property_values(self.state_var))
            leader_ned = LLA2NEU(*leader_state[:3], env.center_lon, env.center_lat, env.center_alt)
            
            # Relative position and velocity (NED)
            dpos = leader_ned - ego_ned
            dvel = leader_state[6:9] - ego_state[6:9]
            
            norm_obs[offset:offset+3] = dpos / 1000   # position (km)
            norm_obs[offset+3:offset+6] = dvel / 100  # velocity (100 m/s)
            offset += 6
        
        # Observe other followers
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                other_state = np.array(env.agents[other_id].get_property_values(self.state_var))
                other_ned = LLA2NEU(*other_state[:3], env.center_lon, env.center_lat, env.center_alt)
                
                dpos = other_ned - ego_ned
                dvel = other_state[6:9] - ego_state[6:9]
                
                norm_obs[offset:offset+3] = dpos / 1000
                norm_obs[offset+3:offset+6] = dvel / 100
                offset += 6
        
        return norm_obs.clip(-10, 10)
    
    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value."""
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.0      # aileron
        norm_act[1] = action[1] / 20 - 1.0      # elevator
        norm_act[2] = action[2] / 20 - 1.0      # rudder
        norm_act[3] = action[3] / 58 + 0.4      # throttle
        return norm_act
    
    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """Compute reward for agent.
        
        For followers: formation maintenance reward
        For leader: altitude/posture reward
        """
        if agent_id == 0:
            # Leader: use base reward functions (altitude, posture)
            reward, info = super().get_reward(env, agent_id, info)
        else:
            # Follower: formation maintenance reward
            reward = self._formation_reward(env, agent_id)
        
        return reward, info
    
    def _formation_reward(self, env, agent_id) -> float:
        """
        Compute formation maintenance reward for follower.
        
        Reward based on:
        1. Distance to formation slot
        2. Velocity alignment with leader
        3. Altitude maintenance
        """
        follower_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        follower_ned = LLA2NEU(*follower_state[:3], env.center_lon, env.center_lat, env.center_alt)
        
        leader_state = np.array(env.agents[0].get_property_values(self.state_var))
        leader_ned = LLA2NEU(*leader_state[:3], env.center_lon, env.center_lat, env.center_alt)
        
        dpos = follower_ned - leader_ned
        
        # Formation slot for this follower (wedge formation example)
        slot_offset = self._get_formation_slot(agent_id)
        
        # Distance error from formation slot
        formation_error = np.linalg.norm(dpos[:2] - slot_offset)
        
        # Reward: closer to formation = higher reward
        formation_reward = -formation_error / 200  # normalize
        
        # Velocity alignment
        dvel = follower_state[6:9] - leader_state[6:9]
        vel_alignment = -np.linalg.norm(dvel) / 100
        
        # Combined reward
        reward = formation_reward * 0.7 + vel_alignment * 0.3
        
        return reward
    
    def _get_formation_slot(self, agent_id) -> np.ndarray:
        """Get the NED offset for formation slot of agent_id.
        
        Wedge formation example:
        - Agent 0 (leader): [0, 0]
        - Agent 1: [-lateral, spacing]
        - Agent 2: [0, spacing]
        - Agent 3: [lateral, spacing]
        """
        if agent_id == 0:
            return np.array([0, 0])
        
        if self.formation_type == 'wedge':
            if self.num_followers == 3:
                slots = [
                    np.array([-self.formation_lateral, self.formation_spacing]),
                    np.array([0, self.formation_spacing]),
                    np.array([self.formation_lateral, self.formation_spacing]),
                ]
                return slots[agent_id - 1]
        
        # Default: line formation
        return np.array([0, agent_id * self.formation_spacing])
