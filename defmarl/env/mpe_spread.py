import jax.numpy as jnp
import numpy as np

from typing import Optional

from ..utils.graph import EdgeBlock
from ..utils.typing import Action, Reward
from .mpe import MPE, MPEEnvState, MPEEnvGraphsTuple


class MPESpread(MPE):

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_obs": 3,
        "obs_radius": 0.05,
        "default_area_size": 1.5,
        "dist2goal": 0.01
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            max_travel: Optional[float] = None,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = MPESpread.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPESpread, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

    @property
    def reward_min(self) -> float:
        return -((self.area_size * np.sqrt(2)) * 0.01 + 0.001 + 0.0001) * self.max_episode_steps

    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)

        # each goal finds the nearest agent
        reward = jnp.zeros(()).astype(jnp.float32)
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1).min(axis=1)
        reward -= dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward

    def edge_blocks(self, state: MPEEnvState) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.ones((self.num_agents, self.num_agents))
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        if self._params["n_obs"] == 0:
            return [agent_agent_edges, agent_goal_edges]
        obs_pos = state.obs[:, :2]
        poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
        dist = jnp.linalg.norm(poss_diff, axis=-1)
        agent_obs_mask = jnp.less(dist, self._params["comm_radius"])
        id_obs = jnp.arange(self._params["n_obs"]) + self.num_agents * 2
        state_diff = state.agent[:, None, :] - state.obs[None, :, :]
        agent_obs_edges = EdgeBlock(state_diff, agent_obs_mask, id_agent, id_obs)

        return [agent_agent_edges, agent_goal_edges, agent_obs_edges]
