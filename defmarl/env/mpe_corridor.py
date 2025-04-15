import jax.numpy as jnp
import jax.random as jr

from typing import Optional, Tuple

from .mpe import MPEEnvState, MPEEnvGraphsTuple
from .utils import get_node_goal_rng
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from .mpe_spread import MPESpread


class MPECorridor(MPESpread):
    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "default_area_size": 1.0,
        "dist2goal": 0.01,
        "n_obs": 2,
        "corridor_width": 0.2,
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
        area_size = MPECorridor.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPESpread, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        if self.params["n_obs"] != 2:
            self.params["n_obs"] = 2
            print("WARNING: n_obs is set to 2 for MPECorridor.")
        # solve for the radius of the obstacle
        self.params["obs_radius"] = (self.area_size - self.params["corridor_width"]) / 4

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.params["car_radius"],
            None,
            (self.area_size - self.params["obs_radius"] * 2) / 2 - 1.5 * self.params["car_radius"],
            self.max_travel,
        )
        goals = goals + jnp.array(
            [0.,
             self.area_size - (self.area_size - self.params["obs_radius"] * 2) / 2 + 1.5 * self.params["car_radius"]])

        # add corridor obstacles
        obs = jnp.array([[self.params["obs_radius"], self.area_size / 2],
                         [self.area_size - self.params["obs_radius"], self.area_size / 2]])

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros_like(goals)], axis=1)
        obs = jnp.concatenate([obs, jnp.zeros_like(obs)], axis=1)

        env_state = MPEEnvState(states, goals, obs)

        return self.get_graph(env_state)

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0.0, 0.0, -1.0, -1.0])
        upper_lim = jnp.array([self.area_size, self.area_size * 2, 1.0, 1.0])
        return lower_lim, upper_lim

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

        # agent - obs connection: always connected
        if self._params["n_obs"] == 0:
            return [agent_agent_edges, agent_goal_edges]
        obs_pos = state.obs[:, :2]
        poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
        dist = jnp.linalg.norm(poss_diff, axis=-1)
        agent_obs_mask = jnp.less(dist, self._params["comm_radius"] * 100)
        id_obs = jnp.arange(self._params["n_obs"]) + self.num_agents * 2
        state_diff = state.agent[:, None, :] - state.obs[None, :, :]
        agent_obs_edges = EdgeBlock(state_diff, agent_obs_mask, id_agent, id_obs)

        return [agent_agent_edges, agent_goal_edges, agent_obs_edges]
