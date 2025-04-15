import jax.numpy as jnp
import jax.random as jr

from jax.lax import while_loop
from typing import Optional, Tuple

from .mpe import MPEEnvState, MPEEnvGraphsTuple
from .utils import get_node_goal_rng
from ..utils.graph import EdgeBlock, GraphsTuple
from ..utils.typing import Array, Cost, State, PRNGKey
from .mpe_spread import MPESpread


class MPEConnectSpread(MPESpread):
    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "default_area_size": 1.0,
        "dist2goal": 0.01,
        "n_obs": 1,
        "obs_radius": 0.25,
        "connect_radius": 0.45,
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
        area_size = MPEConnectSpread.PARAMS["default_area_size"] if area_size is None else area_size
        if params is None:
            params = self.PARAMS
        super(MPESpread, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        if self.params["n_obs"] != 1:
            self.params["n_obs"] = 1
            print("WARNING: n_obs is set to 1 for MPEConnectSpread.")

    @property
    def n_cost(self) -> int:
        return 3

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "connectivity"

    def reset(self, key: Array) -> GraphsTuple:
        # generate agent and goal such that they are connected
        def non_valid_node(inp: Tuple[PRNGKey, Array, Array]):
            _, state_nodes, goal_nodes = inp
            dist = jnp.linalg.norm(jnp.expand_dims(state_nodes, 1) - jnp.expand_dims(state_nodes, 0), axis=-1)
            dist += jnp.eye(self.num_agents) * 1e6
            min_dist = jnp.min(dist, axis=1)
            non_connect_agent = (min_dist > self.params["connect_radius"]).any()
            collide_agent = (min_dist < 2 * self.params["car_radius"]).any()

            dist = jnp.linalg.norm(jnp.expand_dims(goal_nodes, 1) - jnp.expand_dims(goal_nodes, 0), axis=-1)
            dist += jnp.eye(self.num_agents) * 1e6
            min_dist = jnp.min(dist, axis=1)
            non_connect_goal = (min_dist > self.params["connect_radius"]).any()

            return non_connect_agent | collide_agent | non_connect_goal

        def get_node(inp: Tuple[PRNGKey, Array, Array]):
            this_key, use_key = jr.split(inp[0], 2)

            # randomly generate agent and goal
            state_nodes, goal_nodes = get_node_goal_rng(
                use_key,
                self.area_size,
                2,
                self.num_agents,
                2.3 * self.params["car_radius"],
                None,
                (self.area_size - self.params["obs_radius"] * 2) / 2 - 1.5 * self.params["car_radius"],
                self.max_travel
            )
            goal_nodes += jnp.array([0., self.area_size -
                                     (self.area_size - self.params["obs_radius"] * 2) / 2
                                     + 1.5 * self.params["car_radius"]])
            return this_key, state_nodes, goal_nodes

        states = jnp.zeros((self.num_agents, 2))
        goals = jnp.zeros((self.num_agents, 2))

        key, states, goals = while_loop(cond_fun=non_valid_node, body_fun=get_node, init_val=(key, states, goals))

        # add a large obstacle
        obs_key, key = jr.split(key, 2)
        obs_x = jr.uniform(obs_key, (1,), minval=self.params["obs_radius"],
                           maxval=self.area_size - self.params["obs_radius"])
        obs = jnp.array([[obs_x[0], self.area_size / 2]])

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros_like(goals)], axis=1)
        obs = jnp.concatenate([obs, jnp.zeros_like(obs)], axis=1)

        env_state = MPEEnvState(states, goals, obs)

        return self.get_graph(env_state)

    def get_cost(self, graph: MPEEnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])[:, :2]

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # connectivity cost
        connect_cost: Array = (min_dist - self.params["connect_radius"]).max()
        connect_cost = connect_cost.repeat(self.num_agents)

        # collision between agents and obstacles
        if self.params["n_obs"] == 0:
            obs_cost = jnp.zeros(self.num_agents)
        else:
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacles, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obs_cost: Array = self.params["car_radius"] + self.params["obs_radius"] - min_dist

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None], connect_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        assert cost.shape == (self.num_agents, self.n_cost)

        return cost

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
