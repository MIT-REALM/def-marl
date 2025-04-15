import pathlib
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import jax

from typing import Tuple, Optional

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward
from .mpe import MPE, MPEEnvState, MPEEnvGraphsTuple
from .utils import get_node_goal_rng


class MPEFormation(MPE):

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
        area_size = MPEFormation.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPEFormation, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate agent
        states, _ = get_node_goal_rng(
            key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.params["car_radius"],
            None,
            None,
            self.max_travel
        )

        # generate a landmark
        R = self.params["comm_radius"]
        landmark_key, key = jr.split(key)
        landmark = jr.uniform(landmark_key, (1, 2),
                              minval=R + 2 * self.params['car_radius'],
                              maxval=self.area_size - R - 2 * self.params['car_radius'])
        goals = self.landmark2goal(landmark, R)

        # randomly generate obstacles
        def get_obs(inp):
            this_key, _ = inp
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,),
                                        minval=self.params['car_radius'] * 3,
                                        maxval=self.area_size - self.params['car_radius'] * 3)

        def non_valid_obs(inp):
            _, this_obs = inp
            dist_min_agents = jnp.linalg.norm(states - this_obs, axis=1).min()
            dist_min_goals = jnp.linalg.norm(goals - this_obs, axis=1).min()
            collide_agent = dist_min_agents <= self.params["car_radius"] + self.params["obs_radius"]
            collide_goal = dist_min_goals <= self.params["car_radius"] * 2 + self.params["obs_radius"]
            out_region = (jnp.any(this_obs < self.params["car_radius"] * 3) |
                          jnp.any(this_obs > self.area_size - self.params["car_radius"] * 3))
            return collide_agent | collide_goal | out_region

        def get_valid_obs(carry, inp):
            this_key = inp
            use_key, this_key = jr.split(this_key, 2)
            obs_candidate = jr.uniform(use_key, (2,), minval=0, maxval=self.area_size)
            _, valid_obs = jax.lax.while_loop(non_valid_obs, get_obs, (this_key, obs_candidate))
            return carry, valid_obs

        obs_keys = jr.split(key, self.params["n_obs"])
        _, obs = jax.lax.scan(get_valid_obs, None, obs_keys)

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        landmark = jnp.concatenate([landmark, jnp.zeros_like(landmark)], axis=1)
        obs = jnp.concatenate([obs, jnp.zeros_like(obs)], axis=1)

        env_state = MPEEnvState(states, landmark, obs)

        return self.get_graph(env_state)

    @property
    def reward_min(self) -> float:
        return -((self.area_size * np.sqrt(2)) * 0.01 + 0.001 + 0.0001) * self.max_episode_steps

    def landmark2goal(self, landmarks: Pos2d, R: float) -> Pos2d:
        assert landmarks.shape == (1, 2)
        thetas = jnp.linspace(0, 2 * jnp.pi, self.num_agents + 1)[:-1]
        goals = landmarks + R * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
        return goals

    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        landmark = graph.type_states(type_idx=1, n_type=1)[:, :2]
        goals = self.landmark2goal(landmark, self.params['comm_radius'])

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

    def step(
            self, graph: MPEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=1)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])

        # calculate next graph
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MPEEnvState(next_agent_states, goals, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        return self.get_graph(next_env_state), reward, cost, done, info

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
        id_goal = jnp.arange(self.num_agents, self.num_agents + 1)
        agent_goal_mask = jnp.ones((self.num_agents, 1))
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        obs_pos = state.obs[:, :2]
        poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
        dist = jnp.linalg.norm(poss_diff, axis=-1)
        agent_obs_mask = jnp.less(dist, self._params["comm_radius"])
        id_obs = jnp.arange(self._params["n_obs"]) + self.num_agents + 1
        state_diff = state.agent[:, None, :] - state.obs[None, :, :]
        agent_obs_edges = EdgeBlock(state_diff, agent_obs_mask, id_agent, id_obs)

        return [agent_agent_edges, agent_goal_edges, agent_obs_edges]

    def get_graph(self, env_state: MPEEnvState) -> MPEEnvGraphsTuple:
        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + 1 + self.params["n_obs"], self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + 1, :self.state_dim].set(env_state.goal)
        node_feats = node_feats.at[self.num_agents + 1:, :self.state_dim].set(env_state.obs)

        # indicators
        node_feats = node_feats.at[:self.num_agents, 6].set(1.0)
        node_feats = node_feats.at[self.num_agents: self.num_agents + 1, 5].set(1.0)
        node_feats = node_feats.at[self.num_agents + 1:, 4].set(1.0)

        # node type
        node_type = -jnp.ones((self.num_agents + 1 + self.params["n_obs"],), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(MPE.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + 1].set(MPE.GOAL)
        node_type = node_type.at[self.num_agents + 1:].set(MPE.OBS)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal, env_state.obs], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            **kwargs
    ) -> None:
        return super().render_video(rollout, video_path, Ta_is_unsafe, viz_opts, n_goal=1, **kwargs)
