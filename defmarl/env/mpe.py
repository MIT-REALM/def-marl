import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod, abstractproperty

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Reward, State, AgentState
from ..utils.utils import tree_index, MutablePatchCollection, save_anim
from .base import MultiAgentEnv
from .utils import get_node_goal_rng


class MPEEnvState(NamedTuple):
    agent: State
    goal: State
    obs: State

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEEnvGraphsTuple = GraphsTuple[State, MPEEnvState]


class MPE(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_obs": 3,
        "obs_radius": 0.05,
        "default_area_size": 1.0,
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
        area_size = MPE.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPE, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 7  # state dim (4) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @abstractproperty
    def reward_min(self) -> float:
        pass

    @property
    def reward_max(self) -> float:
        return 0.5

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.params["car_radius"],
            None,
            None,
            self.max_travel
        )

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
        goals = jnp.concatenate([goals, jnp.zeros_like(goals)], axis=1)
        if self.params["n_obs"] > 0:
            obs = jnp.concatenate([obs, jnp.zeros_like(obs)], axis=1)
        else:
            obs = None

        env_state = MPEEnvState(states, goals, obs)

        return self.get_graph(env_state)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: MPEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"]) if self.params["n_obs"] > 0 else None

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

    @abstractmethod
    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: MPEEnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])[:, :2]

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params["n_obs"] == 0:
            obs_cost = jnp.zeros(self.num_agents)
        else:
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacles, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obs_cost: Array = self.params["car_radius"] + self.params["obs_radius"] - min_dist

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)

        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            n_goal: int = None,
            **kwargs
    ) -> None:
        n_goal = self.num_agents if n_goal is None else n_goal

        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_xlim(0., self.area_size)
        ax.set_ylim(0., self.area_size)
        ax.set(aspect="equal")
        plt.axis("off")
        if viz_opts is None:
            viz_opts = {}

        # plot the first frame
        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obs_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        if self.params["n_obs"] > 0:
            obs_pos = graph0.type_states(type_idx=MPE.OBS, n_type=self.params["n_obs"])[:, :2]  # [n_obs, 2]
            obs_plots = [plt.Circle((float(obs_pos[i, 0]), float(obs_pos[i, 1])), self.params["obs_radius"],
                                    color=obs_color) for i in range(len(obs_pos))]
            obs_col = MutablePatchCollection(obs_plots, match_original=True, zorder=5)
            ax.add_collection(obs_col)

        # plot agents and goals
        n_color = [agent_color] * self.num_agents + [goal_color] * n_goal
        n_pos = np.array(graph0.states[:self.num_agents + n_goal, :2]).astype(np.float32)
        agent_circs = [plt.Circle((float(n_pos[i, 0]), float(n_pos[i, 1])), self.params["car_radius"],
                                  color=n_color[i], linewidth=0.0)
                       for i in range(self.num_agents + n_goal)]
        agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        ax.add_collection(agent_col)

        # plot edges
        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goal + self.params["n_obs"], axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goal)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)

        # texts
        text_font_opts = dict(
            size=16,
            color="k",
            family="cursive",
            weight="normal",
            transform=ax.transAxes,
        )
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

        # add agent labels
        label_font_opts = dict(
            size=20,
            color="k",
            family="cursive",
            weight="normal",
            ha="center",
            va="center",
            transform=ax.transData,
            clip_on=True,
            zorder=7,
        )
        agent_labels = [ax.text(float(n_pos[ii, 0]), float(n_pos[ii, 1]), f"{ii}", **label_font_opts)
                        for ii in range(self.num_agents)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        # init function for animation
        def init_fn() -> list[plt.Artist]:
            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2]

            # update agent positions
            for ii in range(self.num_agents):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))

            # update edges
            e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goal + self.params["n_obs"], axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
            e_is_goal_t = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents + n_goal)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            edge_col.set_segments(e_lines_t)
            edge_col.set_colors(e_colors_t)

            # update agent labels
            for ii in range(self.num_agents):
                agent_labels[ii].set_position(n_pos_t[ii])

            # update cost and safe labels
            if kk < len(rollout.costs):
                all_costs = ""
                for i_cost in range(rollout.costs[kk].shape[1]):
                    all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
                all_costs = all_costs[:-2]
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            else:
                cost_text.set_text("")
            if kk < len(Ta_is_unsafe):
                a_is_unsafe = Ta_is_unsafe[kk]
                unsafe_idx = np.where(a_is_unsafe)[0]
                safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
            else:
                safe_text[0].set_text("Unsafe: {}")

            kk_text.set_text("kk={:04}".format(kk))

            # Update the z text.
            z_text.set_text(f"z: {rollout.zs[kk]}")

            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    @abstractmethod
    def edge_blocks(self, state: MPEEnvState) -> list[EdgeBlock]:
        pass

    def get_graph(self, env_state: MPEEnvState) -> MPEEnvGraphsTuple:
        # node features
        # states
        node_feats = jnp.zeros((self.num_agents * 2 + self.params["n_obs"], self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, :self.state_dim].set(env_state.goal)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents * 2:, :self.state_dim].set(env_state.obs)

        # indicators
        node_feats = node_feats.at[:self.num_agents, 6].set(1.0)
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 5].set(1.0)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents * 2:, 4].set(1.0)

        # node type
        node_type = -jnp.ones((self.num_agents * 2 + self.params["n_obs"],), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(MPE.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(MPE.GOAL)
        if self.params["n_obs"] > 0:
            node_type = node_type.at[self.num_agents * 2:].set(MPE.OBS)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if self.params["n_obs"] > 0:
            states = jnp.concatenate([states, env_state.obs], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0.0, 0.0, -1.0, -1.0])
        upper_lim = jnp.array([self.area_size, self.area_size, 1.0, 1.0])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        cost = self.get_cost(graph)
        return jnp.any(cost >= 0.0, axis=-1)
