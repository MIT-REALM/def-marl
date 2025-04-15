import numpy as np
import jax.numpy as jnp
import functools as ft
import jax
import jax.random as jr

from typing import Tuple
from jax.lax import while_loop

from ..utils.typing import Array, Radius, BoolScalar, Pos, PRNGKey
from .obstacle import Obstacle


def inside_obstacles(points: Pos, obstacles: Obstacle = None, r: Radius = 0.) -> BoolScalar:
    """
    points: (n, n_dim) or (n_dim, )
    obstacles: tree_stacked obstacles.

    Returns: (n, ) or (,). True if in collision, false otherwise.
    """
    if obstacles is None:
        if points.ndim == 1:
            return jnp.zeros((), dtype=bool)
        return jnp.zeros(points.shape[0], dtype=bool)

    # one point inside one obstacle
    def inside(point: Pos, obstacle: Obstacle):
        return obstacle.inside(point, r)

    # one point inside any obstacle
    def inside_any(point: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(inside, point))(obstacle).max()

    # any point inside any obstacle
    if points.ndim == 1:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros((), dtype=bool)
        is_in = inside_any(points, obstacles)
    else:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros(points.shape[0], dtype=bool)
        is_in = jax.vmap(ft.partial(inside_any, obstacle=obstacles))(points)

    return is_in


def get_node_goal_rng(
        key: PRNGKey,
        side_length: float,
        dim: int,
        n: int,
        min_dist: float,
        obstacles: Obstacle = None,
        side_length_y: float = None,
        max_travel: float = None
) -> [Pos, Pos]:
    max_iter = 1024  # maximum number of iterations to find a valid initial state/goal
    states = jnp.zeros((n, dim))
    goals = jnp.zeros((n, dim))
    side_length_y = side_length if side_length_y is None else side_length_y

    def get_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, this_key, _, all_nodes = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        return i_iter, this_key, jr.uniform(use_key, (dim,),
                                            minval=0, maxval=jnp.array([side_length, side_length_y])), all_nodes

    def non_valid_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, _, node, all_nodes = reset_input
        dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(node, obstacles, r=min_dist / 2)
        valid = ~(collide | inside) | (i_iter >= max_iter)
        return ~valid

    def get_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals
        i_iter, this_key, _, agent, all_goals = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        if max_travel is None:
            return (i_iter, this_key,
                    jr.uniform(use_key, (dim,), minval=0, maxval=jnp.array([side_length, side_length_y])),
                    agent, all_goals)
        else:
            return i_iter, this_key, jr.uniform(
                use_key, (dim,), minval=-max_travel, maxval=max_travel) + agent, agent, all_goals

    def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals
        i_iter, _, goal, agent, all_goals = reset_input
        dist_min = jnp.linalg.norm(all_goals - goal, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(goal, obstacles, r=min_dist / 2)
        outside = jnp.any(goal < 0) | jnp.any(goal > side_length)
        if max_travel is None:
            too_long = np.array(False, dtype=bool)
        else:
            too_long = jnp.linalg.norm(goal - agent) > max_travel
        valid = (~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
        out = ~valid
        assert out.shape == tuple() and out.dtype == jnp.bool_
        return out

    def reset_body(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        agent_key, goal_key, this_key = jr.split(this_key, 3)
        agent_candidate = jr.uniform(agent_key, (dim,), minval=0, maxval=jnp.array([side_length, side_length_y]))
        n_iter_agent, _, agent_candidate, _ = while_loop(
            cond_fun=non_valid_node, body_fun=get_node,
            init_val=(0, agent_key, agent_candidate, all_states)
        )
        all_states = all_states.at[agent_id].set(agent_candidate)

        if max_travel is None:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=jnp.array([side_length, side_length_y]))
        else:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=max_travel) + agent_candidate

        n_iter_goal, _, goal_candidate, _, _ = while_loop(
            cond_fun=non_valid_goal, body_fun=get_goal,
            init_val=(0, goal_key, goal_candidate, agent_candidate, all_goals)
        )
        all_goals = all_goals.at[agent_id].set(goal_candidate)
        agent_id += 1

        # if no solution is found, start over
        agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
        all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
        all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals

        return agent_id, this_key, all_states, all_goals

    def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        return agent_id < n

    _, _, states, goals = while_loop(
        cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))

    return states, goals
