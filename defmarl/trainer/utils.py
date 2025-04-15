import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket

from typing import Callable, TYPE_CHECKING, Optional

from ..utils.typing import PRNGKey, Array
from ..utils.graph import GraphsTuple
from .data import Rollout


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        gamma: float,
        init_graph: Optional[GraphsTuple] = None
) -> Rollout:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, Array, RNN_States, PRNGKey] -> [Action, LogPi, RNN_States]
    init_rnn_state: Array
    key: PRNGKey
    gamma: float, discount factor

    Returns
    -------
    data: Rollout
    """
    key_x0, key_z0, key = jax.random.split(key, 3)
    if init_graph is None:
        init_graph = env.reset(key_x0)
    z0 = jax.random.uniform(key_z0, (1, 1), minval=-env.reward_max, maxval=-env.reward_min)

    z_key, key = jax.random.split(key, 2)
    rng = jax.random.uniform(z_key, (1, 1))
    z0 = jnp.where(rng > 0.7, -env.reward_max, z0)  # use z min
    z0 = jnp.where(rng < 0.2, -env.reward_min, z0)  # use z max

    z0 = jnp.repeat(z0, env.num_agents, axis=0)

    def body(data, key_):
        graph, rnn_state, z = data
        action, log_pi, new_rnn_state = actor(graph, z, rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)

        # z dynamics
        z_next = (z + reward) / gamma
        z_next = jnp.clip(z_next, -env.reward_max, -env.reward_min)

        return ((next_graph, new_rnn_state, z_next),
                (graph, action, rnn_state, reward, cost, done, log_pi, next_graph, z))

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs, zs) = (
        jax.lax.scan(body, (init_graph, init_rnn_state, z0), keys, length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs, zs)
    return rollout_data


def test_rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_actor_rnn_state: Array,
        key: PRNGKey,
        init_Vh_rnn_state: Optional[Array] = None,
        z_fn: Optional[Callable] = None,
        stochastic: bool = False
):
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)
    z0 = jax.random.uniform(key, (env.num_agents, 1), minval=-env.reward_max, maxval=-env.reward_min)

    def body(data, key_):
        graph, actor_rnn_state, Vh_rnn_state = data
        if z_fn is not None:
            z, Vh_rnn_state = z_fn(graph, Vh_rnn_state)
            z_max = np.max(z, axis=0)
            z = jnp.repeat(z_max[None], env.num_agents, axis=0)
        else:
            z = z0
        if not stochastic:
            action, actor_rnn_state = actor(graph, z, actor_rnn_state)
        else:
            action, actor_rnn_state = actor(graph, z, actor_rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        return ((next_graph, actor_rnn_state, Vh_rnn_state),
                (graph, action, actor_rnn_state, reward, cost, done, None, next_graph, z))

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs, zs) = (
        jax.lax.scan(body,
                     (init_graph, init_actor_rnn_state, init_Vh_rnn_state),
                     keys,
                     length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs, zs)
    return rollout_data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def is_connected():
    try:
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close()
        return True
    except OSError:
        pass
    print('No internet connection')
    return False
