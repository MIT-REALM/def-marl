import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle
import flax.linen as nn

from typing import Optional, Tuple
from flax.training.train_state import TrainState

from ..utils.typing import Params, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import jax_vmap, tree_index
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..env.base import MultiAgentEnv
from ..algo.module.value import ValueNet
from .utils import compute_gae
from .informarl import InforMARL


class InforMARLLagr(InforMARL):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            cost_weight: float = 0.,
            actor_gnn_layers: int = 2,
            critic_gnn_layers: int = 2,
            gamma: float = 0.99,
            lr_actor: float = 1e-5,
            lr_critic: float = 1e-5,
            batch_size: int = 8192,
            epoch_ppo: int = 1,
            clip_eps: float = 0.25,
            gae_lambda: float = 0.95,
            coef_ent: float = 1e-2,
            max_grad_norm: float = 2.0,
            seed: int = 0,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            rnn_step: int = 16,
            rollout_length: Optional[int] = None,
            use_lstm: bool = False,
            lagr_init: float = 3.5,
            lr_lagr: float = 1e-7,
            **kwargs
    ):
        super(InforMARLLagr, self).__init__(
            env, node_dim, edge_dim, state_dim, action_dim, n_agents, cost_weight, actor_gnn_layers, critic_gnn_layers,
            gamma, lr_actor, lr_critic, batch_size, epoch_ppo, clip_eps, gae_lambda, coef_ent, max_grad_norm, seed,
            use_rnn, rnn_layers, rnn_step, rollout_length, use_lstm
        )

        # set hyperparameters
        self.lagr_init = lagr_init
        self.lr_lagr = lr_lagr

        # cost value function
        self.cost_critic = ValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            n_out=env.n_cost,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=1,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            use_ef=False,
            decompose=True,
            use_global_info=True
        )

        # initialize the rnn state
        rnn_state_key, key = jr.split(self.key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_cost_rnn_state = jax_vmap(self.cost_critic.initialize_carry)(rnn_state_key)  # (n_agents, rnn_state_dim)
        if type(init_cost_rnn_state) is tuple:
            init_cost_rnn_state = jnp.stack(init_cost_rnn_state, axis=1)  # (n_agents, n_carries, rnn_state_dim)
        else:
            init_cost_rnn_state = jnp.expand_dims(init_cost_rnn_state, axis=1)
        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_cost_rnn_state = init_cost_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        cost_critic_key, key = jr.split(key)
        cost_critic_params = self.cost_critic.net.init(cost_critic_key, self.nominal_graph, self.init_cost_rnn_state,
                                                       self.n_agents, self.nominal_z)
        cost_critic_optim = optax.adam(learning_rate=lr_critic)
        self.cost_critic_optim = optax.apply_if_finite(cost_critic_optim, 1_000_000)
        self.cost_critic_train_state = TrainState.create(
            apply_fn=self.cost_critic.get_value,
            params=cost_critic_params,
            tx=self.cost_critic_optim
        )

        # initialize the lagrange multiplier
        self.lagr = jnp.ones((self.n_agents, self._env.n_cost)) * self.lagr_init

    @property
    def config(self) -> dict:
        return super().config | {
            "lagr_init": self.lagr_init,
            "lr_lagr": self.lr_lagr
        }

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        update_info = {}
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        for i_epoch in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, rollout.dones.shape[1] // self.rnn_step))
            batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // (self.batch_size // rollout.dones.shape[1])))
            critic_train_state, cost_critic_train_state, policy_train_state, lagr, update_info = self.update_inner(
                self.critic_train_state,
                self.cost_critic_train_state,
                self.policy_train_state,
                self.lagr,
                rollout,
                batch_idx,
                rnn_chunk_ids
            )
            self.critic_train_state = critic_train_state
            self.policy_train_state = policy_train_state
            self.cost_critic_train_state = cost_critic_train_state
            self.lagr = lagr
        return update_info

    def scan_cost_value(
            self, graphs: GraphsTuple, init_rnn_state: Array, cost_critic_params: Params
    ) -> Tuple[Array, Array, Array]:

        def body_(rnn_state, graph):
            cost_value, new_rnn_state = self.cost_critic.get_value(cost_critic_params, graph, rnn_state)
            return new_rnn_state, (cost_value, rnn_state)

        final_rnn_state, (cost_value_rollout, rnn_states) = jax.lax.scan(body_, init_rnn_state, graphs)

        return cost_value_rollout, rnn_states, final_rnn_state

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            critic_train_state: TrainState,
            cost_critic_train_state: TrainState,
            policy_train_state: TrainState,
            lagr: Array,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
    ) -> Tuple[TrainState, TrainState, TrainState, Array, dict]:
        # rollout: (n_env, T, n_agent, ...)
        b, T, a, _ = rollout.zs.shape

        # calculate values and next_values
        bT_V, value_rnn_states, final_rnn_states_V = jax_vmap(
            ft.partial(self.scan_value,
                       init_rnn_state=self.init_value_rnn_state,
                       critic_params=critic_train_state.params)
        )(rollout.graph)
        bT_V = bT_V.squeeze(-1).squeeze(-1)

        # calculate cost values
        bTah_Vc, cost_rnn_states, final_rnn_states_Vc = jax_vmap(
            ft.partial(self.scan_cost_value,
                       init_rnn_state=self.init_cost_rnn_state,
                       cost_critic_params=cost_critic_train_state.params)
        )(rollout.graph)

        def final_value_fn(graph, rnn_state_V, rnn_state_Vc):
            value, _ = self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state_V)
            cost_value, _ = self.cost_critic.get_value(
                cost_critic_train_state.params, tree_index(graph, -1), rnn_state_Vc)
            return value.squeeze(), cost_value

        final_V, final_Vc = jax_vmap(final_value_fn)(
            rollout.next_graph, final_rnn_states_V, final_rnn_states_Vc)

        # calculate next values
        bT_next_V = jnp.concatenate([bT_V[:, 1:], final_V[:, None]], axis=1)
        bTah_next_Vc = jnp.concatenate([bTah_Vc[:, 1:], final_Vc[:, None]], axis=1)

        # calculate GAE
        bT_targets, bT_gaes = compute_gae(
            values=bT_V,
            rewards=rollout.rewards,
            dones=rollout.dones,
            next_values=bT_next_V,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # calculate cost GAE
        bTah_cost_targets, bTah_cost_gaes = compute_gae(
            values=bTah_Vc,
            rewards=jnp.maximum(rollout.costs, 0.0),
            dones=rollout.dones[:, :, None, None],
            next_values=bTah_next_Vc,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # update ppo
        def update_fn(carry, idx):
            critic, cost_critic, policy, lagr_lambda = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            critic, cost_critic, value_info = self.update_value(
                critic, cost_critic, rollout_batch, bT_targets[idx], bTah_cost_targets[idx],
                value_rnn_states[idx], cost_rnn_states[idx], rnn_chunk_ids
            )
            policy, lagr_lambda, policy_info = self.update_policy_lagr(
                policy, lagr_lambda, rollout_batch, bT_gaes[idx], bTah_cost_gaes[idx], bTah_Vc[idx], rnn_chunk_ids)

            return (critic, cost_critic, policy, lagr_lambda), value_info | policy_info

        (critic_train_state, cost_critic_train_state, policy_train_state, lagr), update_info = jax.lax.scan(
            update_fn, (critic_train_state, cost_critic_train_state, policy_train_state, lagr), batch_idx)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], update_info)

        return critic_train_state, cost_critic_train_state, policy_train_state, lagr, info

    def update_value(
            self,
            critic_train_state: TrainState,
            cost_critic_train_state: TrainState,
            rollout: Rollout,
            bT_targets: Array,
            bTah_cost_targets: Array,
            value_rnn_states: Array,
            cost_rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        value_rnn_state_inits = jnp.zeros_like(value_rnn_states[:, rnn_chunk_ids[:, 0]])  # (n_env, n_chunk, ...)
        cost_rnn_state_inits = jnp.zeros_like(cost_rnn_states[:, rnn_chunk_ids[:, 0]])
        bcT_targets = bT_targets[:, rnn_chunk_ids]
        bcTah_cost_targets = bTah_cost_targets[:, rnn_chunk_ids]
        graph_chunks = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)

        def get_loss(critic_params, cost_critic_params):
            bcT_value, _, _ = jax.vmap(jax.vmap(
                ft.partial(self.scan_value,
                           critic_params=critic_params)
            ))(graph_chunks, value_rnn_state_inits)
            bcT_value = bcT_value.squeeze(axis=(-1, -2))

            bcTah_cost_value, _, _ = jax.vmap(jax.vmap(
                ft.partial(self.scan_cost_value,
                           cost_critic_params=cost_critic_params)
            ))(graph_chunks, cost_rnn_state_inits)

            loss_value = optax.l2_loss(bcT_value, bcT_targets).mean()
            loss_cost = optax.l2_loss(bcTah_cost_value, bcTah_cost_targets).mean()
            info = {
                "critic/loss": loss_value,
                "critic/loss_Vh": loss_cost
            }
            return loss_value + loss_cost, info

        (grad_value, grad_cost), value_info = jax.grad(get_loss, argnums=(0, 1), has_aux=True)(
            critic_train_state.params, cost_critic_train_state.params)
        grad_value_has_nan = has_any_nan_or_inf(grad_value).astype(jnp.float32)
        grad_cost_has_nan = has_any_nan_or_inf(grad_cost).astype(jnp.float32)
        grad_value, grad_value_norm = compute_norm_and_clip(grad_value, self.max_grad_norm)
        grad_cost, grad_cost_norm = compute_norm_and_clip(grad_cost, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad_value)
        cost_critic_train_state = cost_critic_train_state.apply_gradients(grads=grad_cost)

        return critic_train_state, cost_critic_train_state, (value_info | {'critic/has_nan': grad_value_has_nan,
                                                                           'critic/grad_Vh_has_nan': grad_cost_has_nan,
                                                                           'critic/grad_norm': grad_value_norm,
                                                                           'critic/grad_Vh_norm': grad_cost_norm})

    def update_policy_lagr(
            self,
            policy_train_state: TrainState,
            lagr: Array,
            rollout: Rollout,
            bT_gaes: Array,
            bTah_cost_gaes: Array,
            bTah_Vc: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, Array, dict]:
        bcT_rollout = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout)
        rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])
        bcT_gaes = bT_gaes[:, rnn_chunk_ids]
        bcTa_gaes = jnp.repeat(bcT_gaes[:, :, :, None], self.n_agents, axis=-1)
        bcTah_cost_gaes = bTah_cost_gaes[:, rnn_chunk_ids]
        bcTa_gaes = bcTa_gaes - (bcTah_cost_gaes * lagr[None, None, None, :]).mean(axis=-1)
        bcTah_Vc = bTah_Vc[:, rnn_chunk_ids]

        graph_chunks = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
        action_chunks = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.actions)

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        action_keys = jax.tree_map(lambda x: x[:, rnn_chunk_ids], action_keys)

        def get_loss(params):
            bcTa_log_pis, bcTa_entropy, _, _ = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action, actor_params=params)
            ))(graph_chunks, action_chunks, rnn_state_inits, action_keys)
            bcTa_ratio = jnp.exp(bcTa_log_pis - bcT_rollout.log_pis)
            loss_policy1 = -bcTa_ratio * bcTa_gaes
            loss_policy2 = -jnp.clip(bcTa_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * bcTa_gaes
            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            mean_entropy = bcTa_entropy.mean()
            policy_loss = loss_policy - self.coef_ent * mean_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(bcTa_ratio - 1.0))
            info = {
                'policy/loss': loss_policy,
                'policy/entropy': mean_entropy,
                'policy/clip_frac': clip_frac,
                'policy/total_variation_dist': total_variation_dist
            }
            return policy_loss, (bcTa_ratio, info)

        grad, (bcTa_weight, policy_info) = jax.grad(get_loss, has_aux=True)(policy_train_state.params)
        grad_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        # update lagrange multiplier
        delta_lagr = -(bcTah_Vc * (1 - self.gamma) +
                       bcTa_weight[:, :, :, :, None] * bcTah_cost_gaes).mean(axis=(0, 1, 2))
        lagr = nn.relu(lagr - delta_lagr * self.lr_lagr)

        return policy_train_state, lagr, policy_info | {'policy/has_nan': grad_has_nan,
                                                        'policy/grad_norm': grad_norm,
                                                        'policy/mean_lagr': lagr.mean()}

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.critic_train_state.params, open(os.path.join(model_dir, 'critic.pkl'), 'wb'))
        pickle.dump(self.cost_critic_train_state.params, open(os.path.join(model_dir, 'cost_critic.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.critic_train_state = \
            self.critic_train_state.replace(params=pickle.load(open(os.path.join(path, 'critic.pkl'), 'rb')))
        self.cost_critic_train_state = \
            self.cost_critic_train_state.replace(params=pickle.load(open(os.path.join(path, 'cost_critic.pkl'), 'rb')))
