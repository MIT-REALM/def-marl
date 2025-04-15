import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax

from ..utils.typing import Action, Params, PRNGKey, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import jax_vmap, tree_index
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..trainer.utils import rollout as rollout_fn
from ..env.base import MultiAgentEnv
from ..algo.module.value import ValueNet
from ..algo.module.policy import PPOPolicy
from .utils import compute_gae
from .base import Algorithm


class InforMARL(Algorithm):

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
            **kwargs
    ):
        super(InforMARL, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.cost_weight = cost_weight
        self.actor_gnn_layers = actor_gnn_layers
        self.critic_gnn_layers = critic_gnn_layers
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.rollout_length = rollout_length
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.rnn_step = rnn_step
        self.use_lstm = use_lstm

        # set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph
        self.nominal_z = jnp.zeros((1,))

        # set up PPO policy
        self.policy = PPOPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.actor_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm
        )

        # initialize the rnn state
        key = jr.PRNGKey(seed)
        rnn_state_key, key = jr.split(key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_rnn_state = jax_vmap(self.policy.initialize_carry)(rnn_state_key)  # (n_agents, rnn_state_dim)
        if type(init_rnn_state) is tuple:
            init_rnn_state = jnp.stack(init_rnn_state, axis=1)  # (n_agents, n_carries, rnn_state_dim)
        else:
            init_rnn_state = jnp.expand_dims(init_rnn_state, axis=1)
        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_rnn_state = init_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        # initialize the policy
        policy_key, key = jr.split(key)
        policy_params = self.policy.dist.init(
            policy_key, nominal_graph, self.init_rnn_state, self.n_agents, self.nominal_z.repeat(self.n_agents, axis=0)
        )
        policy_optim = optax.adam(learning_rate=lr_actor)
        self.policy_optim = optax.apply_if_finite(policy_optim, 1_000_000)
        self.policy_train_state = TrainState.create(
            apply_fn=self.policy.sample_action,
            params=policy_params,
            tx=self.policy_optim
        )

        # set up PPO critic
        self.critic = ValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.critic_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            use_ef=False,
            decompose=False
        )

        # initialize the rnn state
        rnn_state_key, key = jr.split(key)
        init_Vl_rnn_state = self.critic.initialize_carry(rnn_state_key)  # (rnn_state_dim,)
        if type(init_Vl_rnn_state) is tuple:
            init_Vl_rnn_state = jnp.stack(init_Vl_rnn_state, axis=0)  # (n_carries, rnn_state_dim)
        else:
            init_Vl_rnn_state = init_Vl_rnn_state[None, :]
        # (n_rnn_layers, 1, n_carries, rnn_state_dim)
        self.init_value_rnn_state = init_Vl_rnn_state[None, :, :].repeat(self.rnn_layers, axis=0)[:, None, :, :]

        # initialize the critic
        critic_key, key = jr.split(key)
        critic_params = self.critic.net.init(
            critic_key, nominal_graph, self.init_value_rnn_state, self.n_agents, self.nominal_z)
        critic_optim = optax.adam(learning_rate=lr_critic)
        self.critic_optim = optax.apply_if_finite(critic_optim, 1_000_000)
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.get_value,
            params=critic_params,
            tx=self.critic_optim
        )

        # set up key
        self.key = key

        def rollout_fn_single_(cur_params, cur_key):
            return rollout_fn(self._env,
                              ft.partial(self.step, params=cur_params),
                              self.init_rnn_state,
                              cur_key,
                              self.gamma)

        def rollout_fn_(cur_params, cur_keys):
            return jax.vmap(ft.partial(rollout_fn_single_, cur_params))(cur_keys)

        self.rollout_fn = jax.jit(rollout_fn_)

    @property
    def config(self) -> dict:
        return {
            'cost_weight': self.cost_weight,
            'actor_gnn_layers': self.actor_gnn_layers,
            'critic_gnn_layers': self.critic_gnn_layers,
            'gamma': self.gamma,
            'rollout_length': self.rollout_length,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'batch_size': self.batch_size,
            'epoch_ppo': self.epoch_ppo,
            'clip_eps': self.clip_eps,
            'gae_lambda': self.gae_lambda,
            'coef_ent': self.coef_ent,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'rnn_step': self.rnn_step,
            'use_lstm': self.use_lstm
        }

    @property
    def params(self) -> Params:
        return self.policy_train_state.params

    def act(
            self,
            graph: GraphsTuple,
            z: Array,
            rnn_state: Array,
            params: Optional[Params] = None,
    ) -> [Action, Array]:
        if params is None:
            params = self.params
        action, rnn_state = self.policy.get_action(params, graph, rnn_state, z)
        return action, rnn_state

    def step(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, key: PRNGKey, params: Optional[Params] = None
    ) -> Tuple[Action, Array, Array]:
        if params is None:
            params = self.params
        action, log_pi, rnn_state = self.policy_train_state.apply_fn(params, graph, rnn_state, key, z)
        return action, log_pi, rnn_state

    def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
        rollout_result = self.rollout_fn(params, b_key)
        return rollout_result

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
            critic_train_state, policy_train_state, update_info = self.update_inner(
                self.critic_train_state, self.policy_train_state, rollout, batch_idx, rnn_chunk_ids
            )
            self.critic_train_state = critic_train_state
            self.policy_train_state = policy_train_state
        return update_info

    def scan_value(
            self, graphs: GraphsTuple, init_rnn_state: Array, critic_params: Params
    ) -> Tuple[Array, Array, Array]:

        def body_(rnn_state, graph):
            value, new_rnn_state = self.critic.get_value(critic_params, graph, rnn_state)
            return new_rnn_state, (value, rnn_state)

        final_rnn_state, (value_rollout, rnn_states) = jax.lax.scan(body_, init_rnn_state, graphs)

        return value_rollout, rnn_states, final_rnn_state

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            critic_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        # rollout: (n_env, length, n_agent, ...)

        # calculate values and next_values
        values, value_rnn_states, final_rnn_states = jax_vmap(
            ft.partial(self.scan_value,
                       init_rnn_state=self.init_value_rnn_state,
                       critic_params=critic_train_state.params)
        )(rollout.graph)  # values: (b, T, 1, 1)

        def final_value_fn(graph, rnn_state):
            return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state)

        final_value, _ = jax_vmap(final_value_fn)(rollout.next_graph, final_rnn_states)  # (b, a, 1)
        next_values = jnp.concatenate([values[:, 1:], final_value[:, None]], axis=1)  # (b, T, a, 1)

        # calculate GAE
        targets, gaes = compute_gae(
            values=values.squeeze(-1).sum(-1),
            rewards=rollout.rewards - self.cost_weight * jnp.maximum(rollout.costs, 0.0).sum(axis=-1).sum(axis=-1),
            dones=rollout.dones,
            next_values=next_values.squeeze(-1).sum(-1),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )  # (b, T)

        # update ppo
        def update_fn(carry, idx):
            critic, policy = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            critic, critic_info = self.update_critic(
                critic, rollout_batch, targets[idx], value_rnn_states[idx], rnn_chunk_ids)
            policy, policy_info = self.update_policy(policy, rollout_batch, gaes[idx], rnn_chunk_ids)
            return (critic, policy), (critic_info | policy_info)

        (critic_train_state, policy_train_state), info = lax.scan(
            update_fn, (critic_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)

        return critic_train_state, policy_train_state, info

    def update_critic(
            self,
            critic_train_state: TrainState,
            rollout: Rollout,
            targets: Array,
            rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        graph_chunks = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
        rnn_state_inits = jnp.zeros_like(rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        def get_value_loss(params):
            values, value_rnn_states, final_value_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_value,
                           critic_params=params)
            ))(graph_chunks, rnn_state_inits)  # values: (b, n_chunks, T_chunk, a, 1)
            values = values.sum(-2).reshape((values.shape[0], -1))
            loss_critic = optax.l2_loss(values, targets).mean()
            return loss_critic

        loss, grad = jax.value_and_grad(get_value_loss)(critic_train_state.params)
        critic_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad)
        return critic_train_state, {'critic/loss': loss,
                                    'critic/grad_norm': grad_norm,
                                    'critic/has_nan': critic_has_nan,
                                    'critic/max_target': jnp.max(targets),
                                    'critic/min_target': jnp.min(targets)}

    def scan_eval_action(
            self,
            graphs: GraphsTuple,
            actions: Action,
            init_rnn_state: Array,
            action_keys: PRNGKey,
            actor_params: Params
    ) -> Tuple[Array, Array, Array, Array]:
        def body_(rnn_state, inp):
            graph, action, key = inp
            log_pi, entropy, new_rnn_state = self.policy.eval_action(actor_params, graph, action, rnn_state, key)
            return new_rnn_state, (log_pi, entropy, rnn_state)

        final_rnn_state, outputs = jax.lax.scan(body_, init_rnn_state, (graphs, actions, action_keys))
        log_pis, entropies, rnn_states = outputs

        return log_pis, entropies, rnn_states, final_rnn_state

    def update_policy(
            self, policy_train_state: TrainState, rollout: Rollout, gaes: Array, rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        # all the agents share the same GAEs
        gaes = gaes[:, :, None]
        gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

        # divide the rollout into chunks (n_env, n_chunks, T, ...)
        graph_chunks = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
        action_chunks = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.actions)
        rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        action_keys = jax.tree_map(lambda x: x[:, rnn_chunk_ids], action_keys)

        def get_policy_loss(params):
            log_pis, policy_entropy, rnn_states, final_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action,
                           actor_params=params)
            ))(graph_chunks, action_chunks, rnn_state_inits, action_keys)
            log_pis = log_pis.reshape((log_pis.shape[0], -1, log_pis.shape[-1]))

            ratio = jnp.exp(log_pis - rollout.log_pis)
            loss_policy1 = -ratio * gaes
            loss_policy2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            total_entropy = policy_entropy.mean()
            policy_loss = loss_policy - self.coef_ent * total_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(ratio - 1.0))
            return policy_loss, {'policy/clip_frac': clip_frac,
                                 'policy/entropy': policy_entropy.mean(),
                                 'policy/total_variation_dist': total_variation_dist}

        (loss, info), grad = jax.value_and_grad(get_policy_loss, has_aux=True)(policy_train_state.params)
        policy_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)

        # clip grad
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)

        # update policy
        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        # get info
        info = {
                   'policy/loss': loss,
                   'policy/grad_norm': grad_norm,
                   'policy/has_nan': policy_has_nan,
                   'policy/log_pi_min': rollout.log_pis.min()
               } | info

        return policy_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.critic_train_state.params, open(os.path.join(model_dir, 'critic.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.critic_train_state = \
            self.critic_train_state.replace(params=pickle.load(open(os.path.join(path, 'critic.pkl'), 'rb')))
