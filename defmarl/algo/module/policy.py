import flax.linen as nn
import functools as ft
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp

from typing import Type, Tuple
from abc import ABC, abstractproperty, abstractmethod

from .distribution import TanhTransformedDistribution, tfd
from .ef_wrapper import ZEncoder
from ...utils.typing import Action, Array
from ...utils.graph import GraphsTuple
from ...nn.utils import default_nn_init, scaled_init
from ...nn.gnn import GNN, GraphTransformerGNN
from ...nn.rnn import RNN
from ...nn.mlp import MLP
from ...utils.typing import PRNGKey, Params


class PolicyNet(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    rnn_cls: Type[RNN] = None
    z_encoder_cls: Type[nn.Module] = None

    @nn.compact
    def __call__(
            self, graph: GraphsTuple, rnn_state: Array, node_type: int = None, n_type: int = None, z: Array = None,
    ) -> [Array, Array]:
        x = self.gnn_cls()(graph, node_type, n_type)
        if self.z_encoder_cls is not None:
            z_enc = self.z_encoder_cls()(z)
            x = jnp.concatenate([x, z_enc], axis=-1)
        x = self.head_cls()(x)
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)
        return x, rnn_state


class PolicyDistribution(nn.Module, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> [tfd.Distribution, Array]:
        pass

    @abstractproperty
    def nu(self) -> int:
        pass


class TanhNormal(PolicyDistribution):
    base_cls: Type[PolicyNet]
    _nu: int
    scale_final: float = 0.01
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5

    @property
    def std_dev_init_inv(self):
        # inverse of log(sum(exp())).
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(
            self, obs: GraphsTuple, rnn_state: Array, n_agents: int, z: Array, *args, **kwargs
    ) -> [tfd.Distribution, Array]:
        x, rnn_state = self.base_cls()(obs, rnn_state=rnn_state, node_type=0, n_type=n_agents, z=z)
        scaler_init = scaled_init(default_nn_init(), self.scale_final)
        feats_scaled = nn.Dense(64, kernel_init=scaler_init, name="ScaleHid")(x)

        means = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseMean")(feats_scaled)
        stds_trans = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseStdTrans")(feats_scaled)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min
        # stds = self.std_dev_min
        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1), rnn_state

    @property
    def nu(self):
        return self._nu


class MultiAgentPolicy(ABC):

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, action_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    @abstractmethod
    def initialize_carry(self, key: PRNGKey) -> Array:
        pass

    @abstractmethod
    def get_action(self, params: Params, obs: GraphsTuple, rnn_state: Array) -> [Action, Array]:
        """
        Get action from the policy.

        Returns
        -------
        action: Action,
            The action to be taken by the agent.
        rnn_state: Array,
            The updated rnn states.
        """
        pass

    @abstractmethod
    def sample_action(
            self, params: Params, obs: GraphsTuple, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Action, Array, Array]:
        """
        Sample action from the policy.

        Returns
        -------
        action: Action,
            The stochastic action to be taken by the agent.
        log_pi: Array,
            The log probability of the action.
        rnn_state: Array,
            The updated rnn states.
        """
        pass

    @abstractmethod
    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Array, Array, Array]:
        pass


class PPOPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 1,
            gnn_out_dim: int = 16,
            use_lstm: bool = False,
            use_ef: bool = False
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.gnn_out_dim = gnn_out_dim
        self.use_rnn = use_rnn
        self.use_ef = use_ef
        self.gnn = ft.partial(
            GraphTransformerGNN,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=3,
            n_layers=gnn_layers
        )
        self.head = ft.partial(
            MLP,
            hid_sizes=(64, 64),
            act=nn.relu,
            act_final=True,
            name='PolicyGNNHead'
        )
        self.z_encoder = ft.partial(
            ZEncoder,
            nz=8,
            z_mean=1.0,
            z_scale=1.0
        ) if use_ef else None
        if use_rnn:
            self.rnn_base = ft.partial(nn.LSTMCell if use_lstm else nn.GRUCell, features=64)
            self.rnn = ft.partial(
                RNN,
                rnn_cls=self.rnn_base,
                rnn_layers=rnn_layers
            )
            self.policy_base = ft.partial(
                PolicyNet,
                gnn_cls=self.gnn,
                head_cls=self.head,
                rnn_cls=self.rnn,
                z_encoder_cls=self.z_encoder
            )
            self.dist = TanhNormal(base_cls=self.policy_base, _nu=action_dim)
        else:
            self.policy_base = ft.partial(
                PolicyNet,
                gnn_cls=self.gnn,
                head_cls=self.head,
                z_encoder_cls=self.z_encoder
            )
            self.dist = TanhNormal(base_cls=self.policy_base, _nu=action_dim)

    def initialize_carry(self, key: PRNGKey) -> Array:
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))

    def get_action(self, params: Params, obs: GraphsTuple, rnn_state: Array, z: Array = None) -> [Action, Array]:
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents, z=z)
        action = dist.mode()
        return action, rnn_state

    def sample_action(
            self, params: Params, obs: GraphsTuple, rnn_state: Array, key: PRNGKey, z: Array = None
    ) -> Tuple[Action, Array, Array]:
        rnn_state: Array
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents, z=z)
        action = dist.sample(seed=key)
        log_pi = dist.log_prob(action)
        return action, log_pi, rnn_state

    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, rnn_state: Array, key: PRNGKey, z: Array = None
    ) -> Tuple[Array, Array, Array]:
        rnn_state: Array
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents, z=z)
        log_pi = dist.log_prob(action)
        entropy = dist.entropy(seed=key)
        return log_pi, entropy, rnn_state
