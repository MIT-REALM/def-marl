import functools as ft
import flax.linen as nn
import jax.numpy as jnp

from typing import Type

from .ef_wrapper import ZEncoder
from ...nn.mlp import MLP
from ...nn.gnn import GraphTransformerGNN, GNN
from ...nn.rnn import RNN
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params, PRNGKey
from ...utils.graph import GraphsTuple


class StateFn(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]

    @nn.compact
    def __call__(
            self, obs: GraphsTuple, rnn_state: Array, n_agents: int, get_value: bool = True, *args, **kwargs
    ) -> [Array, Array]:
        # get node features
        x, _ = self.gnn_cls()(obs, rnn_state, node_type=0, n_type=n_agents)

        # aggregate information using mean
        x = x.mean(axis=0)

        # pass through head class
        x = self.head_cls()(x)

        if get_value:
            x = nn.Dense(1, kernel_init=default_nn_init())(x)

        return x, rnn_state


class RStateFn(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    n_out: int = 1
    rnn_cls: Type[RNN] = None
    z_encoder_cls: Type[nn.Module] = None

    @nn.compact
    def __call__(
            self, graph: GraphsTuple, rnn_state: Array, n_agents: int, z: Array = None, *args, **kwargs
    ) -> [Array, Array]:
        """
        rnn_state: (n_layers, n_carries, hid_size)
        """
        x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)

        # aggregate information using mean
        x = x.mean(axis=0, keepdims=True)  # (1, msg_dim)

        # add z information
        if self.z_encoder_cls is not None:
            z_enc = self.z_encoder_cls()(z)  # (1, nz)
            x = jnp.concatenate([x, z_enc], axis=-1)  # (1, msg_dim + nz)

        # pass through head class
        x = self.head_cls()(x)  # (1, msg_dim)

        # pass through RNN
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)

        # get value
        x = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)  # (1, n_out)

        return x, rnn_state


class DecRStateFn(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    n_out: int = 1
    rnn_cls: Type[RNN] = None
    z_encoder_cls: Type[nn.Module] = None
    use_global_info: bool = False

    @nn.compact
    def __call__(
            self, graph: GraphsTuple, rnn_state: Array, n_agents: int, z: Array = None, *args, **kwargs
    ) -> [Array, Array]:
        """
        rnn_state: (n_layers, n_carries, hid_size)
        """
        x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)  # (n_agent, msg_dim)

        if self.use_global_info:
            x_global = x.mean(axis=0, keepdims=True)  # (1, msg_dim)
            x = jnp.concatenate([x, jnp.tile(x_global, (n_agents, 1))], axis=-1)  # (n_agent, 2 * msg_dim)

        # add z information
        if self.z_encoder_cls is not None:
            z_enc = self.z_encoder_cls()(z)
            x = jnp.concatenate([x, z_enc], axis=-1)

        # pass through head class
        x = self.head_cls()(x)  # (n_agent, msg_dim)
        assert x.shape[0] == n_agents

        # pass through RNN
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)

        # get value
        x = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)  # (n_agent, n_out)
        assert x.shape == (n_agents, self.n_out)

        return x, rnn_state


class ValueNet:

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            n_out: int = 1,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 1,
            gnn_out_dim: int = 16,
            use_lstm: bool = False,
            use_ef: bool = False,
            decompose: bool = False,
            use_global_info: bool = False
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.n_out = n_out
        self.gnn_out_dim = gnn_out_dim
        self.use_ef = use_ef
        self.decompose = decompose
        self.use_global_info = use_global_info

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
            name='ValueGNNHead'
        )
        self.use_rnn = use_rnn
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
            if decompose:
                self.net = DecRStateFn(
                    gnn_cls=self.gnn,
                    head_cls=self.head,
                    n_out=n_out,
                    rnn_cls=self.rnn,
                    z_encoder_cls=self.z_encoder,
                    use_global_info=use_global_info
                )
            else:
                self.net = RStateFn(
                    gnn_cls=self.gnn,
                    head_cls=self.head,
                    n_out=n_out,
                    rnn_cls=self.rnn,
                    z_encoder_cls=self.z_encoder
                )
        else:
            if decompose:
                self.net = DecRStateFn(
                    gnn_cls=self.gnn,
                    head_cls=self.head,
                    n_out=n_out,
                    z_encoder_cls=self.z_encoder,
                    use_global_info=use_global_info
                )
            else:
                self.net = RStateFn(
                    gnn_cls=self.gnn,
                    head_cls=self.head,
                    n_out=n_out,
                    z_encoder_cls=self.z_encoder
                )

    def initialize_carry(self, key: PRNGKey) -> Array:
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))

    def get_value(self, params: Params, obs: GraphsTuple, rnn_state: Array, z: Array = None) -> [Array, Array]:
        values, rnn_state = self.net.apply(params, obs, rnn_state, self.n_agents, z)
        return values, rnn_state
