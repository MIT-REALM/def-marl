import flax.linen as nn
import jax.numpy as jnp

from typing import Type

from ..utils.typing import Array
from ..utils.utils import jax_vmap


class RNN(nn.Module):
    rnn_cls: Type[nn.LSTMCell] | Type[nn.GRUCell]
    rnn_layers: int

    @nn.compact
    def __call__(self, x: Array, rnn_state: Array) -> [Array, Array]:
        """rnn_state: (n_layers, n_agents, n_carries, hid_size)"""
        new_rnn_state = []
        for i in range(self.rnn_layers):
            if isinstance(self.rnn_cls(), nn.GRUCell):
                rnn_state_i, x = jax_vmap(self.rnn_cls())(rnn_state[i, :, 0, :], x)
                rnn_state_i = jnp.expand_dims(rnn_state_i, axis=1)
            elif isinstance(self.rnn_cls(), nn.LSTMCell):
                rnn_state_i, x = jax_vmap(self.rnn_cls())(rnn_state[i], x)
                rnn_state_i = jnp.stack(rnn_state_i, axis=1)
            else:
                raise ValueError(f"Unsupported RNN cell type: {self.rnn_cls()}")
            new_rnn_state.append(rnn_state_i)
        new_rnn_state = jnp.stack(new_rnn_state)

        return x, new_rnn_state
