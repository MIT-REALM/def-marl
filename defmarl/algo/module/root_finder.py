import jax.debug
import jax.numpy as jnp
import functools as ft
import jax.tree_util as jtu
import jraph

from typing import Callable

from defmarl.algo.module.chandrupatla import Chandrupatla
from defmarl.nn.utils import safe_get
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import FloatScalar, IntScalar
from defmarl.utils.utils import jax_vmap


class RootFinder:
    def __init__(
        self,
        z_min: float,
        z_max: float,
        n_agent: int,
        h_tgt: float = -0.2,
        h_eps: float = 1e-2,
        n_iters: int = 20,
        z_comm: bool = False
    ):
        self.z_min = z_min
        self.z_max = z_max
        self.n_agent = n_agent
        self.h_tgt = h_tgt - h_eps
        self.n_iters = n_iters
        self.z_comm = z_comm

    def get_dec_opt_z(self, Vh_fn: Callable, graph: GraphsTuple):
        agent_idx = jnp.arange(self.n_agent)
        solve_fn = ft.partial(self.get_opt_z, Vh_fn)
        opt_z = jax_vmap(solve_fn)(agent_idx)  # (n_agent,)

        def z_comm_(i, z):
            z_map = jtu.tree_map(lambda n: safe_get(n, graph.senders, fill_value=-jnp.inf), z)
            max_z = jraph.segment_max(z_map, segment_ids=graph.receivers, num_segments=graph.nodes.shape[0])
            max_z = max_z[:self.n_agent]
            z = jnp.maximum(max_z, z)
            return z

        if self.z_comm:
            opt_z = jax.lax.fori_loop(0, self.n_agent, z_comm_, opt_z)

        opt_z = opt_z[:, None]
        _, rnn_states = Vh_fn(opt_z)
        return opt_z, rnn_states

    def get_opt_z(self, Vh_fn: Callable, i_agent: IntScalar) -> FloatScalar:
        def z_root_fn(z):
            h_Vh, _ = Vh_fn(z[None, None].repeat(self.n_agent, axis=0))
            h_Vh = h_Vh[i_agent]
            Vh = h_Vh.max()
            root = -(Vh - self.h_tgt)
            return root

        solver = Chandrupatla(z_root_fn, n_iters=self.n_iters, init_t=0.5)
        opt_z, _, init_state = solver.run(self.z_min, self.z_max)

        # The rootfinding is only valid if we bracketed the root.
        #         Vh < h_tgt is safe.
        #         => -(Vh - h_tgt) > 0 is safe.
        # If max(y1, y2) > 0, then both zmin and zmax are safe, so we take zmin.
        # If max(y1, y2) < 0, then both zmin and zmax are unsafe, so we take zmax.
        both_safe = (init_state.y1 > 0) & (init_state.y2 > 0)
        both_unsafe = (init_state.y1 < 0) & (init_state.y2 < 0)
        opt_z = jnp.where(both_safe, self.z_min, jnp.where(both_unsafe, self.z_max, opt_z))

        return opt_z
