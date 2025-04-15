import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from attrs import define
from typing import Callable, NamedTuple

from defmarl.utils.typing import FloatScalar, IntScalar
from defmarl.utils.utils import jax_vmap, tree_where


class ChandrupatlaState(NamedTuple):
    # Endpoint of interval. No guarantees about whether x1 or x2 is larger.
    x1: FloatScalar
    # Endpoint of interval. No guarantees about whether x1 or x2 is larger.
    x2: FloatScalar
    # f(x1)
    y1: FloatScalar
    # f(x2)
    y2: FloatScalar
    # Next position to be evaluated as   (1-t) * x1  +  t * x2
    t: FloatScalar
    # A counter that records how many times we failed to at least halve the interval.
    fail_counter: IntScalar

    def get_xnew(self) -> FloatScalar:
        return self.x2 + self.t * (self.x1 - self.x2)


RootFn = Callable[[FloatScalar], FloatScalar]
_F = FloatScalar


def secant_interp(x1: _F, x2: _F, y1: _F, y2: _F) -> _F:
    both_same = y1 == y2
    both_same_out = 0.5 * (x1 + x2)

    out = x1 - y1 * (x1 - x2) / (y1 - y2)
    out = jnp.where(both_same, both_same_out, out)
    return out


def _inv_quad_interp_t(
    x1: FloatScalar, x2: FloatScalar, x3: FloatScalar, y1: FloatScalar, y2: FloatScalar, y3: FloatScalar
) -> FloatScalar:
    al = (x3 - x2) / (x1 - x2)
    a = y2 / (y1 - y2)
    b = y3 / (y1 - y3)
    c = y2 / (y3 - y2)
    d = y1 / (y3 - y1)
    return a * b + c * d * al


@define
class Chandrupatla:
    root_fn: RootFn
    n_iters: int
    init_t: float = 0.5
    batch_init: bool = True

    def init_state(self, lb: FloatScalar, ub: FloatScalar) -> ChandrupatlaState:
        if self.batch_init:
            return self.batch_init_state(lb, ub)
        else:
            return self.simple_init_state(lb, ub)

    def batch_init_state(self, lb: FloatScalar, ub: FloatScalar):
        ts = jnp.linspace(0.0, 1.0, num=8)
        n_ts = len(ts)

        xs = lb + ts * (ub - lb)
        assert xs.shape == (n_ts,)
        ys = jax_vmap(self.root_fn)(xs)

        LARGE_NUM = np.finfo(np.float32).max
        # Choose lb as the largest negative value.
        # If all values are positive, then take t=0.
        new_lb_idx = jnp.where(ys < 0, ys, -LARGE_NUM).argmax()
        new_lb_idx = jnp.where(jnp.all(ys > 0), 0, new_lb_idx)

        # Choose ub as the smallest positive value.
        # If all values are negative, then take t=1.
        new_ub_idx = jnp.where(ys > 0, ys, LARGE_NUM).argmin()
        new_ub_idx = jnp.where(jnp.all(ys < 0), n_ts - 1, new_ub_idx)

        x1, x2 = xs[new_lb_idx], xs[new_ub_idx]
        y1, y2 = ys[new_lb_idx], ys[new_ub_idx]
        return ChandrupatlaState(x1, x2, y1, y2, jnp.array(self.init_t), jnp.array(0))

    def simple_init_state(self, lb: FloatScalar, ub: FloatScalar) -> ChandrupatlaState:
        x1, x2 = lb, ub
        y1, y2 = self.root_fn(x1), self.root_fn(x2)
        return ChandrupatlaState(x1, x2, y1, y2, jnp.array(self.init_t), jnp.array(0))

    def update_step(self, state: ChandrupatlaState, _):
        x1, x2, y1, y2, t, fail_counter = state
        x = state.get_xnew()
        y = self.root_fn(x)

        # Swap x1 and x2 to ensure that f(x) has the same sign as f(x_2).
        # i.e., the bounds will be [f(x_1), f(x_2)] -> [f(x_1), f(x)].
        should_swap = jnp.sign(y) == jnp.sign(y1)
        x1, x2, y1, y2, t = tree_where(should_swap, [x2, x1, y2, y1, 1 - t], [x1, x2, y1, y2, t])

        # Tighten the bounds, x3 <- x2, x2 <- x
        x2, x3 = x, x2
        y2, y3 = y, y2

        # >> Keep count of how many times t fails to halve.
        #   t=0 => x = x2, so repeatedly having small ts means that x1 is not changing.
        fail_to_halve = t < 0.5
        failed = fail_to_halve
        fail_counter = jnp.where(failed, fail_counter + 1, 0)

        # >> Propose next point to eval.
        # OPTION 1: Inverse Quadratic Interpolation, if the interpolation is monotonic.
        x_ratio = (x2 - x1) / (x3 - x1)
        y_ratio = (y2 - y1) / (y3 - y1)
        use_iqi = jnp.logical_and(y_ratio**2 < x_ratio, x_ratio < 1 - (1 - y_ratio) ** 2)
        iqi_t = _inv_quad_interp_t(x1, x2, x3, y1, y2, y3)
        # Switch to bisection if out of bounds.
        iqi_t_oob = jnp.logical_or(jnp.logical_or(iqi_t < 0, iqi_t > 1), jnp.isnan(iqi_t))
        iqi_t = jnp.where(iqi_t_oob, 0.5, iqi_t)

        # OPTION 2: Illinois method if we have failed 3 times.
        #           i.e., halve the y-value of the retained endpoint ( x1 ).
        use_illinois = fail_counter == 3

        # m = 1 - y2 / y3
        # m = jnp.where(m > 0, m, 0.5)
        m = 0.5

        illinois_y1 = m * y1
        # illinois_y1 = 0.5 * y3 * ((x1 - x2) / (x3 - x2))
        illinois_t = secant_interp(x1, x2, illinois_y1, y2)
        # Don't go past bisection.
        illinois_t = jnp.minimum(0.5, illinois_t)

        # OPTION 3: Bisection.
        bisec_t = 0.5
        new_t = jnp.where(use_iqi, iqi_t, jnp.where(use_illinois, illinois_t, bisec_t))

        used_which = jnp.where(
            use_iqi, np.array([1, 0, 0]), jnp.where(use_illinois, np.array([0, 1, 0]), np.array([0, 0, 1]))
        )

        new_state = ChandrupatlaState(x1, x2, y1, y2, new_t, fail_counter)
        return new_state, (x, y, used_which)

    def _update(self, state: ChandrupatlaState, _):
        new_state, (x, y, used_which) = self.update_step(state, _)
        return new_state, used_which

    def bracket_valid(self, lb: FloatScalar, ub: FloatScalar) -> bool:
        return (self.root_fn(lb) < 0) and (self.root_fn(ub) > 0)

    def refine_output(self, state: ChandrupatlaState) -> FloatScalar:
        # Secant approximation of the root. We can do this safely because we have tight brackets.
        x1, x2, y1, y2, t, fail_counter = state
        secant_x = (y2 * x1 - y1 * x2) / (y2 - y1)
        return jnp.where(~jnp.isfinite(secant_x), 0.5 * (x1 + x2), secant_x)

    def run(self, lb: FloatScalar, ub: FloatScalar):
        init_state = self.init_state(lb, ub)
        final_state, used_which = lax.scan(self._update, init_state, None, length=self.n_iters)
        best_x = self.refine_output(final_state)
        return best_x, jnp.sum(used_which, axis=0), init_state

    def run_detailed(self, lb: FloatScalar, ub: FloatScalar):
        init_state = self.init_state(lb, ub)
        final_state, outputs = lax.scan(self.update_step, init_state, None, length=self.n_iters)
        best_x = self.refine_output(final_state)
        return best_x, outputs
