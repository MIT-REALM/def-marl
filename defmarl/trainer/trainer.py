import wandb
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft
import jax.numpy as jnp

from time import time
from tqdm import tqdm

from .data import Rollout
from .utils import test_rollout
from ..env import MultiAgentEnv
from ..algo.base import Algorithm


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='defmarl', group=env.__class__.__name__, dir=self.log_dir)

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']
        self.full_eval_interval = params['full_eval_interval']

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        assert 'full_eval_interval' in params, 'full_eval_interval not found in params'
        assert params['full_eval_interval'] > 0, 'full_eval_interval must be positive'
        return True

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        init_rnn_state = self.algo.init_rnn_state

        # preprocess the test function
        zmax_fn = lambda graph, value_rnn_state, params: \
            (jnp.array([[-self.env_test.reward_min]]).repeat(self.env.num_agents, axis=0), value_rnn_state)
        zmin_fn = lambda graph, value_rnn_state, params: \
            (jnp.array([[-self.env_test.reward_max]]).repeat(self.env.num_agents, axis=0), value_rnn_state)

        def test_fn_single(params, z_fn, key):
            act_fn = ft.partial(self.algo.act, params=params)
            z_fn = ft.partial(z_fn, params=params) if z_fn is not None else None
            return test_rollout(
                self.env_test,
                act_fn,
                init_rnn_state,
                key,
                init_rnn_state,
                z_fn
            )

        test_opt_fn = lambda params, keys: (
            jax.vmap(ft.partial(
                test_fn_single, params, self.algo.get_opt_z if hasattr(self.algo, 'get_opt_z') else None))(keys)
        )
        test_zmin_fn = lambda params, keys: jax.vmap(ft.partial(test_fn_single, params, zmin_fn))(keys)
        test_zmax_fn = lambda params, keys: jax.vmap(ft.partial(test_fn_single, params, zmax_fn))(keys)

        test_opt_fn = jax.jit(test_opt_fn)
        test_zmin_fn = jax.jit(test_zmin_fn)  # aggressive
        test_zmax_fn = jax.jit(test_zmax_fn)  # conservative

        # start training
        test_key = jr.PRNGKey(self.seed)
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]
        test_zmax_keys = jr.split(test_key, 1_000)[self.n_env_test: 2 * self.n_env_test]
        test_zmin_keys = jr.split(test_key, 1_000)[2 * self.n_env_test: 3 * self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=80)
        for step in range(0, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                eval_info = {}
                if step % self.full_eval_interval == 0:
                    # full test with optimal z
                    test_rollouts: Rollout = test_opt_fn(self.algo.params, test_keys)
                    total_reward = test_rollouts.rewards.sum(axis=-1)
                    reward_min, reward_max = total_reward.min(), total_reward.max()
                    reward_mean = np.mean(total_reward)
                    reward_final = np.mean(test_rollouts.rewards[:, -1])
                    cost = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                    unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)
                    eval_info = eval_info | {
                        "eval/reward": reward_mean,
                        "eval/reward_final": reward_final,
                        "eval/cost": cost,
                        "eval/unsafe_frac": unsafe_frac,
                        "eval/opt_z0": test_rollouts.zs[0, 0, 0, 0],
                    }
                    time_since_start = time() - start_time
                    eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                    f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                    f'unsafe_frac: {unsafe_frac:6.2f}')
                    tqdm.write(eval_verbose)

                # partial test with zmin and zmax
                test_zmax_rollouts: Rollout = test_zmax_fn(self.algo.params, test_zmax_keys)
                test_zmin_rollouts: Rollout = test_zmin_fn(self.algo.params, test_zmin_keys)
                reward_mean_zmax = np.mean(test_zmax_rollouts.rewards.sum(axis=-1))
                reward_mean_zmin = np.mean(test_zmin_rollouts.rewards.sum(axis=-1))
                reward_final_zmax = np.mean(test_zmax_rollouts.rewards[:, -1])
                reward_final_zmin = np.mean(test_zmin_rollouts.rewards[:, -1])
                cost_zmin = jnp.maximum(test_zmin_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                cost_zmax = jnp.maximum(test_zmax_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                unsafe_frac_zmin = np.mean(test_zmin_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)
                unsafe_frac_zmax = np.mean(test_zmax_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)
                eval_info = eval_info | {
                    "eval/reward_zmin": reward_mean_zmin,
                    "eval/reward_zmax": reward_mean_zmax,
                    "eval/reward_final_zmin": reward_final_zmin,
                    "eval/reward_final_zmax": reward_final_zmax,
                    "eval/cost_zmin": cost_zmin,
                    "eval/cost_zmax": cost_zmax,
                    "eval/unsafe_frac_zmin": unsafe_frac_zmin,
                    "eval/unsafe_frac_zmax": unsafe_frac_zmax,
                }
                wandb.log(eval_info, step=self.update_steps)

            # save the model
            if self.save_log and step % self.save_interval == 0:
                self.algo.save(os.path.join(self.model_dir), step)

            # collect rollouts
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts = self.algo.collect(self.algo.params, key_x0)

            # update the algorithm
            update_info = self.algo.update(rollouts, step)
            wandb.log(update_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
