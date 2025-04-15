from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

from ..utils.typing import Action, Params, PRNGKey, Array
from ..utils.graph import GraphsTuple
from ..trainer.data import Rollout
from ..env.base import MultiAgentEnv


class Algorithm(ABC):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            n_agents: int
    ):
        self._env = env
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._n_agents = n_agents
        self.init_rnn_state = None

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        return self._edge_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @abstractproperty
    def config(self) -> dict:
        pass

    @abstractproperty
    def params(self) -> Params:
        pass

    @abstractmethod
    def act(self, graph: GraphsTuple, z: Array, rnn_state: Array, params: Optional[Params] = None) -> [Action, Array]:
        """
        Get action from the policy.

        Returns
        -------
        action: Action,
            The action to be taken by the agent.
        rnn_states: Array,
            The updated rnn states.
        """
        pass

    @abstractmethod
    def step(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, key: PRNGKey, params: Optional[Params] = None
    ) -> Tuple[Action, Array, Array]:
        """
        Sample action from the policy, used for training.

        Returns
        -------
        action: Action,
            The stochastic action to be taken by the agent.
        z: Array,
            The z value used for EFPPO.
        log_pi: Array,
            The log probability of the action.
        rnn_states: Array,
            The updated rnn states.
        """
        pass

    @abstractmethod
    def collect(self, params: Params, key: PRNGKey) -> Rollout:
        pass

    @abstractmethod
    def update(self, rollout: Rollout, step: int) -> dict:
        pass

    @abstractmethod
    def save(self, save_dir: str, step: int):
        pass

    @abstractmethod
    def load(self, load_dir: str, step: int):
        pass
