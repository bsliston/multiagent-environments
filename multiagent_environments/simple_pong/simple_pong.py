"""Simple Pong multi-agent environment implementation.

This environment is a PettingZoo multi-agent environment implementation of the
game Pong. The observation space of this environment is meant to be abstracted
and return non-raw features only.
"""

from typing import Optional, Tuple, Dict, List, Union
import gymnasium
import numpy as np
from pettingzoo.utils.env import ParallelEnv, ObsDict, ActionDict
from gymnasium.utils import seeding


# Additional environment alias variables are set for ease of readability
# throughout the implementation.
AgentsRewards = Dict[str, float]
AgentsTerminated = Dict[str, bool]
AgentsTruncated = Dict[str, bool]
AgentsInfo = Dict[str, dict]

# For every environment step, the following step information is returned to the
# user(s) interacting with the environment.
StepInformation = Tuple[
    ObsDict, AgentsRewards, AgentsTerminated, AgentsTruncated, AgentsInfo
]


class SimplePong(ParallelEnv):
    """Simple pong multiagent environment.

    Pong abstract multiagent environment for experimental reinforcement
    learning applications that desire a simplistic abstracted observation
    space and action space. The raw rendering will not be returned in the core
    step method training and only made accessible through rendering.

    Attributes:
        metadata: Mapping of metadata of the environment.
        agents: Sequence of agents in the IDs in the environment.
        possible_agents: Sequence of possible agents in the IDs in the
            environment.
        action_spaces: Mapping of agent IDs to respective action spaces in
            the enviroment.
        observation_spaces: Mapping of agent IDs to respective observation
            spaces in the enviroment.
    """

    def __init__(self, observation_type: str = "human"):
        """Initializes simple pong environment."""
        # The environment SimplePong is setup a non-configurable 1 vs 1
        # (2 agent) environment. Each agent has an action space, which they can
        # only move up, down, or not move their cursors.
        env_name: str = "simple_pong"
        num_players: int = 2
        action_size: int = 3

        self._observation_type = observation_type

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "name": env_name,
        }
        self.agents = [f"player_{player}" for player in range(num_players)]
        self.possible_agents = self.agents[:]

        observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32, shape=(16,)
        )

        self.action_spaces = {
            agent: gymnasium.spaces.Discrete(action_size)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ObsDict:
        """Resets the environment and returns a mapping of observations.

        Args:
            seed: Optional random seed to initialize environment with. Defaults
                to None to generate random initial seeding.
            return_info: Whether or not to return additional environment reset
                info. Defaults to False.
            options: Optional reset options to set environment with. Defaults
                to None.

        Returns:
            Observation mapping from agent IDs to respective observation spaces
                in the enviroment.
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets random seed for environment reproducibility.

        Args:
            seed: Random seed to set for reproducible environment. If None
                will generate and set a random seed when called. Defaults to
                None.
        """
        if seed is None:
            _, seed = seeding.np_random()

        return NotImplemented
        # set seed ...

    def step(self, actions: ActionDict) -> StepInformation:
        """Increments and computes step in environment.

        Args:
            actions: Mapping of agent's respective actions.

        Returns:
            Agent's respective observation space mapping.
            Agent's respective reward mapping.
            Agent's respective terminated mapping.
            Agent's respective truncated mapping.
            Agent's respective info mapping.
        """
        raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, List]:
        """Renders the current environment state.

        Renders the current environment state and either displays it to the
        user or returns the environment rendering as an array. Note, rendering
        type is specified in the initialization of the environment.

        Returns:
            Environment rendering.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Closes rendering window."""
        raise NotImplementedError

    def state(self) -> np.ndarray:
        """Returns the global environment state of all agents.

        Returns:
            Global environment observation state.
        """
        return NotImplemented


if __name__ == "__main__":
    import pdb

    env = SimplePong()
    pdb.set_trace()
