from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from flightning.envs.env_base import EnvTransition, Env, EnvState
from flightning.utils import spaces
from flightning.utils.math import normalize
from flightning.utils.spaces import Space


class EnvWrapper(Env):
    """
    Base class for environment wrappers.
    """

    def __init__(self, env: Env):
        self._env = env

    def __getattr__(self, item):
        return getattr(self._env, item)

    def reset(self, key, state=None):
        return self._env.reset(key, state)

    def _step(self, state, action, key) -> EnvTransition:
        return self._env._step(state, action, key)

    @property
    def action_space(self) -> Space:
        return self._env.action_space

    @property
    def observation_space(self) -> Space:
        return self._env.observation_space

    def step(self, state, action, key) -> EnvTransition:
        return self._env.step(state, action, key)

    @property
    def unwrapped(self):
        return self._env.unwrapped


class FlattenObservationWrapper(EnvWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: Env):
        super().__init__(env)

    @property
    def observation_space(self) -> spaces.Box:
        assert isinstance(
            self._env.observation_space, spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state=None):
        obs, state = self._env.reset(key, state)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, key) -> EnvTransition:
        transition = self._env.step(state, action, key)
        obs = jnp.reshape(transition.obs, (-1,))
        transition = transition._replace(obs=obs)
        return transition


@jdc.pytree_dataclass
class LogEnvState(EnvState):
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(EnvWrapper):
    """Adds logging to the environment."""

    def __init__(self, env: Env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state=None):
        if state is not None:
            env_state = state.env_state
        else:
            env_state = None
        state, obs = self._env.reset(key, env_state)
        # noinspection PyArgumentList
        log_state = LogEnvState(
            env_state=state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            timestep=0,
        )
        return log_state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action, key) -> EnvTransition:
        transition = self._env.step(state.env_state, action, key)
        env_state, obs, reward, terminated, truncated, info = transition
        done = jnp.logical_or(terminated, truncated)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        # noinspection PyArgumentList
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns
            * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths
            * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return transition._replace(state=state, info=info)

    def _get_obs(self, state, asymmetric=False):
        return self._env._get_obs(state.env_state, asymmetric)


class VecEnv(EnvWrapper):
    """Vectorize an environment for parallelization."""

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, 0))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0))
        self._get_obs = jax.vmap(self._env._get_obs, in_axes=(0, None))


class NormalizeActionWrapper(EnvWrapper):
    """Normalize the actions of the environment to [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)

    def map_action(self, action):
        # map action from [-1, 1] to action bounds
        action = (action + 1.0) / 2.0 * (
            self._env.action_space.high - self._env.action_space.low
        ) + self._env.action_space.low
        return action

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, key) -> EnvTransition:
        action = self.map_action(action)
        return self._env.step(state, action, key)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action, key) -> EnvTransition:
        action = self.map_action(action)
        return self._env._step(state, action, key)

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_space.shape
        )


class MinMaxObservationWrapper(EnvWrapper):
    """Normalize the observations of the environment to [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self._obs_min = jnp.array(self._env.observation_space.low)
        self._obs_max = jnp.array(self._env.observation_space.high)
        # check for infinities
        assert jnp.isinf(self._obs_max).sum() == 0, "Obs space has infinities"
        assert jnp.isinf(self._obs_min).sum() == 0, "Obs space has infinities"

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=self._obs_min.shape)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state=None):
        state, obs = self._env.reset(key, state)
        obs = normalize(obs, self._obs_min, self._obs_max)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, key) -> EnvTransition:
        transition = self._env.step(state, action, key)
        obs = normalize(transition.obs, self._obs_min, self._obs_max)
        return transition._replace(obs=obs)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action, key) -> EnvTransition:
        transition = self._env._step(state, action, key)
        obs = normalize(transition.obs, self._obs_min, self._obs_max)
        return transition._replace(obs=obs)
