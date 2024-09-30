from functools import partial
from typing import Any, Dict, Generic, Optional, TypeVar
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from flightning.utils.pytrees import CustomPyTree, tree_select
from flightning.utils.spaces import Space

# a transition consists of State, Obs, Reward, Terminated, Truncated, Info

TEnvState = TypeVar("TEnvState", bound="EnvState")


class EnvTransition(NamedTuple):
    state: TEnvState
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    info: Dict[str, Any]


@jdc.pytree_dataclass
class EnvState(CustomPyTree):
    pass


class Env(Generic[TEnvState]):

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: TEnvState, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:
        """
        Env step function. Handles reset of environment if episode is terminated
        or truncated. Calls _step().
        :param state: environment state
        :param action: action to take in the environment
        :param key: PRNGKey
        :return: state, obs, reward, terminated, truncated, info
        """

        key_step, key_reset = jax.random.split(key)
        step_state, step_obs, reward, terminated, truncated, info = self._step(
            state, action, key_step
        )
        reset_state, reset_obs = self.reset(key_reset, state)
        # Reset environment if needed
        done = jnp.logical_or(terminated, truncated)
        state = tree_select(done, reset_state, step_state)
        obs = tree_select(done, reset_obs, step_obs)

        return EnvTransition(state, obs, reward, terminated, truncated, info)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, state: Optional[TEnvState] = None
    ) -> tuple[TEnvState, jax.Array]:
        """
        :param state: last state of the environment, needed for curricula etc.
        :param key: random key
        :return: state, obs
        """
        raise NotImplementedError

    def _step(
        self, state: TEnvState, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:
        """
        Environment-specific step.
        :return: state, obs, reward, terminated, truncated, info
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def action_space(self) -> Space:
        """Action space of the environment."""
        raise NotImplementedError

    @property
    def observation_space(self) -> Space:
        """Observation space of the environment."""
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self


def rollout(
    env,
    key,
    policy,
    state: Optional[EnvState] = None,
    *,
    real_step: bool = False,
    num_steps=None,
) -> EnvTransition:
    """
    Rollout the environment with a given policy
    :param env: environment
    :param key: random key
    :param policy: callable policy function: (obs, key) -> action
    :param state: environment state needed for curricula etc.
    :param real_step: whether to use the real step function or the _step function
    :param num_steps: number of steps, defaults to env.max_steps_in_episode
    :return: last transition and all transitions
    """
    if num_steps is None:
        num_steps = env.max_steps_in_episode
    state, obs = env.reset(key, state)
    trans_init = EnvTransition(
        state, obs, jnp.array(0), jnp.array(0), jnp.array(0), dict()
    )

    def step_fn(step_state, key_step):
        env_state, obs = step_state
        key_policy, key_step = jax.random.split(key_step)
        action = policy(obs, key_policy)
        if real_step:
            trans = env.step(env_state, action, key_step)
        else:
            trans = env._step(env_state, action, key_step)
        return (trans.state, trans.obs), trans

    keys_steps = jax.random.split(key, num_steps)
    _, transitions = jax.lax.scan(step_fn, (state, obs), keys_steps)
    # concatenate all transitions
    transitions = jax.tree.map(
        lambda l0, l1: jnp.concatenate([l0[None], l1]), trans_init, transitions
    )

    return transitions


def rollout_recurrent(
    env, key, policy, carry_init, state: Optional[EnvState] = None
) -> EnvTransition:
    """
    Rollout the environment with a given policy
    :param env: environment
    :param key: random key
    :param policy: callable policy function: (obs, key) -> action
    :param state: environment state needed for curricula etc.
    :return: last transition and all transitions
    """
    num_steps = env.max_steps_in_episode
    state, obs = env.reset(key, state)
    trans_init = EnvTransition(
        state, obs, jnp.array(0), jnp.array(0), jnp.array(0), dict()
    )

    def step_fn(step_state, key_step):
        carry, env_state, obs = step_state
        key_policy, key_step = jax.random.split(key_step)
        carry, action = policy(carry, obs, key_policy)
        trans = env._step(env_state, action, key_step)
        return (carry, trans.state, trans.obs), trans

    keys_steps = jax.random.split(key, num_steps)
    _, transitions = jax.lax.scan(
        step_fn, (carry_init, state, obs), keys_steps
    )
    # concatenate all transitions
    transitions = jax.tree.map(
        lambda l0, l1: jnp.concatenate([l0[None], l1]), trans_init, transitions
    )

    return transitions
