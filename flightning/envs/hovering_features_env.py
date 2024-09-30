import functools

import chex
import jax
import numpy as np
from jax import numpy as jnp

from flightning.envs import HoveringStateEnv
from flightning.envs.env_base import EnvTransition
from flightning.envs.hovering_state_env import EnvState
from flightning.sensors import DoubleSphereCamera, CameraNames
from flightning.utils import spaces
from flightning.utils.pytrees import pytree_roll, pytree_at_set
from flightning.utils.random import key_generator
from flightning.utils.math import normalize


class HoveringFeaturesEnv(HoveringStateEnv):

    def __init__(
        self,
        *,
        max_steps_in_episode=10000,
        dt=0.02,
        delay=0.02,
        yaw_scale=0.1,
        pitch_roll_scale=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        drone_path=None,
        reward_sharpness=1.0,
        action_penalty_weight=1.0,
        num_last_quad_states=10,
        skip_frames=1
    ):
        super().__init__(
            max_steps_in_episode=max_steps_in_episode,
            dt=dt,
            delay=delay,
            yaw_scale=yaw_scale,
            pitch_roll_scale=pitch_roll_scale,
            velocity_std=velocity_std,
            omega_std=omega_std,
            drone_path=drone_path,
            reward_sharpness=reward_sharpness,
            action_penalty_weight=action_penalty_weight,
            num_last_quad_states=num_last_quad_states,
        )
        self.cam = DoubleSphereCamera.from_name(CameraNames.EXAMPLE)
        # the camera is looking down
        self.cam.pitch = -90.0  # degrees
        self.skip_frames = skip_frames

    @property
    def observation_space(self) -> spaces.Box:
        num_frames = int(np.ceil(self.num_last_quad_states / self.skip_frames))
        return spaces.Box(
            low=1,
            high=-1,
            shape=(2 * 7 * num_frames + 4 * self.num_last_actions,),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: EnvState, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:

        # clip action
        action = jnp.clip(
            action, self.action_space.low, self.action_space.high
        )

        # add action to last actions
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # 1 step
        dt_1 = self.delay % self.dt
        action_1 = last_actions[0]
        f_1, omega_1 = action_1[0], action_1[1:]
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, dt_1
        )

        if self.delay > 0:
            # 2 step
            dt_2 = self.dt - dt_1
            action_2 = last_actions[1]
            f_2, omega_2 = action_2[0], action_2[1:]
            quadrotor_state = self.quadrotor.step(
                quadrotor_state, f_2, omega_2, dt_2
            )

        # attach the last quadrotor states
        last_quad_states = state.last_quadrotor_states
        last_quad_states = pytree_roll(last_quad_states, shift=-1, axis=0)
        last_quad_states = pytree_at_set(last_quad_states, -1, quadrotor_state)

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quad_states,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state)
        terminated = self._is_colliding(next_state)
        truncated = jnp.greater_equal(
            next_state.step_idx, self.max_steps_in_episode
        )

        return EnvTransition(
            next_state, obs, reward, terminated, truncated, dict()
        )

    def _get_obs(self, state: EnvState, asymmetric=False) -> jax.Array:

        # get pixel coordinates of gate corners
        def project_fn(p, R):
            feature_pos = jnp.array(
                [
                    [0.5, 0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [-0.5, -0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [-0.5, 0, 0.0],
                    [0, 0.5, 0.0],
                    [0.5, 0, 0.0],
                ]
            )
            points_C = self.cam.project_points_with_pose(feature_pos, p, R)
            # remove the valid flag
            points_C = points_C[:, :2]

            points_normalized = points_C / jnp.array(
                [self.cam.width, self.cam.height], dtype=float
            ) * jnp.array([2.0, 2.0]) - jnp.array([1.0, 1.0])
            return points_normalized

        project_vmap = jax.vmap(project_fn, in_axes=(0, 0))
        points_normalized = project_vmap(
            state.last_quadrotor_states.p[:: -self.skip_frames],
            state.last_quadrotor_states.R[:: -self.skip_frames],
        )

        # add the last action of buffer to observation
        last_actions = state.last_actions.flatten()
        # normalize action to be in range [-1, 1]
        actions_low = jnp.concatenate(
            [self.action_space.low] * self.num_last_actions
        )
        actions_high = jnp.concatenate(
            [self.action_space.high] * self.num_last_actions
        )
        last_action_normalized = normalize(
            last_actions, actions_low, actions_high
        )

        obs = jnp.concatenate(
            [points_normalized.flatten(), last_action_normalized]
        )

        return obs


if __name__ == "__main__":
    key_gen = key_generator(0)

    env = HoveringFeaturesEnv()

    state, *_ = env.reset(next(key_gen))
    random_action = env.action_space.sample(next(key_gen))
    state, obs, *_ = env.step(state, random_action, next(key_gen))
    print(obs)
    state, obs, *_ = env.step(state, random_action, next(key_gen))
    print(obs)
