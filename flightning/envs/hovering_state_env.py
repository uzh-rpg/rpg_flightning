import csv
import functools
from typing import Optional

import chex
import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from tqdm import tqdm

from flightning.objects import Quadrotor, QuadrotorState, WorldBox
from flightning.utils import math as math_utils
from flightning.utils import spaces
from flightning.utils.pytrees import pytree_get_item, stack_pytrees
from flightning.utils.math import smooth_l1, rot_to_quat
from flightning.utils.random import random_rotation, key_generator
import flightning.envs.env_base as env_base
from flightning.envs.env_base import EnvTransition


@jdc.pytree_dataclass
class EnvState(env_base.EnvState):
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    # last_quadrotor_states: QuadrotorState
    # queues last actions to handle delay
    # last_actions[-1] is the most recent action
    # last_actions[0] is the oldest action
    last_actions: jax.Array
    last_quadrotor_states: QuadrotorState


class HoveringStateEnv(env_base.Env[EnvState]):
    """State-based hovering environment."""

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
        margin=0.5,
    ):
        self.goal: jnp.ndarray = jnp.array([0.0, 0.0, 1.0])
        self.world_box = WorldBox(
            jnp.array([-5.0, -5.0, 0.0]), jnp.array([5.0, 5.0, 3.0])
        )
        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)
        # random state parameters
        self.yaw_scale = yaw_scale
        self.pitch_roll_scale = pitch_roll_scale
        self.velocity_std = velocity_std
        self.omega_std = omega_std
        # quadrotor
        if drone_path is not None:
            self.quadrotor = Quadrotor.from_yaml(drone_path)
        else:
            self.quadrotor = Quadrotor.default_quadrotor()
        self.omega_min = self.quadrotor._omega_max * -1
        self.omega_max = self.quadrotor._omega_max
        self.thrust_min = self.quadrotor._thrust_min
        self.thrust_max = self.quadrotor._thrust_max
        self.v_min = jnp.array([-10.0, -10.0, -10.0])
        self.v_max = jnp.array([10.0, 10.0, 10.0])

        assert delay >= 0.0, "Delay must be non-negative"
        self.delay = np.array(delay)
        self.num_last_actions = int(np.ceil(delay / dt)) + 1

        self.reward_sharpness = reward_sharpness
        self.action_penalty_weight = action_penalty_weight

        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        self.num_last_quad_states = num_last_quad_states
        self.margin = margin

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[EnvState] = None
    ) -> tuple[EnvState, jax.Array]:

        key_p, key_R, key_v, key_omega, key_dr = jax.random.split(key, 5)
        p = jax.random.uniform(
            key_p,
            shape=(3,),
            minval=self.world_box.min + self.margin,
            maxval=self.world_box.max - self.margin,
        )

        rot = random_rotation(
            key_R, self.yaw_scale, self.pitch_roll_scale, self.pitch_roll_scale
        )
        R = rot.as_matrix()
        v = self.velocity_std * jax.random.normal(key_v, shape=(3,))

        omega = self.omega_std * jax.random.normal(key_omega, shape=(3,))

        quadrotor_state = self.quadrotor.create_state(
            p=p, R=R, v=v, omega=omega, dr_key=key_dr
        )

        last_actions = jnp.tile(
            self.hovering_action, (self.num_last_actions, 1)
        )

        last_quadrotor_states = [quadrotor_state] * self.num_last_quad_states
        last_quadrotor_states = stack_pytrees(last_quadrotor_states)

        # noinspection PyArgumentList
        state = EnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
        )

        obs = self._get_obs(state)
        return state, obs

    def _get_obs(self, state: EnvState) -> jax.Array:
        return jnp.concatenate(
            [
                state.quadrotor_state.p,
                math_utils.vec(state.quadrotor_state.R),
                state.quadrotor_state.v,
                state.last_actions.flatten(),
            ]
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

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
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

    def _get_reward(
        self, last_state: EnvState, next_state: EnvState
    ) -> jax.Array:
        action = next_state.last_actions[-1]
        p = next_state.quadrotor_state.p
        acc = next_state.quadrotor_state.acc

        # compute lipschitz continuous reward
        pos_cost = (
            smooth_l1(self.reward_sharpness * (p - self.goal))
            / self.reward_sharpness
        )
        vel_cost = 0.1 * smooth_l1(next_state.quadrotor_state.v)
        omega_cost = 0.1 * smooth_l1(next_state.quadrotor_state.omega)
        acc_cost = 0.1 * smooth_l1(acc)
        goal_cost = pos_cost + vel_cost + omega_cost + acc_cost

        action_cost = smooth_l1(action - self.hovering_action)
        action_cost = self.action_penalty_weight * action_cost

        # action and smoothness penalty
        action_last = last_state.last_actions[-1]
        smoothness_cost = 0.01 * jnp.inner(action, action_last)

        cost = goal_cost + action_cost + smoothness_cost

        # penalize collision
        time_left = self.max_steps_in_episode - next_state.step_idx
        collision_cost = jax.lax.select(
            self._is_colliding(next_state), time_left * cost, 0.0
        )
        # cost += collision_cost
        cost += jax.lax.stop_gradient(collision_cost)

        # scale by time
        reward = -self.dt * cost
        return reward

    def _is_colliding(self, state: EnvState) -> jax.Array:
        quad_state = state.quadrotor_state
        world_box = self.world_box
        world_collision = jnp.logical_not(world_box.contains(quad_state.p))
        is_colliding = world_collision
        return is_colliding

    @property
    def action_space(self) -> spaces.Box:
        low = jnp.concatenate(
            [jnp.array([self.thrust_min * 4]), self.omega_min]
        )
        high = jnp.concatenate(
            [jnp.array([self.thrust_max * 4]), self.omega_max]
        )
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        n = self.num_last_actions
        action_high_repeated = jnp.concatenate([self.action_space.high] * n)
        action_low_repeated = jnp.concatenate([self.action_space.low] * n)

        return spaces.Box(
            low=jnp.concatenate(
                [
                    self.world_box.min,
                    -jnp.ones(9),
                    self.v_min,
                    action_low_repeated,
                ]
            ),
            high=jnp.concatenate(
                [
                    self.world_box.max,
                    jnp.ones(9),
                    self.v_max,
                    action_high_repeated,
                ]
            ),
            shape=(15 + n * 4,),
        )

    @classmethod
    def generate_csv(cls, traj: EnvTransition, filename: str):

        num_dim = traj.reward.ndim
        if num_dim == 1:
            pytree_get_item(traj, None)
        num_trajectories = traj.reward.shape[0]
        for i in tqdm(range(num_trajectories)):
            traj_i = pytree_get_item(traj, i)
            cls._generate_csv(traj_i, f"{filename}_{i}.csv")

    @staticmethod
    def _generate_csv(traj: EnvTransition, filename: str):
        done = jnp.logical_or(traj.terminated, traj.truncated)
        traj_length = jnp.where(done)[0][0].item() + 1

        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "index",
                "t",
                "px",
                "py",
                "pz",
                "qw",
                "qx",
                "qy",
                "qz",
                "vx",
                "vy",
                "vz",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            rows = []

            writer.writeheader()
            for i in range(traj_length):
                transition = pytree_get_item(traj, i)
                t = transition.state.time
                p = transition.state.quadrotor_state.p
                R = transition.state.quadrotor_state.R
                quat = rot_to_quat(Rotation.from_matrix(R))
                row_dict = {
                    "index": i,
                    "t": t,
                    "px": p[0],
                    "py": p[1],
                    "pz": p[2],
                    "qw": quat[0],
                    "qx": quat[1],
                    "qy": quat[2],
                    "qz": quat[3],
                    "vx": transition.state.quadrotor_state.v[0],
                    "vy": transition.state.quadrotor_state.v[1],
                    "vz": transition.state.quadrotor_state.v[2],
                }
                rows.append(row_dict)

            writer.writerows(rows)

    def plot_trajectories(self, traj: EnvTransition):
        assert traj.reward.ndim == 2
        num_trajs = traj.reward.shape[0]
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)

        for i in range(num_trajs):
            # find first index where truncated or terminated is true
            idx = np.where(done[i])[0][0].item() + 1
            x = state.quadrotor_state.p[i, :idx, 0]
            y = state.quadrotor_state.p[i, :idx, 1]
            z = state.quadrotor_state.p[i, :idx, 2]
            R = state.quadrotor_state.R[i, :idx]
            t = state.time[i, :idx]
            ax1.plot(x, y)
            if i == 0:
                ax1.scatter(x[0], y[0], color="green", label="start")
                ax1.scatter(x[-1], y[-1], color="red", label="end")
            else:
                ax1.scatter(x[0], y[0], color="green")
                ax1.scatter(x[-1], y[-1], color="red")
            # show the orientation of the drone every n_orient steps
            n_orient = 10
            ax1.quiver(
                x[::n_orient],
                y[::n_orient],
                R[::n_orient, 0, 0],
                R[::n_orient, 1, 0],
                color="blue",
                scale=8.0,
            )

            # label the axes with x and y
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.legend()

            ax2.plot(t, z)
            ax2.set_ylabel("z")
            ax2.set_xlabel("Time")
        fig.show()


if __name__ == "__main__":
    key_gen = key_generator(0)

    env = HoveringStateEnv()

    state, *_ = env.reset(next(key_gen))
    random_action = env.action_space.sample(next(key_gen))
    state, obs, *_ = env.step(state, random_action, next(key_gen))
    print(obs)
    state, obs, *_ = env.step(state, random_action, next(key_gen))
    print(obs)
