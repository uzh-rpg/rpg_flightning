from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState

from flightning.envs.env_base import Env, EnvState
from flightning.envs.wrappers import LogWrapper, VecEnv


class TrajectoryState(PyTreeNode):
    reward: jnp.array


def progress_callback_host(episode_loss):
    episode, loss = episode_loss
    print(f"Episode: {episode}, Loss: {loss:.2f}")


NUM_EPOCHS_PER_CALLBACK = 10


def progress_callback(episode, loss):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            progress_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, loss),
    )


def grad_callback_host(episode_grad):
    episode, grad = episode_grad
    print(f"Episode: {episode}, Grad max: {grad:.4f}")


def grad_callback(episode, grad_norm):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            grad_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, grad_norm),
    )


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


def train(
    env: Env,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    key: chex.PRNGKey,
):

    env = LogWrapper(env)
    env = VecEnv(env)

    def _train(runner_state: RunnerState):
        def epoch_fn(epoch_state: RunnerState, _unused):

            @partial(jax.value_and_grad, has_aux=True)
            def loss_fn(params, runner_state: RunnerState):

                def rollout(runner_state: RunnerState):
                    def step_fn(old_runner_state: RunnerState, _unsused):

                        # extract states and obs
                        train_state, env_state, last_obs, key, epoch_idx = (
                            old_runner_state
                        )

                        # get action
                        action = train_state.apply_fn(params, last_obs)

                        # env step
                        key, key_ = jax.random.split(key)
                        key_step = jax.random.split(key_, num_envs)
                        (
                            env_state,
                            obs,
                            reward,
                            _terminated,
                            _truncated,
                            info,
                        ) = env.step(env_state, action, key_step)
                        runner_state = RunnerState(
                            train_state, env_state, obs, key, epoch_idx
                        )

                        return (
                            runner_state,
                            TrajectoryState(reward=reward),
                        )

                    runner_state, trajectory = jax.lax.scan(
                        step_fn, runner_state, None, num_steps_per_epoch
                    )
                    return runner_state, trajectory

                # collect data
                runner_state, trajectory = rollout(runner_state)
                loss = -trajectory.reward.sum() / num_envs
                return loss, runner_state

            # compute reward
            train_state = epoch_state.train_state
            (loss, epoch_state), grad = loss_fn(
                train_state.params, epoch_state
            )
            # update params
            train_state = train_state.apply_gradients(grads=grad)

            # calc stats on grad
            leaves = jax.tree_util.tree_leaves(grad)
            flattened_leaves = [jnp.ravel(leaf) for leaf in leaves]
            grad_vec = jnp.concatenate(flattened_leaves)
            grad_max = jnp.max(jnp.abs(grad_vec))

            progress_callback(epoch_state.epoch_idx, loss)
            grad_callback(epoch_state.epoch_idx, grad_max)
            epoch_state = epoch_state._replace(
                train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
            )

            return epoch_state, loss

        # run epochs
        runner_state_final, losses = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )

        return {"runner_state": runner_state_final, "metrics": losses}

    # intialize environments
    key, key_ = jax.random.split(key)
    key_reset = jax.random.split(key_, num_envs)
    env_state, obs = env.reset(key_reset, None)
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    return jax.jit(_train)(runner_state)


if __name__ == "__main__":
    # Example usage
    from flax.linen import Dense
    from flightning.envs.simple_hovering_env import SimpleHoveringEnv
    import optax

    x = jnp.zeros(15)
    model = Dense(4)
    params = model.init(jax.random.key(0), x)
    tx = optax.adam(1e-3)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    env = SimpleHoveringEnv()

    res_dict = train(
        env,
        train_state,
        num_envs=10,
        num_epochs=10,
        num_steps_per_epoch=10,
        key=jax.random.key(0),
    )

    loss = res_dict["metrics"]
    print(f"Final loss: {loss[-1]}")
