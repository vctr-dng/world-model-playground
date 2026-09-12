"""Shared CartpoleBalance environment and visualization helpers."""

import argparse
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src.mjx_env import MjxEnv


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def load_cartpole(*, full_circle_reset: bool = False) -> MjxEnv:
    """Loads CartpoleBalance, optionally resetting across the full pole circle."""
    config = registry.get_default_config("CartpoleBalance")
    config.impl = "jax"
    env = registry.load("CartpoleBalance", config=config)

    if full_circle_reset:
        slider_qposadr = env.mj_model.joint("slider").qposadr[0]
        hinge_qposadr = env.mj_model.joint("hinge_1").qposadr[0]

        def reset_full_circle(rng: jax.Array) -> jax.Array:
            cart_rng, pole_rng = jax.random.split(rng)
            qpos = jnp.zeros(env.mjx_model.nq)
            qpos = qpos.at[slider_qposadr].set(
                jax.random.uniform(cart_rng, (), minval=-0.1, maxval=0.1)
            )
            return qpos.at[hinge_qposadr].set(
                jax.random.uniform(pole_rng, (), minval=-jnp.pi, maxval=jnp.pi)
            )

        env._reset_randomize = reset_full_circle  # pyright: ignore[reportPrivateUsage]

    return env


def run_live_rollout[Carry](
    env: MjxEnv,
    carry: Carry,
    step_fn: Callable[[Carry], Carry],
    get_state: Callable[[Carry], Any],
    num_steps: int,
    state_index: int | None = None,
) -> tuple[Carry, int, float]:
    """Steps a rollout while showing one state in MuJoCo's live viewer."""
    render_data = mujoco.MjData(env.mj_model)
    completed_steps = 0
    start = time.perf_counter()

    with mujoco.viewer.launch_passive(env.mj_model, render_data) as viewer:
        for _ in range(num_steps):
            if not viewer.is_running():
                break

            step_start = time.perf_counter()
            carry = step_fn(carry)
            state = get_state(carry)
            qpos = state.data.qpos
            qvel = state.data.qvel
            if state_index is not None:
                qpos = qpos[state_index]
                qvel = qvel[state_index]
            qpos, qvel = jax.device_get((qpos, qvel))

            np.copyto(render_data.qpos, qpos)
            np.copyto(render_data.qvel, qvel)
            mujoco.mj_forward(env.mj_model, render_data)
            viewer.sync()
            completed_steps += 1

            remaining = env.dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    return carry, completed_steps, time.perf_counter() - start
