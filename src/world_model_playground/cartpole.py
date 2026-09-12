"""Shared CartpoleBalance environment and visualization helpers."""

import argparse
import time
from collections.abc import Callable
from typing import Any

import jax
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


def load_cartpole() -> MjxEnv:
    """Loads CartpoleBalance using the MJX-JAX implementation."""
    config = registry.get_default_config("CartpoleBalance")
    config.impl = "jax"
    return registry.load("CartpoleBalance", config=config)


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
