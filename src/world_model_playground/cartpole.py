"""Shared CartpoleBalance environment and visualization helpers."""

import argparse
import time
from collections.abc import Callable, Sequence
from typing import Any, Self

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src.mjx_env import MjxEnv

# Frame size whose horizontal camera view every rendered resolution reproduces.
REFERENCE_HEIGHT, REFERENCE_WIDTH = 240, 320


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


def parse_resolution(value: str) -> tuple[int, int]:
    """Parses ``HEIGHTxWIDTH`` (or a single side for square frames) into a size."""
    try:
        sides = tuple(int(side) for side in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must look like 64x64 or 64") from error
    if len(sides) == 1:
        sides = sides * 2
    if len(sides) != 2 or min(sides) < 1:
        raise argparse.ArgumentTypeError("must look like 64x64 or 64")
    return sides


def matched_fovy(fovy: float, height: int, width: int) -> float:
    """Vertical field of view, in degrees, keeping the horizontal view of a 320x240 frame.

    MuJoCo's ``fovy`` is vertical, so square crops of the ``fixed`` camera would
    otherwise lose the rail ends and hide the cart near the slider limits.
    """
    half_tan = np.tan(np.deg2rad(fovy) / 2) * (REFERENCE_WIDTH / REFERENCE_HEIGHT)
    return float(np.rad2deg(2 * np.arctan(half_tan * height / width)))


class FrameRenderer:
    """Renders rows of ``qpos``/``qvel`` at fixed resolutions of one camera.

    Every resolution keeps the reference horizontal field of view (see
    ``matched_fovy``). One MuJoCo renderer per resolution is created up front and
    reused, since creating renderers costs far more than rendering a frame.
    Close it (or use it as a context manager): a garbage-collected renderer can
    free a GL context another renderer still uses, which corrupts its frames.
    Rendering is much faster headless, with ``MUJOCO_GL=egl``, than through GLFW.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        resolutions: Sequence[tuple[int, int]],
        camera: str,
    ):
        self._model = mj_model
        self._data = mujoco.MjData(mj_model)
        self._camera = camera
        self._camera_id = mj_model.camera(camera).id
        self._fovy = float(mj_model.cam_fovy[self._camera_id])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        try:
            for height, width in dict.fromkeys(resolutions):
                self._renderers[height, width] = mujoco.Renderer(
                    mj_model, height=height, width=width
                )
        except Exception:
            self.close()
            raise

    def fovy(self, height: int, width: int) -> float:
        """Vertical field of view used for frames of this resolution."""
        return matched_fovy(self._fovy, height, width)

    def render(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray | None = None,
        *,
        resolution: tuple[int, int],
    ) -> np.ndarray:
        height, width = resolution
        renderer = self._renderers[height, width]
        frames = np.empty((len(qpos), height, width, 3), dtype=np.uint8)
        try:
            self._model.cam_fovy[self._camera_id] = self.fovy(height, width)
            for index, positions in enumerate(qpos):
                self._data.qpos[:] = positions
                if qvel is not None:
                    self._data.qvel[:] = qvel[index]
                mujoco.mj_forward(self._model, self._data)
                renderer.update_scene(self._data, camera=self._camera)
                frames[index] = renderer.render()
        finally:
            self._model.cam_fovy[self._camera_id] = self._fovy
        return frames

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def render_frames(
    mj_model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
    *,
    height: int,
    width: int,
    camera: str,
) -> np.ndarray:
    """Renders rows of ``qpos``/``qvel`` once; prefer ``FrameRenderer`` in loops."""
    with FrameRenderer(mj_model, [(height, width)], camera) as renderer:
        return renderer.render(qpos, qvel, resolution=(height, width))


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
