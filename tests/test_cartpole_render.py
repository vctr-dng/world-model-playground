"""Tests for rendering CartPole physics states at arbitrary resolutions."""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from world_model_playground import cartpole


def test_parse_resolution_accepts_square_and_rectangular_sizes():
    assert cartpole.parse_resolution("64") == (64, 64)
    assert cartpole.parse_resolution("240x320") == (240, 320)
    for invalid in ("0x64", "64x", "a", "1x2x3"):
        with pytest.raises(argparse.ArgumentTypeError):
            cartpole.parse_resolution(invalid)


def test_matched_fovy_keeps_the_reference_horizontal_view():
    assert cartpole.matched_fovy(45.0, 240, 320) == pytest.approx(45.0)
    square = cartpole.matched_fovy(45.0, 64, 64)
    # A square frame needs a taller vertical view to keep the same width.
    np.testing.assert_allclose(
        np.tan(np.deg2rad(square) / 2), np.tan(np.deg2rad(45.0) / 2) * 320 / 240
    )


def test_frame_renderer_shows_the_cart_at_both_rail_ends():
    env = cartpole.load_cartpole()
    # Upright pole with the cart at the left end, centre, and right end of the rail.
    qpos = np.array([[-1.8, 0.0], [0.0, 0.0], [1.8, 0.0]])
    try:
        renderer = cartpole.FrameRenderer(env.mj_model, [(64, 64), (32, 48)], "fixed")
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"MuJoCo rendering unavailable: {error}")
    with renderer:
        square = renderer.render(qpos, resolution=(64, 64))
        wide = renderer.render(qpos, resolution=(32, 48))

    assert square.shape == (3, 64, 64, 3) and square.dtype == np.uint8
    assert wide.shape == (3, 32, 48, 3)
    # The cart is orange (red > blue) on a blue background; it must stay in frame.
    orange = square[..., 0].astype(int) - square[..., 2] > 60
    columns = [np.flatnonzero(mask.any(axis=0)) for mask in orange]
    assert all(len(found) for found in columns)
    assert columns[0].mean() < columns[1].mean() < columns[2].mean()
    assert float(env.mj_model.cam_fovy[env.mj_model.camera("fixed").id]) == 45.0
