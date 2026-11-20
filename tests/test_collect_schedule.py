"""Tests for the action-source schedule of collect_obs.py."""

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "cartpole"
    / "world_model"
    / "collect_obs.py"
)
_spec = importlib.util.spec_from_file_location("collect_obs", _SCRIPT)
collect_obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_obs)


def test_held_schedule_keeps_sources_and_random_actions_over_segments():
    sources, uniform = collect_obs.held_schedule(4, 25, (0.4, 0.3), 10, seed=0)

    assert sources.shape == (4, 25) and uniform.shape == (4, 25, 1)
    for row_sources, row_uniform in zip(sources, uniform[..., 0], strict=True):
        for start in range(0, 25, 10):
            segment = slice(start, start + 10)
            assert len(set(row_sources[segment])) == 1
            assert len(set(row_uniform[segment])) == 1
    # Exact proportions over the 4 x 3 segments: 5 / 4 / 3 (rounded to the largest
    # remainders), each held for its segment's steps.
    segment_sources = sources[:, ::10]
    assert np.bincount(segment_sources.ravel(), minlength=3).tolist() == [5, 4, 3]


def test_held_schedule_with_a_hold_of_one_is_per_step():
    sources, uniform = collect_obs.held_schedule(2, 200, (0.4, 0.3), 1, seed=0)

    assert np.bincount(sources.ravel()).tolist() == [160, 120, 120]
    assert len(np.unique(uniform)) == uniform.size
