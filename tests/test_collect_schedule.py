"""Tests for the action-source schedule of collect_obs.py."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from world_model_playground.cartpole import load_cartpole

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


def zero_policy(observations, key):
    del key
    return jnp.zeros((observations.shape[0], 1), observations.dtype), {}


def run_rollout(probability: float, *, full_circle: bool = False, steps: int = 60):
    env = load_cartpole(full_circle_reset=full_circle)
    rollout = collect_obs.make_batched_rollout(
        env,
        zero_policy,
        zero_policy,
        perturb_probability=probability,
        perturb_steps=(5, 8),
        perturb_upright=0.9,
    )
    batch = 6
    sources = jnp.zeros((batch, steps), dtype=jnp.uint8)
    random_actions = jnp.zeros((batch, steps, 1))
    keys = jax.random.split(jax.random.key(0), batch)
    _, records = jax.device_get(
        rollout(keys, sources, random_actions, jax.random.key(1))
    )
    return records


def test_perturbations_hold_one_force_for_the_requested_length():
    records = run_rollout(probability=1.0)
    perturbation = collect_obs._ACTION_SOURCES.index("perturbation")

    for sources, actions in zip(records["sources"], records["actions"][..., 0]):
        # Balance resets start upright, so a perturbation starts at step 0.
        assert sources[0] == perturbation
        flags = np.concatenate(([0], (sources == perturbation).astype(int), [0]))
        starts = np.flatnonzero(np.diff(flags) == 1)
        stops = np.flatnonzero(np.diff(flags) == -1)
        for start, stop in zip(starts, stops, strict=True):
            # With probability 1 perturbations follow each other without a gap,
            # so a run is split where the held force changes.
            run = actions[start:stop]
            cuts = np.flatnonzero(np.diff(run)) + 1
            lengths = np.diff(np.concatenate(([0], cuts, [len(run)])))
            assert np.all(run != 0.0)
            # The last piece may be cut short by the end of the episode.
            truncated = stop == len(actions)
            inner = lengths[:-1] if truncated else lengths
            assert np.all((inner >= 5) & (inner <= 8)), lengths
        # Outside perturbations the scheduled (zero) policy acts.
        assert np.all(actions[sources != perturbation] == 0.0)


def test_perturbations_only_start_upright_and_are_off_by_default():
    assert not np.any(run_rollout(probability=0.0)["sources"])
    hanging = run_rollout(probability=1.0, full_circle=True, steps=3)
    perturbation = collect_obs._ACTION_SOURCES.index("perturbation")
    upright_at_start = hanging["observations"][:, 0, 1] > 0.9
    started = hanging["sources"][:, 0] == perturbation
    # A perturbation can only start from a state that was upright.
    assert not np.any(
        started & ~upright_at_start & (hanging["observations"][:, 0, 1] < 0.5)
    )
