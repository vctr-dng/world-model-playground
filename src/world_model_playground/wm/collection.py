"""Action-source schedules and the batched rollout behind ``collect_obs.py``.

Collections mix actions from the trained policy (deterministic and stochastic),
uniformly random actions, and forces that perturb an upright pole. The schedule
fixes exact proportions of each source per segment of steps, and the rollout
follows it in parallel MJX environments.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Scheduled sources come first; perturbations override them while upright.
ACTION_SOURCES = ("deterministic", "stochastic", "random", "perturbation")
PERTURBATION = ACTION_SOURCES.index("perturbation")


def action_counts(
    num_transitions: int, deterministic_fraction: float, stochastic_fraction: float
) -> np.ndarray:
    """Apportions the requested transition count across the three action sources."""
    random_fraction = 1.0 - deterministic_fraction - stochastic_fraction
    if random_fraction < -1e-12:
        raise ValueError(
            "--deterministic-fraction and --stochastic-fraction must sum to at most 1"
        )
    random_fraction = max(0.0, random_fraction)

    requested = (
        np.array([deterministic_fraction, stochastic_fraction, random_fraction])
        * num_transitions
    )
    counts = np.floor(requested).astype(np.int64)
    remainder = num_transitions - int(counts.sum())
    for index in np.argsort(-(requested - counts), kind="stable")[:remainder]:
        counts[index] += 1
    return counts


def shuffled_action_sources(counts: np.ndarray, seed: int) -> np.ndarray:
    """Returns an exact, reproducibly shuffled action-source schedule."""
    sources = np.repeat(np.arange(len(counts), dtype=np.uint8), counts)
    np.random.default_rng(seed).shuffle(sources)
    return sources


def held_schedule(
    num_episodes: int,
    episode_length: int,
    fractions: tuple[float, float],
    hold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Action-source schedule ``(E, T)`` and random actions ``(E, T, 1)``.

    Sources are drawn per segment of ``hold`` steps, with exact proportions over
    the segments, and each random segment holds one uniform action.
    """
    segments = -(-episode_length // hold)
    counts = action_counts(num_episodes * segments, *fractions)
    sources = shuffled_action_sources(counts, seed).reshape(num_episodes, segments)
    uniform = np.random.default_rng(seed).uniform(
        -1.0, 1.0, size=(num_episodes, segments, 1)
    )

    def per_step(values):
        return np.repeat(values, hold, axis=1)[:, :episode_length]

    return per_step(sources), per_step(uniform).astype(np.float32)


def make_batched_rollout(
    env,
    deterministic_policy,
    stochastic_policy,
    *,
    perturb_probability: float = 0.0,
    perturb_steps: tuple[int, int] = (10, 30),
    perturb_upright: float = 0.9,
):
    """Builds a jitted rollout of parallel episodes following a source schedule.

    The returned function maps reset keys ``(B,)``, a schedule ``(B, T)`` of
    action-source indices, the random actions ``(B, T, 1)`` used by the random
    source, and a key to the initial and per-step arrays, all batch-major:
    ``initial`` has ``(B, ...)`` and ``steps`` ``(B, T, ...)``.

    While the pole is upright (``cos θ > perturb_upright``), an episode that is
    not already perturbed starts a perturbation with ``perturb_probability``
    per step: one uniform force held for ``perturb_steps`` (inclusive range)
    over the scheduled actions, which then have to recover the pole. Planning
    models need such sustained forces near balance to learn what actions do.
    ``steps["sources"]`` records the source of every executed action.
    """
    shortest, longest = perturb_steps

    def rollout(reset_keys, sources, random_actions, rng):
        state = jax.vmap(env.reset)(reset_keys)
        batch = sources.shape[0]
        remaining = jnp.zeros(batch, dtype=jnp.int32)
        force = jnp.zeros((batch, 1), dtype=state.obs.dtype)

        def step(carry, inputs):
            state, remaining, force = carry
            source, uniform, key = inputs
            keys = jax.random.split(key, 5)
            deterministic = deterministic_policy(state.obs, keys[0])[0]
            stochastic = stochastic_policy(state.obs, keys[1])[0]
            scheduled = source[:, None]
            action = jnp.where(
                scheduled == 0,
                deterministic,
                jnp.where(scheduled == 1, stochastic, uniform),
            )
            start = (
                (remaining == 0)
                & (state.obs[:, 1] > perturb_upright)
                & (jax.random.uniform(keys[2], (batch,)) < perturb_probability)
            )
            remaining = jnp.where(
                start,
                jax.random.randint(keys[3], (batch,), shortest, longest + 1),
                remaining,
            )
            force = jnp.where(
                start[:, None],
                jax.random.uniform(keys[4], force.shape, force.dtype, -1.0, 1.0),
                force,
            )
            perturbed = remaining > 0
            action = jnp.where(perturbed[:, None], force, action)
            next_state = jax.vmap(env.step)(state, action)
            record = {
                "observations": next_state.obs,
                "actions": action,
                "sources": jnp.where(perturbed, PERTURBATION, source),
                "rewards": next_state.reward,
                "dones": next_state.done,
                "qpos": next_state.data.qpos,
                "qvel": next_state.data.qvel,
            }
            return (next_state, jnp.maximum(remaining - 1, 0), force), record

        step_keys = jax.random.split(rng, sources.shape[1])
        _, steps = jax.lax.scan(
            step,
            (state, remaining, force),
            (sources.T, jnp.swapaxes(random_actions, 0, 1), step_keys),
        )
        initial = {
            "observations": state.obs,
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
        }
        steps = jax.tree.map(lambda values: jnp.swapaxes(values, 0, 1), steps)
        return initial, steps

    return jax.jit(rollout)
