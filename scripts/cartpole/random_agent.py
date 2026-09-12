"""Run a random agent in parallel MJX CartpoleBalance environments."""

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from world_model_playground.cartpole import (
    load_cartpole,
    non_negative_int,
    positive_int,
    run_live_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=positive_int, default=1024)
    parser.add_argument("--num-steps", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        action="store_true",
        help="show one environment in a live MuJoCo viewer",
    )
    parser.add_argument(
        "--render-env",
        type=non_negative_int,
        default=0,
        help="index of the parallel environment to display",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_env >= args.num_envs:
        raise ValueError("--render-env must be smaller than --num-envs")

    env = load_cartpole()
    _, reset_rng, rollout_rng = jax.random.split(jax.random.key(args.seed), 3)
    reset_keys = jax.random.split(reset_rng, args.num_envs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    state = jax.block_until_ready(reset(reset_keys))

    def random_step(state, rng, returns, done_count):
        rng, action_rng = jax.random.split(rng)
        actions = jax.random.uniform(
            action_rng,
            (args.num_envs, env.action_size),
            minval=-1.0,
            maxval=1.0,
        )
        state = step(state, actions)
        return (
            state,
            rng,
            returns + state.reward,
            done_count + jnp.sum(state.done),
        )

    initial_carry = (
        state,
        rollout_rng,
        jnp.zeros(args.num_envs),
        jnp.zeros(()),
    )

    if args.render:
        compile_start = time.perf_counter()
        compiled_step = jax.jit(random_step).lower(*initial_carry).compile()
        compile_seconds = time.perf_counter() - compile_start
        carry, completed_steps, rollout_seconds = run_live_rollout(
            env,
            initial_carry,
            lambda values: compiled_step(*values),
            lambda values: values[0],
            args.num_steps,
            state_index=args.render_env,
        )
        _, _, returns, done_count = jax.device_get(carry)
    else:

        def rollout(carry):
            def scan_step(carry, _):
                return random_step(*carry), None

            return jax.lax.scan(scan_step, carry, xs=None, length=args.num_steps)[0]

        compile_start = time.perf_counter()
        compiled_rollout = jax.jit(rollout).lower(initial_carry).compile()
        compile_seconds = time.perf_counter() - compile_start
        rollout_start = time.perf_counter()
        _, _, returns, done_count = jax.block_until_ready(
            compiled_rollout(initial_carry)
        )
        rollout_seconds = time.perf_counter() - rollout_start
        completed_steps = args.num_steps

    transitions = args.num_envs * completed_steps
    throughput = transitions / rollout_seconds
    print(f"Device: {jax.devices()[0]}")
    print("Environment: CartpoleBalance (MJX-JAX)")
    print(f"Parallel environments: {args.num_envs:,}")
    print(f"Steps per environment: {completed_steps:,}")
    print(f"Compilation: {compile_seconds:.2f} s")
    print(f"Rollout: {rollout_seconds:.3f} s")
    print(f"Throughput: {throughput:,.0f} transitions/s")
    print(f"Mean return: {float(np.mean(returns)):.3f}")
    print(f"Terminations: {int(done_count)}")


if __name__ == "__main__":
    main()
