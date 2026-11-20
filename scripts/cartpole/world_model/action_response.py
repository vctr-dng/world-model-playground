"""Measure how world models respond to actions they were not shown.

Open-loop RMSE along logged actions (train_latent_wm.py) cannot tell whether a
model knows what an action does: a model that ignores actions can still track
logged trajectories when those actions are predictable from the state. A
planner, however, compares candidate actions, so it needs the counterfactual.

From states of the test episodes, this script applies constant forces of +1
and -1 for ``--horizon`` steps in MuJoCo and in each model (the latent model,
decoded by its probe, and the linear baseline), then compares the response,
the difference between the two outcomes, for the cart position and the pole
angle. A gain of 1 means the model moves the system as much as the physics; a
gain near 0 means the model is blind to the action.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
import mujoco
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_model_playground.cartpole import load_cartpole, positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.episodes import episode_lengths, load_episode
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.planning import decode_latents, linear_dynamics
from world_model_playground.wm.predictors import load_predictor
from world_model_playground.wm.probes import load_probe
from world_model_playground.wm.runtime import torch_device

FORCES = (1.0, -1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument(
        "--predictor",
        type=Path,
        help="latent predictor (default: artifacts/cartpole_world_model/latent_wm/"
        "<collection_id>/<embeddings>/predictor_h4_r8.pt)",
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        help="collection with the qpos/qvel of the episodes (default: the path in "
        "the embeddings manifest)",
    )
    parser.add_argument("--horizon", type=positive_int, default=30)
    parser.add_argument(
        "--start-every",
        type=positive_int,
        default=20,
        help="start counterfactuals every this many steps of each test episode",
    )
    parser.add_argument(
        "--upright-only",
        action="store_true",
        help="only start from states with cos θ > 0.9, where balancing happens",
    )
    parser.add_argument("--seed", type=int, default=0, help="episode split seed")
    parser.add_argument("--output", type=Path, help="JSON report (and a .png figure)")
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def simulate(model, data, qpos, qvel, force: float, horizon: int) -> np.ndarray:
    """Observations ``(H, 5)`` of MuJoCo after each step of a constant force."""
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    data.ctrl[:] = force
    observations = []
    for _ in range(horizon):
        mujoco.mj_step(model, data)
        # The environment observes the pole through xmat, as computed by the step.
        observations.append(
            [data.qpos[0], data.xmat[2, 8], data.xmat[2, 2], *data.qvel[:2]]
        )
    return np.asarray(observations)


def pose(observations: np.ndarray) -> np.ndarray:
    """Cart position and unwrapped pole angle ``(..., H, 2)``."""
    theta = np.unwrap(np.arctan2(observations[..., 2], observations[..., 1]), axis=-1)
    return np.stack((observations[..., 0], theta), axis=-1)


def response_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Least-squares gain of the predicted on the true response, and its RMSE.

    The gain is undefined (None) while the true response is still zero, e.g.
    for the pole angle after one step, which the observation reports one
    physics step late.
    """
    energy = float(np.dot(truth, truth))
    gain = float(np.dot(predicted, truth) / energy) if energy > 1e-10 else None
    return {
        "gain": gain,
        "rmse": float(np.sqrt(np.mean(np.square(predicted - truth)))),
    }


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = load_embeddings_manifest(args.embeddings_dir)
    collection_dir = args.collection_dir or Path(manifest["collection_dir"])
    predictor_path = args.predictor or (
        Path("artifacts/cartpole_world_model/latent_wm")
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / "predictor_h4_r8.pt"
    )
    predictor, metadata = load_predictor(predictor_path, device)
    probe, _ = load_probe(Path(metadata["params"]["probe"]), device)

    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    train_ids, _, test_ids = split_episode_ids(ids, seed=args.seed)
    train = load_embedded_episodes(
        args.embeddings_dir, train_ids, fields=("observations", "actions")
    )
    linear = linear_dynamics(
        fit_linear_dynamics(
            np.concatenate([e["observations"][:-1] for e in train.values()]),
            np.concatenate([e["actions"] for e in train.values()]),
            np.concatenate([e["observations"][1:] for e in train.values()]),
            ridge=1e-6,
        ),
        device,
    )
    test = load_embedded_episodes(
        args.embeddings_dir, test_ids, fields=("features", "observations", "actions")
    )

    mj_model = load_cartpole().mj_model
    data = mujoco.MjData(mj_model)
    history = predictor.config.history
    outcomes = {name: [] for name in ("simulator", "latent_model", "linear_model")}
    for episode_id, episode in test.items():
        physics = load_episode(collection_dir, episode_id, fields=("qpos", "qvel"))
        last_start = len(episode["actions"]) - args.horizon
        for start in range(history, last_start + 1, args.start_every):
            if args.upright_only and episode["observations"][start, 1] <= 0.9:
                continue
            first = start - history + 1
            plans = torch.tensor(FORCES, device=device).view(-1, 1, 1)
            plans = plans.expand(-1, args.horizon, 1)
            context = torch.as_tensor(
                episode["features"][None, first : start + 1], device=device
            ).expand(len(FORCES), -1, -1)
            context_actions = torch.as_tensor(
                episode["actions"][None, first:start], device=device
            ).expand(len(FORCES), -1, -1)
            with torch.no_grad():
                latents = predictor.rollout(context, context_actions, plans)
                imagined = decode_latents(probe, context, latents).cpu().numpy()
                state = torch.as_tensor(
                    episode["observations"][start], device=device
                ).expand(len(FORCES), -1)
                linear_steps = []
                for step in range(args.horizon):
                    state = linear(state, plans[:, step])
                    linear_steps.append(state)
                linear_rollout = torch.stack(linear_steps, 1).cpu().numpy()
            outcomes["simulator"].append(
                [
                    simulate(
                        mj_model,
                        data,
                        physics["qpos"][start],
                        physics["qvel"][start],
                        force,
                        args.horizon,
                    )
                    for force in FORCES
                ]
            )
            outcomes["latent_model"].append(imagined)
            outcomes["linear_model"].append(linear_rollout)
    if not outcomes["simulator"]:
        raise ValueError("no start state matched; lower --start-every")

    # Response: outcome under +1 minus outcome under -1, (starts, H, 2).
    responses = {
        name: np.diff(pose(np.asarray(values))[:, ::-1], axis=1)[:, 0]
        for name, values in outcomes.items()
    }
    truth = responses["simulator"]
    horizons = [h for h in (1, 5, 10, 20, 30, 50) if h <= args.horizon]
    report = {
        "params": {
            "embeddings_dir": str(args.embeddings_dir.resolve()),
            "predictor": str(predictor_path),
            "horizon": args.horizon,
            "start_every": args.start_every,
            "upright_only": args.upright_only,
            "num_starts": len(truth),
        },
        "simulator_response": {
            str(h): dict(
                zip(
                    ("cart_position", "pole_angle"),
                    np.abs(truth[:, h - 1]).mean(0).tolist(),
                )
            )
            for h in horizons
        },
        "models": {},
    }
    for name in ("latent_model", "linear_model"):
        predicted = responses[name]
        report["models"][name] = {
            str(h): {
                f"{quantity}_{metric}": value
                for index, quantity in enumerate(("cart_position", "pole_angle"))
                for metric, value in response_metrics(
                    predicted[:, h - 1, index], truth[:, h - 1, index]
                ).items()
            }
            for h in horizons
        }

    print(f"Response to +1 vs -1 force from {len(truth)} test states")
    print("horizon | sim |Δx| | sim |Δθ| | latent gain x, θ | linear gain x, θ")

    def gains(name: str, horizon: int) -> str:
        values = report["models"][name][str(horizon)]
        return ", ".join(
            "-" if values[f"{q}_gain"] is None else f"{values[f'{q}_gain']:.2f}"
            for q in ("cart_position", "pole_angle")
        )

    for h in horizons:
        sim = report["simulator_response"][str(h)]
        print(
            f"{h:7d} | {sim['cart_position']:.3f} | {sim['pole_angle']:.3f} | "
            f"{gains('latent_model', h)} | {gains('linear_model', h)}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        plot(responses, args.horizon, args.output.with_suffix(".png"))
        print(f"Report: {args.output}")


def plot(responses: dict, horizon: int, path: Path) -> None:
    steps = np.arange(1, horizon + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for index, label in enumerate(
        ("cart position (+1 minus -1)", "pole angle (+1 minus -1)")
    ):
        for name, values in responses.items():
            mean = values[..., index].mean(axis=0)
            spread = values[..., index].std(axis=0)
            axes[index].plot(steps, mean, label=name)
            axes[index].fill_between(steps, mean - spread, mean + spread, alpha=0.15)
        axes[index].set(
            title=f"Response of the {label}", xlabel="steps of constant force"
        )
    axes[0].legend()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
