"""Tests for the replanning cadence and strided context of the MPC controllers."""

import numpy as np
import pytest
import torch

from world_model_playground.wm import ac_world_model, control
from world_model_playground.wm.encoders import EncoderOutput
from world_model_playground.wm.planning import CEMConfig, CEMPlanner
from world_model_playground.wm.probes import StateProbe

TOKENS, CHANNELS = 4, 16


class StubRenderer:
    """Frames whose every pixel encodes the cart position."""

    def render(self, qpos, qvel, *, resolution):
        values = np.clip(np.round(100 + 50 * qpos[:, 0]), 0, 255).astype(np.uint8)
        return np.broadcast_to(
            values[:, None, None, None], (len(qpos), *resolution, 3)
        ).copy()


class StubEncoder:
    """Tokens that are a fixed function of the frame, as a frozen encoder's are."""

    context_frames = 1

    def __init__(self):
        generator = torch.Generator().manual_seed(5)
        self.basis = torch.randn(TOKENS, CHANNELS, generator=generator)

    def __call__(self, clips):
        level = clips.float().mean(dim=(1, 2, 3, 4))[:, None, None] / 255
        tokens = self.basis * level + level
        return EncoderOutput(tokens.mean(1), tokens)


def tiny_model():
    torch.manual_seed(0)
    config = ac_world_model.ACConfig(
        step_stride=2,
        history=2,
        pretrained=False,
        predictor={
            "img_size": (32, 32),
            "patch_size": 16,
            "num_frames": 8,
            "tubelet_size": 2,
            "embed_dim": CHANNELS,
            "predictor_embed_dim": 48,
            "depth": 1,
            "num_heads": 2,
        },
    )
    return ac_world_model.build_ac_model(config).eval()


def test_latent_mpc_plans_every_model_step_from_strided_context(monkeypatch):
    model = tiny_model()
    torch.manual_seed(3)
    probe = StateProbe(2 * CHANNELS, history=2, stride=2).eval()
    encoder = StubEncoder()
    calls = []
    real_evaluator = control.latent_evaluator

    def recording_evaluator(predictor, probe, context, context_actions, **kwargs):
        calls.append((context.clone(), context_actions.clone()))
        return real_evaluator(predictor, probe, context, context_actions, **kwargs)

    monkeypatch.setattr(control, "latent_evaluator", recording_evaluator)
    config = CEMConfig(horizon=4, candidates=4, iterations=1, elites=2, action_repeat=2)
    controller = control.LatentMPC(
        CEMPlanner(config, batch_size=2, shift=2, seed=0),
        encoder=encoder,
        renderer=StubRenderer(),
        resolution=(32, 32),
        predictor=model,
        probe=probe,
        objective="goal",
        goal=torch.zeros(TOKENS, CHANNELS),
        replan_every=2,
    )
    controller.reset()
    qpos = [np.array([[0.1 * t, 0.0], [-0.1 * t, 0.0]]) for t in range(6)]
    actions, infos = [], []
    for t in range(6):
        action, info = controller.act(None, np.zeros((2, 5)), qpos[t], np.zeros((2, 2)))
        actions.append(action)
        infos.append(info)

    assert [bool(i["replanned"][0]) for i in infos] == [True, False] * 3
    assert all(i["plan_seconds"][0] == 0 for i in infos[1::2])
    assert infos[0]["plan_observations"].shape == (2, 2, 5)  # H / stride steps
    # The plan made at step t predicts the latent of step t + 2.
    errors = [bool(np.isfinite(i["latent_error"]).all()) for i in infos]
    assert errors == [False, False, True, False, True, False]

    # At step 4 the context holds steps 2 and 4 and the forces of steps 2 and 3.
    assert len(calls) == 3
    context, context_actions = calls[2]
    expected = [
        encoder(
            torch.from_numpy(
                StubRenderer().render(qpos[t], None, resolution=(32, 32))
            ).permute(0, 3, 1, 2)[:, None]
        ).tokens
        for t in (2, 4)
    ]
    torch.testing.assert_close(context, torch.stack(expected, 1))
    torch.testing.assert_close(
        context_actions[:, 0],
        torch.as_tensor(np.concatenate((actions[2], actions[3]), axis=-1)),
    )
    assert calls[0][1].shape == (2, 0, 2)  # no earlier steps at the first plan


def test_latent_mpc_rejects_a_cadence_off_the_model_stride():
    config = CEMConfig(horizon=4, candidates=4, iterations=1, elites=2, action_repeat=2)
    with pytest.raises(ValueError, match="multiple of the model stride"):
        control.LatentMPC(
            CEMPlanner(config, batch_size=1),
            encoder=StubEncoder(),
            renderer=StubRenderer(),
            resolution=(32, 32),
            predictor=tiny_model(),
            probe=StateProbe(2 * CHANNELS, history=2, stride=2),
            objective="goal",
            replan_every=1,
        )


def test_summarize_scores_one_model_step_at_replanning_steps():
    steps = 6
    observations = np.zeros((1, steps + 1, 5))
    observations[0, :, 0] = np.arange(steps + 1)
    plan_observations = np.zeros((1, steps, 2, 5))
    # Plans at steps 0, 2, 4 predict the cart position two steps later exactly.
    plan_observations[0, :, 0, 0] = np.arange(steps) + 2
    plan_observations[0, 1::2, 0, 0] = 99  # held between plans, never scored
    result = {
        "rewards": np.ones((1, steps)),
        "observations": observations,
        "plan_observations": plan_observations,
        "replanned": np.array([[True, False] * 3]),
        "prediction_stride": np.full((1, steps), 2),
        "plan_seconds": np.array([[0.4, 0, 0.4, 0, 0.4, 0]]),
        "latent_error": np.array([[np.nan, np.nan, 0.5, np.nan, 0.7, np.nan]]),
    }
    summary = control.summarize(result)
    assert summary["one_step_pose_rmse"] == 0.0
    assert summary["plan_seconds"] == pytest.approx(0.2)
    assert summary["one_step_latent_rmse"] == pytest.approx(0.6)
