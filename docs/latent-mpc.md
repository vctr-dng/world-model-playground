# Latent MPC in closed loop

Control the real environment by planning with CEM through a world model, compared with PPO, random actions, and planners on the true and linear dynamics. Also measures what a model learned about actions, which training metrics never check.

Control the real environment with the latent world model. Every step renders
the frames, encodes them with the frozen encoder, plans with CEM through the
predictor, executes the first action, and replans:

```bash
uv run python scripts/cartpole/control/latent_mpc.py \
  --embeddings-dir data/embeddings/<collection_id>/vjepa2-vitl_ctx2
uv run marimo edit scripts/cartpole/notebooks/latent_mpc_viewer.py
```

The same episodes (same reset keys) run under every controller: `ppo`,
`random`, and CEM through the true MJX dynamics (`mpc-true`, the planner's
ceiling), the linear model (`mpc-linear`), and the latent model, scoring the
CartPole reward of probe-decoded latents (`mpc-latent`) or the distance to the
latent of an upright, centered, still pole (`mpc-latent-goal`). All planners
share the CEM settings (`--horizon 30`, 256 candidates, 4 iterations, knots
held for 2 steps) and optimize the environment's own dense reward
(`wm.planning.cartpole_reward`). Starts are near upright (`balance`) or at any
pole angle (`full-circle`). Trajectories, chosen plans, a summary, and figures
go to `artifacts/cartpole_world_model/mpc/<collection_id>/<embeddings>/`; the
viewer replays the real pole against each step's plan.

Planning asks a model a question its training metrics never checked: what
happens under actions other than the logged ones. `action_response.py`
applies constant +1 and -1 forces from test states in MuJoCo and in each model
and reports the gain, the share of the true response a model predicts:

```bash
uv run python scripts/cartpole/evaluate/action_response.py \
  --embeddings-dir data/embeddings/<collection_id>/vjepa2-vitl_ctx2 --upright-only
```

Results over 10 episodes of 200 steps (2 s), V-JEPA 2 ViT-L features, return
± standard error (maximum 200):

| Controller | balance (100 ep. data) | balance (held-action data) | full-circle (100 ep.) | full-circle (held) |
| --- | --- | --- | --- | --- |
| PPO | 199.7 ± 0.1 | 199.7 ± 0.1 | 44.2 ± 6.2 | 44.2 ± 6.2 |
| CEM, true dynamics | 194.1 ± 1.2 | 194.1 ± 1.2 | 61.1 ± 8.9 | 66.7 ± 8.1 |
| CEM, latent model (reward) | 118.1 ± 1.4 | 141.7 ± 7.3 | 52.0 ± 7.9 | 40.3 ± 6.4 |
| CEM, latent model (goal latent) | 96.5 ± 10.6 | 127.3 ± 7.4 | 26.8 ± 9.7 | 21.1 ± 6.8 |
| CEM, linear model | 115.3 ± 3.2 | 121.4 ± 2.9 | 52.9 ± 4.5 | 42.2 ± 5.9 |
| Random | 106.0 ± 2.0 | 106.0 ± 2.0 | 23.5 ± 8.1 | 23.5 ± 8.1 |

With true dynamics the planner balances (194), so the gap is the models'. The
latent model reads the state well (probe pose RMSE 0.025 to 0.06 in closed
loop) but barely knows what actions do. Near upright, after 30 steps of
force, it predicts this share of the true response:

| Training data | latent: cart | latent: pole angle | linear: cart | linear: pole angle |
| --- | --- | --- | --- | --- |
| 100 episodes, sources mixed per step | 0.38 | -0.03 | 0.97 | 0.15 |
| 200 episodes, `--source-hold 10` | 0.66 | 0.16 | 0.90 | 0.09 |

Mixed per step, random forces never last long enough to show their effect, and
logged expert actions are predictable from the state, so a predictor can track
logged trajectories ([latent world model](latent-world-model.md)) while ignoring its actions. Holding each action
source for 10 steps (`collect_obs.py --source-hold 10`) raises the latent
model's response, and balance return from 118 to 142, but the pole-angle
response near upright stays far below the physics: random forces quickly
topple the pole, so the data still rarely shows sustained forces while
balancing. The linear model, fitted across every pole orientation, averages
away the action-to-angle coupling whose sign flips between upright and
hanging. On full-circle starts, no controller reliably swings up within 2 s
(PPO included): true-dynamics MPC does best, upright 33 to 41% of the second
half, and the latent model plans at the level of the linear model.

Planning through the latent model costs about 1.1 s per step for 10 episodes
(256 candidates x 30 steps x 4 iterations), against 0.04 s with MJX.

## Perturbations around balance

To show the model sustained forces where balancing happens, the data recipe in [Data collection](data-collection.md#perturbations-around-balance) starts episodes near upright and, while the pole is upright, perturbs it with a uniform random force held for 10 to 30 steps. Embedding, probe, and predictor then train as before.

Near upright, after 10 to 30 steps of force, the latent model now predicts most of the true response:

| Training data | latent: cart | latent: pole angle | linear: cart | linear: pole angle |
| --- | --- | --- | --- | --- |
| 100 episodes, sources mixed per step | 0.38 | -0.03 | 0.97 | 0.15 |
| 200 episodes, `--source-hold 10` | 0.66 | 0.16 | 0.90 | 0.09 |
| 200 episodes, perturbations around balance | 0.92 | 0.81 | 0.86 | -0.10 |

and closed-loop balance return (10 episodes, same settings as above) reaches:

| Controller | held-action data | perturbation data |
| --- | --- | --- |
| CEM, latent model (goal latent) | 127.3 ± 7.4 | 183.8 ± 2.1 |
| CEM, latent model (reward) | 141.7 ± 7.3 | 175.5 ± 5.3 |
| CEM, linear model | 121.4 ± 2.9 | 174.9 ± 5.4 |
| CEM, true dynamics | 194.1 ± 1.2 | 194.1 ± 1.2 |

With the goal latent, the planner keeps the pole upright 97% of the second
half of each episode. Planning on the probe-decoded reward is held back by the
decoded angular velocity: those plans push twice as hard as true-dynamics MPC
and let the pole oscillate until it falls. Full-circle starts get worse (latent
33 to 39), since this collection rarely shows a hanging pole; mix start
distributions when swing-up matters.

Training the predictor with a 24-step instead of an 8-step open-loop loss
(`train_latent_wm.py --rollout-steps 24`) improves rollouts beyond 50 steps but
degrades the first 10 to 25, which the 30-step planner relies on, and lowers
balance return to 152.6 (reward) and 141.5 (goal latent).

---

Previous: [Latent world model](latent-world-model.md) · Next: [V-JEPA 2-AC](vjepa2-ac.md) · [README](../README.md)
