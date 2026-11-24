# CartPole scripts

Command-line entry points, grouped by pipeline stage. Each script is a thin CLI
over the `world_model_playground` package (`src/`); run `--help` on any of them for
its flags. Everything is run from the repository root with `uv run python <script>`.

```
expert  ->  data  ->  train  ->  evaluate  ->  control
 PPO        collect    probes     rollout       closed-loop
 policy     + embed    + world    error,        MPC with the
                       models     action        learned model
                                  response
```

| Stage | Script | What it does | Reads | Writes |
| --- | --- | --- | --- | --- |
| `expert/` | `random_agent.py` | Random actions in parallel MJX environments, with an optional live viewer. | | |
| | `train_ppo.py` | Brax PPO on `CartpoleBalance`; the expert whose episodes the rest uses. | | MLflow experiment `cartpole-ppo` |
| `data/` | `collect_obs.py` | Rolls out the PPO policy, stochastic and random actions, and perturbations, rendering frames. | best PPO run in MLflow | `data/cartpole_trajectories/<collection>/` |
| | `embed.py` | Encodes a collection with a frozen encoder (DINOv2, V-JEPA 2, V-JEPA 2-AC, or a trained LeWM). | a collection | `data/embeddings/<collection>/<encoder>_ctx<N>/` |
| | `consolidate_tokens.py` | Copies cached patch tokens into one flat memmap for training. | embeddings with tokens | the same embeddings directory |
| | `export_dreamer.py` | Writes a collection as DreamerV3 episode files. | a collection | DreamerV3 episodes |
| `train/` | `train_probe.py` | Fits state probes on cached features: how much state does an encoder keep? | embeddings | `artifacts/cartpole_world_model/probes/` |
| | `train_patch_probe.py` | The same, on the pooled grid of patch tokens. | embeddings with tokens | `.../probes/` |
| | `train_latent_wm.py` | Action-conditioned predictor on pooled encoder features. | embeddings, a probe | `.../latent_wm/` |
| | `train_ac_wm.py` | Post-trains V-JEPA 2-AC's predictor on CartPole tokens. | embeddings with tokens, a probe | `.../ac_wm/` |
| | `train_lewm.py` | Trains a LeWM JEPA end to end from pixels (prediction loss plus SIGReg). | a collection | `.../lewm/<collection>/` |
| | `train_patch_wm.py` | DINO-WM-style predictor of the whole patch-token grid. | consolidated tokens, a probe | `.../patch_wm/` |
| | `train_linear.py` | The linear residual baseline. | a collection | `.../linear_model.npz` |
| `evaluate/` | `eval_lewm.py` | Decoded open-loop rollout error of a trained LeWM, against the baselines. | a LeWM model, embeddings, a probe | `<model>.eval.json`, MLflow |
| | `eval_patch_wm.py` | The same for a patch-token model. | a patch-token model, embeddings | metrics JSON (`--output`) |
| | `action_response.py` | Counterfactual test: does the model respond to forces it was not shown? | any latent model | JSON report and figure (`--output`) |
| `control/` | `latent_mpc.py` | Closed-loop CEM planning with any world model against PPO, random actions, and planners on true and linear dynamics. | a model, embeddings | `artifacts/cartpole_world_model/mpc/...`, MLflow |
| `notebooks/` | `visualize_obs.py` | Browse a collection: frames, camera overlay, action sources. | `data/` | |
| | `latent_mpc_viewer.py` | Replay and compare MPC runs. | `mpc/` results | |
| | `mpc_divergence.py` | Linear-model rollouts against the real environment. | the linear model | |

`notebooks/` are [marimo](https://marimo.io) apps: `uv run marimo edit scripts/cartpole/notebooks/<name>.py`.

The shared logic lives in the package: `wm.control` (controllers and the episode
runner), `wm.collection` (action-source schedules), `wm.tracking` and
`wm.rollout_eval` (MLflow runs and rollout scoring), and `cli` (argument types).
