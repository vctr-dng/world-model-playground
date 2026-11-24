# Graph Report - chore-portfolio-cleanup  (2026-11-24)

## Corpus Check
- 83 files · ~57,247 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 937 nodes · 1365 edges · 83 communities (41 shown, 42 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 77 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `814e8a8c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- opencode.json
- graphify.js
- test_episodes.py
- World models
- cartpole.py
- visualize_obs.py
- episodes.py
- train_ac_wm.py
- collect_obs.py
- ACWorldModel
- LatentPredictor
- probes.py
- __init__.py
- LinearResidualDynamics
- test_encoders.py
- up.sh
- .expand
- test_predictors.py
- ac_predictor.py
- action_response.py
- test_ac_world_model.py
- StubEncoder
- EpisodeClipDataset
- latent_mpc.py
- data.py
- CEMPlanner
- load_ac_encoder_backbone
- .__init__
- FrameRenderer
- Levels of Working with World Models
- test_wm_eval.py
- test_probes.py
- StateProbe
- load_cartpole
- data.py
- test_linear_model.py
- test_cartpole_render.py
- __init__.py
- __init__.py
- __init__.py
- train_ppo.py
- eval_lewm.py
- planning.py
- torch_device
- control.py
- eval_lewm.py
- LatentMPC
- embed.py
- PlanningController
- Getting started
- export_dreamer.py
- test_cli.py
- Tensor
- run_episodes
- linear_dynamics
- PPOController
- RandomController
- Namespace
- Path
- README.md
- Namespace
- Namespace
- Path
- Namespace
- ndarray
- Path
- Namespace
- Path
- Namespace
- Namespace
- Namespace
- Path
- Namespace
- Path
- Namespace
- Namespace
- Path
- Namespace
- Path
- Namespace
- Path

## God Nodes (most connected - your core abstractions)
1. `StateProbe` - 28 edges
2. `ACWorldModel` - 21 edges
3. `LatentPredictor` - 20 edges
4. `Levels of Working with World Models` - 13 edges
5. `tiny_model()` - 13 edges
6. `LinearResidualDynamics` - 12 edges
7. `PlanningController` - 11 edges
8. `LatentMPC` - 11 edges
9. `FrozenEncoder` - 11 edges
10. `LatentWorldModel` - 11 edges

## Surprising Connections (you probably didn't know these)
- `test_linear_dynamics_matches_the_numpy_model()` --calls--> `fit_linear_dynamics()`  [INFERRED]
  tests/test_planning.py → src/world_model_playground/wm/linear.py
- `test_strided_step_windows()` --calls--> `step_windows()`  [INFERRED]
  tests/test_ac_world_model.py → src/world_model_playground/wm/episodes.py
- `test_latent_evaluator_scores_token_models_per_model_step()` --calls--> `latent_evaluator()`  [INFERRED]
  tests/test_ac_world_model.py → src/world_model_playground/wm/planning.py
- `test_decode_latents_stacks_history_like_the_probe_features()` --calls--> `stack_history()`  [INFERRED]
  tests/test_planning.py → src/world_model_playground/wm/probes.py
- `test_latent_rollout_decodes_predictions_with_the_probe()` --calls--> `StateProbe`  [INFERRED]
  tests/test_predictors.py → src/world_model_playground/wm/probes.py

## Import Cycles
- None detected.

## Communities (83 total, 42 thin omitted)

### Community 0 - "opencode.json"
Cohesion: 0.22
Nodes (11): fit_linear_probe(), fit_mlp_probe(), _normalized_mse(), device, ndarray, Tensor, Ridge regression on normalized features, picking the ridge on validation.      F, Trains an MLP probe and keeps the epoch with the lowest validation loss. (+3 more)

### Community 1 - "graphify.js"
Cohesion: 0.43
Nodes (7): LeWM, main(), parameter_counts(), parse_args(), Namespace, Train a LeWM JEPA end to end from CartPole frames.  The encoder and the action-c, run_name()

### Community 2 - "test_episodes.py"
Cohesion: 0.09
Nodes (29): evaluate_policy(), main(), parse_args(), Namespace, Path, Train a Brax PPO agent to solve MJX CartpoleBalance., Records one deterministic policy rollout to an MP4 file., Displays a deterministic policy in the live MuJoCo viewer. (+21 more)

### Community 3 - "World models"
Cohesion: 0.29
Nodes (6): Documentation, Headline results, Pipeline, Quickstart, Repository layout, World Model Playground

### Community 4 - "cartpole.py"
Cohesion: 0.07
Nodes (5): load_transitions(), main(), parse_args(), Namespace, Train a linear residual world model from CartPole observation transitions.

### Community 5 - "visualize_obs.py"
Cohesion: 0.10
Nodes (16): Collecting episodes, Data collection, DreamerV3 export, Perturbations around balance, Frozen encoders and state probes, Brax PPO, Expert policy, Parallel MJX Random Agent (+8 more)

### Community 6 - "episodes.py"
Cohesion: 0.18
Nodes (9): collection(), make_episode(), Tests for episodic collections, their DreamerV3 view, and the clip dataset., Builds a valid episode whose frames encode their step index., Writes a three-episode collection with a format-2 manifest., test_to_dreamer_episode_aligns_steps(), test_to_dreamer_episode_marks_truncation_as_non_terminal(), test_validate_episode_rejects_malformed_fields() (+1 more)

### Community 7 - "train_ac_wm.py"
Cohesion: 0.33
Nodes (5): ActionAlignment, ndarray, Converting episodic collections into DreamerV3-style step sequences.  DreamerV3, Returns the ``T+1`` step arrays of one episode in DreamerV3's layout.      ``sta, to_dreamer_episode()

### Community 9 - "ACWorldModel"
Cohesion: 0.05
Nodes (44): memmap, ac_losses(), ac_rollouts(), ACConfig, ACWorldModel, _autocast(), build_ac_model(), evaluate_ac_losses() (+36 more)

### Community 10 - "LatentPredictor"
Cohesion: 0.07
Nodes (36): decode_steps(), evaluate_losses(), fit_predictor(), latent_rollout(), LatentPredictor, LatentWindows, load_predictor(), normalized_latent_rollout() (+28 more)

### Community 11 - "probes.py"
Cohesion: 0.14
Nodes (14): check_probe(), load_probe(), probe_arrays(), Path, State probes that read the CartPole observation back out of frozen features.  A, Per-observation R² and RMSE, plus the absolute pole-angle error in radians., Restores a probe written by :func:`save_probe` and its metadata., Rejects a probe of another encoder or one that saw the test episodes. (+6 more)

### Community 13 - "LinearResidualDynamics"
Cohesion: 0.09
Nodes (27): load_transitions(), partition_mask(), project_angle_pair(), ndarray, Path, Loading, validating, and splitting collected CartPole transitions., Read every array of a collection's ``transitions.npz`` into memory., Validate the observation-only transition fields required by the trainers. (+19 more)

### Community 14 - "test_encoders.py"
Cohesion: 0.08
Nodes (27): NamedTuple, Dinov2Encoder, EncoderOutput, EncoderSpec, FrozenEncoder, load_encoder(), preprocess_frames(), device (+19 more)

### Community 17 - "test_predictors.py"
Cohesion: 0.08
Nodes (34): PredictNext, slice, EpisodeClipDataset, Dataset, Tensor, PyTorch datasets over episodic CartPole collections., Fixed-length clips of frames, observations, and actions from episode shards., episode_rollout_rmse() (+26 more)

### Community 18 - "ac_predictor.py"
Cohesion: 0.11
Nodes (11): ACBlock, ACRoPEAttention, build_action_block_causal_attention_mask(), DropPath, MLP, Action Conditioned Vision Transformer Predictor, :param x: context tokens, Drop paths (Stochastic Depth) per sample  (when applied in main path of residual (+3 more)

### Community 19 - "action_response.py"
Cohesion: 0.05
Nodes (46): main(), parse_args(), Namespace, Path, Score a trained LeWM like the Level 1 latent models: decoded open-loop rollouts., Stores the probe in the artifact's params, where the planners look for it., record_probe(), by_episode_id() (+38 more)

### Community 20 - "test_ac_world_model.py"
Cohesion: 0.05
Nodes (62): ndarray, Path, default_embeddings_dir(), embed_frames(), load_embedded_episodes(), load_embeddings_manifest(), ndarray, Path (+54 more)

### Community 21 - "StubEncoder"
Cohesion: 0.24
Nodes (8): Tests for the replanning cadence and strided context of the MPC controllers., Frames whose every pixel encodes the cart position., Tokens that are a fixed function of the frame, as a frozen encoder's are., StubEncoder, StubRenderer, test_latent_mpc_plans_every_model_step_from_strided_context(), test_latent_mpc_rejects_a_cadence_off_the_model_stride(), tiny_model()

### Community 22 - "EpisodeClipDataset"
Cohesion: 0.23
Nodes (12): main(), parse_args(), plot(), pose(), Namespace, Path, Measure how world models respond to actions they were not shown.  Open-loop RMSE, Cart position and unwrapped pole angle ``(..., H, 2)``. (+4 more)

### Community 23 - "latent_mpc.py"
Cohesion: 0.31
Nodes (8): Path, anchors(), check(), main(), Check the documentation: relative links, heading anchors, and script paths.  Run, The anchor GitHub generates for a heading., Anchors of the headings of a page, ignoring comments in code blocks., slug()

### Community 25 - "CEMPlanner"
Cohesion: 0.14
Nodes (11): Evaluator, CEMConfig, CEMPlanner, device, Cross-entropy method over piecewise-constant action sequences., Receding-horizon CEM for ``batch_size`` episodes, warm-started across steps., ``shift`` is the number of control steps between successive plans,         by wh, Forgets the warm start, e.g. at the start of new episodes. (+3 more)

### Community 26 - "load_ac_encoder_backbone"
Cohesion: 0.17
Nodes (15): clean_keys(), ensure_checkpoint(), hf_encoder_state_dict(), load_ac_encoder_backbone(), load_checkpoint(), dtype, Module, Path (+7 more)

### Community 27 - ".__init__"
Cohesion: 0.07
Nodes (25): Candidate directions, Models and methods, References, Simulation and training, Closed-loop control: LeWM collection, Closed-loop control: patch-token collection, Closed-loop control: pooled features, Open-loop accuracy of the decoded pose (+17 more)

### Community 28 - "FrameRenderer"
Cohesion: 0.06
Nodes (36): Any, Carry, MjModel, MjxEnv, log_to_mlflow(), main(), parse_args(), plot_results() (+28 more)

### Community 29 - "Levels of Working with World Models"
Cohesion: 0.33
Nodes (8): evaluate(), log_to_mlflow(), main(), parse_args(), Namespace, Path, Train an action-conditioned latent world model on cached encoder features.  A ca, Decoded-observation and latent rollout RMSE of the model and baselines.

### Community 30 - "test_wm_eval.py"
Cohesion: 0.28
Nodes (12): as_episodes(), exact_dynamics(), make_episodes(), ndarray, Tests for the model-agnostic world-model evaluation metrics., Regroups shuffled transition rows into per-episode step arrays., Builds episodes whose next observation is the current one plus the action., test_episode_rollout_rmse_matches_rollout_rmse_from_the_first_step() (+4 more)

### Community 31 - "test_probes.py"
Cohesion: 0.20
Nodes (5): ndarray, Tests for state probes on synthetic features with a known observation decoding., Observations with a valid (cos, sin) pair, linearly mixed into features., splits(), synthetic_split()

### Community 32 - "StateProbe"
Cohesion: 0.27
Nodes (6): CEMPlanner, FrameRenderer, CEM through an observation-space model rolled out from true observations., StateModelMPC, to_torch(), Tensor

### Community 42 - "eval_lewm.py"
Cohesion: 0.25
Nodes (7): flat_probe_metrics(), flat_rollout_metrics(), MLflow run setup and metric flattening shared by the training scripts.  MLflow t, Starts an MLflow run tagged with its pipeline ``stage`` and yields ``mlflow``., Flattens an ``evaluate`` record: observation RMSE (overall and per group)     pe, Flattens the metrics of a state probe: angle error, RMSE and per-state R²., tracked_run()

### Community 43 - "planning.py"
Cohesion: 0.12
Nodes (20): Protocol, cartpole_reward(), decode_latents(), group_step_actions(), latent_distance(), latent_evaluator(), LatentWorldModel, project_angle() (+12 more)

### Community 45 - "torch_device"
Cohesion: 0.12
Nodes (17): EpisodeRollout, evaluate_rollouts(), evaluation_horizons(), Open-loop rollout scoring of a latent world model against the same baselines.  L, Rollout horizons where a model stepping ``stride`` control steps is exact., Decoded-observation and latent rollout RMSE of a model and the baselines.      `, Tests for metric flattening and the shared rollout evaluation.  The flattened ke, Decodes a feature to the observations it was built from. (+9 more)

### Community 46 - "control.py"
Cohesion: 0.29
Nodes (9): ndarray, encode_clips(), encode_goal(), pose_rmse(), Controllers for closed-loop CartPole control and the episode runner.  A controll, Pooled features and tokens of uint8 clips ``(B, L, H, W, 3)``.      In bfloat16,, Latent of an upright, centered, still pole: the same frame repeated., Episode returns and, for planners, how well they predicted reality. (+1 more)

### Community 47 - "eval_lewm.py"
Cohesion: 0.33
Nodes (8): device, log_to_mlflow(), main(), parse_args(), Namespace, Path, Train an action-conditioned world model on cached frozen-encoder patch tokens., storage_device()

### Community 49 - "LatentMPC"
Cohesion: 0.36
Nodes (3): LatentMPC, CEM through a latent world model from the encoded rendered frames.      Every co, The last ``count`` entries ``step_stride`` apart, oldest first.

### Community 51 - "PlanningController"
Cohesion: 0.18
Nodes (6): PlanningController, CEM through batched MJX rollouts of the real environment., Shared CEM bookkeeping: replanning cadence, timing, and the chosen plan.      A, Per-step bookkeeping before planning; returns per-step info., Called with every new plan., TrueDynamicsMPC

### Community 52 - "Getting started"
Cohesion: 0.29
Nodes (6): Getting started, MLflow, Requirements, Sharing a GPU, Tests and linting, Where the code lives

### Community 57 - "run_episodes"
Cohesion: 0.18
Nodes (8): device, Running PyTorch world models alongside JAX/MJX on a single GPU., Returns the requested device, defaulting to CUDA when it is available., Stops JAX from reserving most of the GPU so PyTorch can allocate memory too., share_gpu_with_jax(), torch_device(), Tests that PyTorch world models can run next to JAX on the same GPU., test_torch_device_honours_explicit_choice()

### Community 59 - "linear_dynamics"
Cohesion: 0.17
Nodes (11): LinearResidualDynamics, Action-conditioned ridge model that predicts the next observation delta., Evaluation, linear_dynamics(), Plan, Scores of candidate plans and per-candidate tensors to keep for the chosen one., The best candidate of the last iteration for every episode., Scores candidates by the CartPole reward of an observation-space rollout.      ` (+3 more)

### Community 60 - "PPOController"
Cohesion: 0.24
Nodes (9): action_counts(), held_schedule(), make_batched_rollout(), Action-source schedules and the batched rollout behind ``collect_obs.py``.  Coll, Apportions the requested transition count across the three action sources., Returns an exact, reproducibly shuffled action-source schedule., Action-source schedule ``(E, T)`` and random actions ``(E, T, 1)``.      Sources, Builds a jitted rollout of parallel episodes following a source schedule.      T (+1 more)

## Knowledge Gaps
- **46 isolated node(s):** `Requirements`, `MLflow`, `Sharing a GPU`, `Tests and linting`, `Where the code lives` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StateProbe` connect `opencode.json` to `LatentPredictor`, `probes.py`, `planning.py`, `LinearResidualDynamics`, `test_predictors.py`, `test_ac_world_model.py`, `CEMPlanner`, `linear_dynamics`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Why does `step_windows()` connect `test_ac_world_model.py` to `LatentPredictor`, `probes.py`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `make_probe()` connect `LatentPredictor` to `opencode.json`, `probes.py`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `StateProbe` (e.g. with `CEMConfig` and `CEMPlanner`) actually correct?**
  _`StateProbe` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LatentPredictor` (e.g. with `StateProbe` and `test_latent_evaluator_scores_reward_and_goal_objectives()`) actually correct?**
  _`LatentPredictor` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Requirements`, `MLflow`, `Sharing a GPU` to the rest of the system?**
  _269 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_episodes.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09247311827956989 - nodes in this community are weakly interconnected._