# Graph Report - world-model-playground  (2026-11-23)

## Corpus Check
- 65 files · ~38,791 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 816 nodes · 1456 edges · 40 communities (32 shown, 8 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 273 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7acf9156`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- OpenCode Configuration
- Graphify Plugin
- Main Entry Point
- Project Package
- parse_args
- run_live_rollout
- dependencies
- collect_obs.py
- parse_args
- scalar_metrics
- StateProbe
- __init__.py
- fit_linear_dynamics
- test_encoders.py
- up.sh
- Tensor
- main
- ac_predictor.py
- main
- test_ac_world_model.py
- CEMConfig
- test_predictors.py
- main
- LatentMPC
- CEMPlanner
- load_ac_encoder_backbone
- test_episodes.py
- FrameRenderer
- Levels of Working with World Models
- test_wm_eval.py
- test_probes.py
- Evaluation
- load_cartpole
- test_collect_schedule.py
- test_linear_model.py
- test_cartpole_render.py
- __init__.py
- __init__.py
- __init__.py

## God Nodes (most connected - your core abstractions)
1. `StateProbe` - 32 edges
2. `main()` - 24 edges
3. `CEMPlanner` - 24 edges
4. `main()` - 23 edges
5. `main()` - 22 edges
6. `FrameRenderer` - 21 edges
7. `ACWorldModel` - 21 edges
8. `LatentPredictor` - 20 edges
9. `CEMConfig` - 19 edges
10. `main()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_linear_dynamics_matches_the_numpy_model()` --calls--> `fit_linear_dynamics()`  [INFERRED]
  tests/test_planning.py → src/world_model_playground/wm/linear.py
- `parse_args()` --indirect_call--> `positive_int()`  [INFERRED]
  scripts/cartpole/random_agent.py → src/world_model_playground/cartpole.py
- `main()` --calls--> `load_cartpole()`  [INFERRED]
  scripts/cartpole/random_agent.py → src/world_model_playground/cartpole.py
- `parse_args()` --indirect_call--> `positive_int()`  [INFERRED]
  scripts/cartpole/train_ppo.py → src/world_model_playground/cartpole.py
- `render_policy()` --calls--> `run_live_rollout()`  [INFERRED]
  scripts/cartpole/train_ppo.py → src/world_model_playground/cartpole.py

## Import Cycles
- None detected.

## Communities (40 total, 8 thin omitted)

### Community 0 - "OpenCode Configuration"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 2 - "Main Entry Point"
Cohesion: 0.09
Nodes (30): _(), evaluate_policy(), main(), parse_args(), Namespace, Path, Train a Brax PPO agent to solve MJX CartpoleBalance., Records one deterministic policy rollout to an MP4 file. (+22 more)

### Community 3 - "Project Package"
Cohesion: 0.13
Nodes (14): Brax PPO, Cartpole Balance, Docker, DreamerV3 export, Frozen encoders and state probes, Latent MPC in closed loop, Latent world model, Linear Model (+6 more)

### Community 4 - "parse_args"
Cohesion: 0.20
Nodes (8): Any, Carry, MjxEnv, main(), Run a random agent in parallel MJX CartpoleBalance environments., Shared CartpoleBalance environment and visualization helpers., Steps a rollout while showing one state in MuJoCo's live viewer., run_live_rollout()

### Community 6 - "run_live_rollout"
Cohesion: 0.05
Nodes (58): ActionAlignment, main(), main(), parse_args(), Namespace, Encode an episodic Cartpole collection with a frozen pretrained encoder.  Writes, Smallest square collected resolution of at least ``size``, if any., source_resolution() (+50 more)

### Community 7 - "dependencies"
Cohesion: 0.50
Nodes (3): @opencode-ai/plugin, dependencies, @opencode-ai/plugin

### Community 8 - "collect_obs.py"
Cohesion: 0.14
Nodes (19): action_counts(), episode_length(), fraction(), held_schedule(), make_batched_rollout(), parse_args(), Namespace, ndarray (+11 more)

### Community 9 - "parse_args"
Cohesion: 0.05
Nodes (45): memmap, ac_losses(), ac_rollouts(), ACConfig, ACWorldModel, _autocast(), build_ac_model(), evaluate_ac_losses() (+37 more)

### Community 10 - "scalar_metrics"
Cohesion: 0.07
Nodes (43): evaluate(), flat_metrics(), log_to_mlflow(), main(), non_negative_float(), parse_args(), Namespace, Path (+35 more)

### Community 11 - "StateProbe"
Cohesion: 0.07
Nodes (39): flat_metrics(), log_to_mlflow(), main(), parse_args(), probe_filename(), Namespace, Path, Fit state probes on cached encoder features and compare encoders.  For each embe (+31 more)

### Community 13 - "fit_linear_dynamics"
Cohesion: 0.07
Nodes (38): main(), parse_args(), positive_float(), Namespace, Train a linear residual world model from CartPole observation transitions., Parse a strictly positive floating-point value., load_transitions(), load_transitions() (+30 more)

### Community 14 - "test_encoders.py"
Cohesion: 0.08
Nodes (25): Dinov2Encoder, EncoderSpec, FrozenEncoder, load_encoder(), preprocess_frames(), device, dtype, Module (+17 more)

### Community 16 - "Tensor"
Cohesion: 0.09
Nodes (27): Evaluator, Protocol, cartpole_reward(), decode_latents(), group_step_actions(), latent_distance(), latent_evaluator(), LatentWorldModel (+19 more)

### Community 17 - "main"
Cohesion: 0.09
Nodes (32): PredictNext, parse_args(), Namespace, evaluate(), evaluation_horizons(), flat_metrics(), log_to_mlflow(), main() (+24 more)

### Community 18 - "ac_predictor.py"
Cohesion: 0.12
Nodes (10): ACBlock, ACRoPEAttention, build_action_block_causal_attention_mask(), DropPath, MLP, Action Conditioned Vision Transformer Predictor, :param x: context tokens, Drop paths (Stochastic Depth) per sample  (when applied in main path of residual (+2 more)

### Community 19 - "main"
Cohesion: 0.11
Nodes (21): main(), parse_args(), plot(), pose(), Namespace, ndarray, Path, Measure how world models respond to actions they were not shown.  Open-loop RMSE (+13 more)

### Community 20 - "test_ac_world_model.py"
Cohesion: 0.17
Nodes (20): Tensor, random_latents(), Tests for the V-JEPA 2-AC adapter, on tiny random-weight predictors., test_ac_rollouts_hold_each_model_step_over_its_control_steps(), test_action_encoder_reads_grouped_forces_from_pretrained_columns(), test_actions_change_the_prediction(), test_fit_runs_and_keeps_the_best_validation_weights(), test_forward_predicts_layer_normalized_tokens_for_every_step() (+12 more)

### Community 21 - "CEMConfig"
Cohesion: 0.15
Nodes (13): NamedTuple, EncoderOutput, CEMConfig, Cross-entropy method over piecewise-constant action sequences., test_cem_warm_start_advances_by_the_replanning_shift(), Tests for the replanning cadence and strided context of latent_mpc.py controller, Frames whose every pixel encodes the cart position., Tokens that are a fixed function of the frame, as a frozen encoder's are. (+5 more)

### Community 22 - "test_predictors.py"
Cohesion: 0.13
Nodes (17): slice, EpisodeClipDataset, Dataset, Tensor, PyTorch datasets over episodic CartPole collections., Fixed-length clips of frames, observations, and actions from episode shards., test_held_schedule_keeps_sources_and_random_actions_over_segments(), make_episodes() (+9 more)

### Community 23 - "main"
Cohesion: 0.18
Nodes (18): encode_clips(), encode_goal(), log_to_mlflow(), main(), parse_args(), plot_results(), pose_rmse(), Namespace (+10 more)

### Community 24 - "LatentMPC"
Cohesion: 0.15
Nodes (7): LatentMPC, PlanningController, Shared CEM bookkeeping: replanning cadence, timing, and the chosen plan.      A, Per-step bookkeeping before planning; returns per-step info., Called with every new plan., CEM through a latent world model from the encoded rendered frames.      Every co, The last ``count`` entries ``step_stride`` apart, oldest first.

### Community 25 - "CEMPlanner"
Cohesion: 0.15
Nodes (11): Tensor, CEM through batched MJX rollouts of the real environment., CEM through an observation-space model rolled out from true observations., StateModelMPC, to_torch(), TrueDynamicsMPC, CEMPlanner, device (+3 more)

### Community 26 - "load_ac_encoder_backbone"
Cohesion: 0.17
Nodes (15): clean_keys(), ensure_checkpoint(), hf_encoder_state_dict(), load_ac_encoder_backbone(), load_checkpoint(), dtype, Module, Path (+7 more)

### Community 27 - "test_episodes.py"
Cohesion: 0.18
Nodes (9): collection(), make_episode(), Tests for episodic collections, their DreamerV3 view, and the clip dataset., Builds a valid episode whose frames encode their step index., Writes a three-episode collection with a format-2 manifest., test_to_dreamer_episode_aligns_steps(), test_to_dreamer_episode_marks_truncation_as_non_terminal(), test_validate_episode_rejects_malformed_fields() (+1 more)

### Community 28 - "FrameRenderer"
Cohesion: 0.21
Nodes (8): MjModel, Self, FrameRenderer, ndarray, Vertical field of view used for frames of this resolution., Renders rows of ``qpos``/``qvel`` once; prefer ``FrameRenderer`` in loops., Renders rows of ``qpos``/``qvel`` at fixed resolutions of one camera.      Every, render_frames()

### Community 29 - "Levels of Working with World Models"
Cohesion: 0.15
Nodes (12): Level 0: Use a pretrained world model as a black box, Level 1: Frozen pretrained encoder, train only the dynamics (DINO-WM style), Level 2: Train a JEPA end-to-end from pixels (LeWM style), Level 3: Compare and analyze, Level 4: Add a research contribution, Level 5: Other families of world models, Levels of Working with World Models, Open items (+4 more)

### Community 30 - "test_wm_eval.py"
Cohesion: 0.28
Nodes (12): as_episodes(), exact_dynamics(), make_episodes(), ndarray, Tests for the model-agnostic world-model evaluation metrics., Regroups shuffled transition rows into per-episode step arrays., Builds episodes whose next observation is the current one plus the action., test_episode_rollout_rmse_matches_rollout_rmse_from_the_first_step() (+4 more)

### Community 31 - "test_probes.py"
Cohesion: 0.20
Nodes (5): ndarray, Tests for state probes on synthetic features with a known observation decoding., Observations with a valid (cos, sin) pair, linearly mixed into features., splits(), synthetic_split()

### Community 32 - "Evaluation"
Cohesion: 0.20
Nodes (4): PPOController, RandomController, Evaluation, Scores of candidate plans and per-candidate tensors to keep for the chosen one.

### Community 33 - "load_cartpole"
Cohesion: 0.22
Nodes (8): _(), _(), camera_model(), load_cartpole(), matched_fovy(), Loads CartpoleBalance, optionally resetting across the full pole circle., Vertical field of view, in degrees, keeping the horizontal view of a 320x240 fra, test_cartpole_reward_matches_the_environment()

### Community 34 - "test_collect_schedule.py"
Cohesion: 0.43
Nodes (5): Tests for the action-source schedule of collect_obs.py., run_rollout(), test_perturbations_hold_one_force_for_the_requested_length(), test_perturbations_only_start_upright_and_are_off_by_default(), zero_policy()

## Knowledge Gaps
- **24 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `@opencode-ai/plugin`, `up.sh script`, `Parallel MJX Random Agent` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `main` to `load_cartpole`, `Main Entry Point`, `run_live_rollout`, `parse_args`, `StateProbe`, `fit_linear_dynamics`, `test_encoders.py`, `main`, `CEMConfig`, `FrameRenderer`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `run_live_rollout`, `parse_args`, `scalar_metrics`, `StateProbe`, `fit_linear_dynamics`, `main`, `load_ac_encoder_backbone`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `StateProbe` connect `StateProbe` to `Evaluation`, `scalar_metrics`, `Tensor`, `test_ac_world_model.py`, `CEMConfig`, `test_predictors.py`, `CEMPlanner`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `StateProbe` (e.g. with `CEMConfig` and `CEMPlanner`) actually correct?**
  _`StateProbe` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `main()` (e.g. with `load_cartpole()` and `load_policy()`) actually correct?**
  _`main()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `CEMPlanner` (e.g. with `LatentMPC` and `PlanningController`) actually correct?**
  _`CEMPlanner` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `main()` (e.g. with `load_checkpoint()` and `ACConfig`) actually correct?**
  _`main()` has 18 INFERRED edges - model-reasoned connections that need verification._