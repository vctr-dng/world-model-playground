# Graph Report - world-model-playground  (2026-09-12)

## Corpus Check
- 20 files · ~2,524 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 57 nodes · 71 edges · 15 communities (10 shown, 5 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9225a23a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- OpenCode Configuration
- Graphify Plugin
- Main Entry Point
- Project Package
- parse_args
- load_policy
- run_live_rollout
- dependencies
- load_cartpole
- parse_args
- scalar_metrics
- cartpole.py
- __init__.py
- make_fixed_scale_ppo_networks
- load_policy

## God Nodes (most connected - your core abstractions)
1. `main()` - 8 edges
2. `run_live_rollout()` - 7 edges
3. `parse_args()` - 5 edges
4. `make_fixed_scale_ppo_networks()` - 5 edges
5. `parse_args()` - 5 edges
6. `load_policy()` - 5 edges
7. `load_cartpole()` - 5 edges
8. `main()` - 4 edges
9. `resolve_checkpoint()` - 4 edges
10. `render_policy()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `parse_args()` --indirect_call--> `non_negative_int()`  [INFERRED]
  scripts/cartpole/random_agent.py → src/world_model_playground/cartpole.py
- `parse_args()` --indirect_call--> `positive_int()`  [INFERRED]
  scripts/cartpole/random_agent.py → src/world_model_playground/cartpole.py
- `main()` --calls--> `load_cartpole()`  [INFERRED]
  scripts/cartpole/random_agent.py → src/world_model_playground/cartpole.py
- `main()` --calls--> `run_live_rollout()`  [INFERRED]
  scripts/cartpole/random_agent.py → src/world_model_playground/cartpole.py
- `render_policy()` --calls--> `run_live_rollout()`  [INFERRED]
  scripts/cartpole/train_ppo.py → src/world_model_playground/cartpole.py

## Import Cycles
- None detected.

## Communities (15 total, 5 thin omitted)

### Community 0 - "OpenCode Configuration"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 2 - "Main Entry Point"
Cohesion: 0.40
Nodes (5): evaluate_policy(), main(), Evaluates a deterministic policy over a batch of environments., Displays a deterministic policy in the live MuJoCo viewer., render_policy()

### Community 3 - "Project Package"
Cohesion: 0.33
Nodes (5): Brax PPO, Docker, MLflow, Parallel MJX Random Agent, World Model Playground

### Community 4 - "parse_args"
Cohesion: 0.40
Nodes (5): main(), parse_args(), Namespace, Run a random agent in parallel MJX CartpoleBalance environments., non_negative_int()

### Community 5 - "load_policy"
Cohesion: 0.50
Nodes (3): Train a Brax PPO agent to solve MJX CartpoleBalance., Returns a checkpoint directory, resolving a root to its latest step., resolve_checkpoint()

### Community 6 - "run_live_rollout"
Cohesion: 0.50
Nodes (4): Any, Carry, Steps a rollout while showing one state in MuJoCo's live viewer., run_live_rollout()

### Community 7 - "dependencies"
Cohesion: 0.50
Nodes (3): @opencode-ai/plugin, dependencies, @opencode-ai/plugin

### Community 8 - "load_cartpole"
Cohesion: 0.67
Nodes (3): MjxEnv, load_cartpole(), Loads CartpoleBalance, optionally resetting across the full pole circle.

### Community 9 - "parse_args"
Cohesion: 0.33
Nodes (6): Path, parse_args(), Namespace, Records one deterministic policy rollout to an MP4 file., record_policy_video(), positive_int()

### Community 13 - "make_fixed_scale_ppo_networks"
Cohesion: 0.50
Nodes (4): make_fixed_scale_ppo_networks(), Applies fixed per-feature scaling to Cartpole observations., Creates PPO networks that always use fixed observation scaling., scale_observations()

## Knowledge Gaps
- **6 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `@opencode-ai/plugin`, `Parallel MJX Random Agent`, `Brax PPO` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_live_rollout()` connect `run_live_rollout` to `load_cartpole`, `Main Entry Point`, `cartpole.py`, `parse_args`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `main()` connect `Main Entry Point` to `load_policy`, `load_cartpole`, `parse_args`, `make_fixed_scale_ppo_networks`, `load_policy`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `render_policy()` connect `Main Entry Point` to `load_policy`, `run_live_rollout`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `main()` (e.g. with `make_fixed_scale_ppo_networks()` and `load_cartpole()`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_live_rollout()` (e.g. with `main()` and `render_policy()`) actually correct?**
  _`run_live_rollout()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `parse_args()` (e.g. with `non_negative_int()` and `positive_int()`) actually correct?**
  _`parse_args()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `make_fixed_scale_ppo_networks()` (e.g. with `main()` and `scale_observations()`) actually correct?**
  _`make_fixed_scale_ppo_networks()` has 2 INFERRED edges - model-reasoned connections that need verification._
