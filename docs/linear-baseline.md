# Linear baseline

The linear residual model every learned model is compared with, and a notebook showing where its rollouts diverge from the real environment.

Fit the linear residual baseline on a collection, then compare it against the
real environment under MPC:

```bash
uv run python scripts/cartpole/train/train_linear.py \
  --collection-dir data/cartpole_trajectories/<collection_id>
uv run marimo edit scripts/cartpole/notebooks/mpc_divergence.py
```

---

Previous: [Patch-token world model](patch-token-wm.md) · [README](../README.md)
