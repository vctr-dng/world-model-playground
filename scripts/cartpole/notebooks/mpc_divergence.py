# marimo cells display their last expression, which B018 flags as useless.
# ruff: noqa: B018

"""Compare a linear CartPole world-model rollout against the real environment."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import tempfile
    from pathlib import Path

    import anywidget
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import traitlets

    return (
        Path,
        anywidget,
        jax,
        jnp,
        json,
        mlflow,
        mo,
        np,
        plt,
        tempfile,
        traitlets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stochastic PPO MPC: model divergence

    Each candidate action sequence is sampled autoregressively from the
    stochastic PPO policy while rolling forward through the learned linear world
    model. The candidate whose imagined states stay closest to the upright,
    centered CartPole configuration is selected. Its fixed action sequence is
    then applied to the real Brax environment so the plot exposes the difference
    between imagined and actual dynamics.

    This is an intentionally open-loop execution over the selected horizon.
    Standard MPC would execute only the first action and replan; keeping the
    plan fixed makes accumulated model error visible.
    """)


@app.cell
def _(mo):
    project_root = mo.notebook_dir().parents[2]
    collection_dir = (
        project_root / "data/cartpole_trajectories/eb6b4f702db048849f96207af13bd57e"
    )
    model_path = project_root / "artifacts/cartpole_world_model/linear_model.npz"
    return collection_dir, model_path


@app.cell
def _(collection_dir, json, model_path):
    from world_model_playground.wm.linear import load_linear_model

    collection = json.loads((collection_dir / "collection.json").read_text())
    world_model = load_linear_model(model_path)
    return collection, world_model


@app.cell
def _(Path, collection, jax, mlflow, tempfile):
    from world_model_playground.cartpole import load_cartpole
    from world_model_playground.policy import load_policy, select_run_checkpoint

    mlflow.set_tracking_uri(collection["mlflow_tracking_uri"])
    env = load_cartpole(full_circle_reset=collection["full_circle_reset"])
    with tempfile.TemporaryDirectory(
        prefix="cartpole_mpc_checkpoint_"
    ) as checkpoint_dir:
        _, _, checkpoint_path = select_run_checkpoint(
            collection["mlflow_run_id"], Path(checkpoint_dir)
        )
        stochastic_policy = jax.jit(
            load_policy(env, checkpoint_path, deterministic=False)
        )
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)
    return env, reset, step, stochastic_policy


@app.cell
def _(jax, jnp, np):
    def balance_cost(observation):
        """Score an imagined state; lower is better for balance around theta=0."""
        cart_position, cos_theta, sin_theta, cart_velocity, angular_velocity = (
            observation
        )
        theta = np.arctan2(sin_theta, cos_theta)
        return (
            2.0 * cart_position**2
            + 6.0 * theta**2
            + 0.1 * cart_velocity**2
            + 0.05 * angular_velocity**2
        )

    def sample_policy_plan(world_model, policy, initial_observation, horizon, rng):
        """Sample a fixed action sequence from PPO along an imagined model rollout."""
        prediction = initial_observation[None, :]
        predictions = [initial_observation]
        actions = []
        for _ in range(horizon):
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(
                policy(jnp.asarray(prediction, dtype=jnp.float32), action_rng)[0],
                dtype=np.float64,
            ).reshape(-1)
            prediction = world_model.predict_next(prediction, action[None, :])
            actions.append(action)
            predictions.append(prediction[0])
        return np.asarray(actions), np.asarray(predictions)

    def run_divergence_trial(
        env,
        reset,
        step,
        world_model,
        policy,
        *,
        horizon,
        candidates,
        seed,
    ):
        rng = jax.random.key(seed)
        rng, reset_rng = jax.random.split(rng)
        initial_state = reset(reset_rng)
        initial_observation = np.asarray(initial_state.obs, dtype=np.float64)

        candidate_actions = []
        candidate_predictions = []
        candidate_scores = []
        for _ in range(candidates):
            rng, candidate_rng = jax.random.split(rng)
            actions, predictions = sample_policy_plan(
                world_model,
                policy,
                initial_observation,
                horizon,
                candidate_rng,
            )
            candidate_actions.append(actions)
            candidate_predictions.append(predictions)
            candidate_scores.append(
                sum(balance_cost(state) for state in predictions[1:])
            )

        selected_index = int(np.argmin(candidate_scores))
        selected_actions = candidate_actions[selected_index]
        predicted_observations = candidate_predictions[selected_index]

        actual_state = initial_state
        actual_states = [initial_state]
        actual_observations = [initial_observation]
        rewards = []
        for action in selected_actions:
            actual_state = step(actual_state, jnp.asarray(action, dtype=jnp.float32))
            actual_states.append(actual_state)
            actual_observations.append(np.asarray(actual_state.obs, dtype=np.float64))
            rewards.append(float(actual_state.reward))

        actual_observations = np.asarray(actual_observations)
        divergence = predicted_observations - actual_observations
        return {
            "actions": selected_actions,
            "actual": actual_observations,
            "actual_states": actual_states,
            "candidate_scores": np.asarray(candidate_scores),
            "divergence": divergence,
            "predicted": predicted_observations,
            "rewards": np.asarray(rewards),
            "selected_index": selected_index,
        }

    return balance_cost, run_divergence_trial, sample_policy_plan


@app.cell
def _(mo):
    horizon_slider = mo.ui.slider(5, 50, value=20, step=1, label="Rollout horizon")
    candidates_slider = mo.ui.slider(4, 64, value=24, step=4, label="PPO candidates")
    seed_slider = mo.ui.slider(0, 1000, value=0, step=1, label="Reset and policy seed")
    run_button = mo.ui.run_button(label="Sample PPO plans and compare")
    mo.vstack(
        [
            mo.hstack([horizon_slider, candidates_slider, seed_slider]),
            run_button,
        ]
    )
    return candidates_slider, horizon_slider, run_button, seed_slider


@app.cell
def _(
    candidates_slider,
    env,
    horizon_slider,
    mo,
    reset,
    run_button,
    run_divergence_trial,
    seed_slider,
    step,
    stochastic_policy,
    world_model,
):
    mo.stop(run_button.value is False, mo.md("Press **Sample PPO plans and compare**."))
    trial = run_divergence_trial(
        env,
        reset,
        step,
        world_model,
        stochastic_policy,
        horizon=int(horizon_slider.value),
        candidates=int(candidates_slider.value),
        seed=int(seed_slider.value),
    )
    return (trial,)


@app.cell(hide_code=True)
def _(np, trial):
    final_divergence = float(np.linalg.norm(trial["divergence"][-1]))
    one_step_divergence = float(np.linalg.norm(trial["divergence"][1]))
    summary = {
        "candidate_score_range": [
            float(trial["candidate_scores"].min()),
            float(trial["candidate_scores"].max()),
        ],
        "final_state_divergence_l2": final_divergence,
        "one_step_divergence_l2": one_step_divergence,
        "selected_candidate": trial["selected_index"],
        "total_actual_reward": float(trial["rewards"].sum()),
    }
    return (summary,)


@app.cell(hide_code=True)
def _(mo, summary):
    mo.md(
        "\n".join(
            [
                "## Selected plan",
                f"- Candidate: `{summary['selected_candidate']}`",
                f"- Actual cumulative reward: `{summary['total_actual_reward']:.3f}`",
                f"- One-step state divergence: `{summary['one_step_divergence_l2']:.4f}`",
                f"- Final state divergence: `{summary['final_state_divergence_l2']:.4f}`",
            ]
        )
    )


@app.cell(hide_code=True)
def _(anywidget, horizon_slider, mo, traitlets):
    class PolePlayback(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const root = document.createElement("div");
          root.className = "pole-playback";
          const button = document.createElement("button");
          const range = document.createElement("input");
          range.type = "range";
          range.min = "0";
          range.step = "1";
          const label = document.createElement("span");
          const loopLabel = document.createElement("label");
          const loop = document.createElement("input");
          loop.type = "checkbox";
          loopLabel.append(loop, " Loop");
          root.append(button, range, label, loopLabel);
          el.append(root);

          let timer = null;
          const save = (key, value) => {
            model.set(key, value);
            model.save_changes();
          };
          const tick = () => {
            const maximum = model.get("maximum");
            const next = model.get("step") + 1;
            if (next > maximum) {
              if (model.get("loop")) {
                save("step", 0);
              } else {
                save("playing", false);
              }
            } else {
              save("step", next);
            }
          };
          const syncTimer = () => {
            if (timer !== null) {
              clearInterval(timer);
              timer = null;
            }
            if (model.get("playing")) {
              timer = setInterval(tick, model.get("interval_ms"));
            }
          };
          const update = () => {
            const step = model.get("step");
            range.max = String(model.get("maximum"));
            range.value = String(step);
            label.textContent = `Step ${step} / ${model.get("maximum")}`;
            button.textContent = model.get("playing") ? "Pause" : "Play";
            loop.checked = model.get("loop");
          };
          range.addEventListener("input", () => save("step", Number(range.value)));
          button.addEventListener("click", () => save("playing", !model.get("playing")));
          loop.addEventListener("change", () => save("loop", loop.checked));
          model.on("change:step", update);
          model.on("change:maximum", update);
          model.on("change:loop", update);
          model.on("change:playing", () => { update(); syncTimer(); });
          model.on("change:interval_ms", syncTimer);
          update();
          syncTimer();
          return () => { if (timer !== null) clearInterval(timer); };
        }
        export default { render };
        """
        _css = """
        .pole-playback { display: flex; align-items: center; gap: 0.75rem; font-family: system-ui, sans-serif; }
        .pole-playback button { background: #161616; border: 1px solid #161616; border-radius: 0.35rem; color: white; cursor: pointer; padding: 0.35rem 0.8rem; }
        .pole-playback input[type=range] { accent-color: #e56b1f; flex: 1; }
        .pole-playback label, .pole-playback span { font-size: 0.85rem; white-space: nowrap; }
        @media (prefers-color-scheme: dark) { .pole-playback button { background: #f0f0f0; border-color: #f0f0f0; color: #151515; } }
        """
        step = traitlets.Int(0).tag(sync=True)
        maximum = traitlets.Int(50).tag(sync=True)
        playing = traitlets.Bool(False).tag(sync=True)
        loop = traitlets.Bool(True).tag(sync=True)
        interval_ms = traitlets.Int(250).tag(sync=True)

    pole_playback = mo.ui.anywidget(PolePlayback(maximum=int(horizon_slider.value)))
    pole_playback
    return PolePlayback, pole_playback


@app.cell(hide_code=True, expand_output=True)
def _(np, plt, pole_playback, trial):
    steps = np.arange(len(trial["predicted"]))
    predicted = trial["predicted"]
    actual = trial["actual"]
    theta_predicted = np.unwrap(np.arctan2(predicted[:, 2], predicted[:, 1]))
    theta_actual = np.unwrap(np.arctan2(actual[:, 2], actual[:, 1]))
    divergence_norm = np.linalg.norm(trial["divergence"], axis=1)
    playback_step = min(int(pole_playback.value["step"]), len(actual) - 1)

    figure, axes = plt.subplot_mosaic(
        [
            ["cart", "angle", "overlay"],
            ["velocity", "divergence", "overlay"],
            ["action", "action", "overlay"],
        ],
        figsize=(16, 10),
        layout="constrained",
    )
    axes["cart"].plot(steps, actual[:, 0], label="actual")
    axes["cart"].plot(steps, predicted[:, 0], "--", label="model")
    axes["cart"].plot(
        playback_step, actual[playback_step, 0], "o", color="tab:blue", markersize=8
    )
    axes["cart"].plot(
        playback_step,
        predicted[playback_step, 0],
        "o",
        color="tab:orange",
        markersize=8,
    )
    axes["cart"].set(title="Cart position", xlabel="step")
    axes["cart"].legend()

    axes["angle"].plot(steps, theta_actual, label="actual")
    axes["angle"].plot(steps, theta_predicted, "--", label="model")
    axes["angle"].plot(
        playback_step, theta_actual[playback_step], "o", color="tab:blue", markersize=8
    )
    axes["angle"].plot(
        playback_step,
        theta_predicted[playback_step],
        "o",
        color="tab:orange",
        markersize=8,
    )
    axes["angle"].set(title="Pole angle", xlabel="step", ylabel="radians")
    axes["angle"].legend()

    axes["velocity"].plot(steps, actual[:, 4], label="actual angular velocity")
    axes["velocity"].plot(steps, predicted[:, 4], "--", label="model angular velocity")
    axes["velocity"].plot(
        playback_step, actual[playback_step, 4], "o", color="tab:blue", markersize=8
    )
    axes["velocity"].plot(
        playback_step,
        predicted[playback_step, 4],
        "o",
        color="tab:orange",
        markersize=8,
    )
    axes["velocity"].set(title="Angular velocity", xlabel="step")
    axes["velocity"].legend()

    axes["divergence"].plot(steps, divergence_norm, color="tab:red")
    axes["divergence"].plot(
        playback_step,
        divergence_norm[playback_step],
        "o",
        color="tab:red",
        markersize=8,
    )
    axes["divergence"].set(title="State divergence", xlabel="step", ylabel="L2 error")

    axes["action"].step(steps[:-1], trial["actions"][:, 0], where="post")
    if playback_step < len(trial["actions"]):
        axes["action"].plot(
            playback_step,
            trial["actions"][playback_step, 0],
            "o",
            color="tab:green",
            markersize=8,
        )
    axes["action"].set(
        title="Selected stochastic PPO action sequence", xlabel="step", ylabel="force"
    )

    _overlay_axis = axes["overlay"]
    _actual_state = actual[playback_step]
    _predicted_state = predicted[playback_step]
    _overlay_axis.axhline(0.0, color="0.65", linewidth=2, zorder=0)
    for _state, _color, _alpha, _label in (
        (_actual_state, "black", 1.0, "actual"),
        (_predicted_state, "tab:orange", 0.55, "predicted"),
    ):
        _cart_position, _cos_theta, _sin_theta = _state[:3]
        _pole_tip = (_cart_position + _sin_theta, _cos_theta)
        _overlay_axis.plot(
            [_cart_position, _pole_tip[0]],
            [0.0, _pole_tip[1]],
            color=_color,
            alpha=_alpha,
            linewidth=4,
            label=_label,
        )
        _overlay_axis.plot(
            _cart_position,
            0.0,
            marker="s",
            color=_color,
            alpha=_alpha,
            markersize=12,
        )
    _overlay_extent = max(
        1.5, np.abs(np.array([_actual_state[0], _predicted_state[0]])).max() + 1.25
    )
    _overlay_axis.set(
        aspect="equal",
        title=f"Pole overlay: step {playback_step}",
        xlabel="cart position",
        xlim=(-_overlay_extent, _overlay_extent),
        ylim=(-1.2, 1.2),
    )
    _overlay_axis.legend(loc="upper right")
    _overlay_axis.text(
        0.02,
        0.03,
        f"state L2 error: {np.linalg.norm(_predicted_state - _actual_state):.3f}",
        transform=_overlay_axis.transAxes,
    )
    figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## True Receding-Horizon MPC

    At each control step, the planner samples fresh stochastic-PPO action sequences,
    scores their learned-model forecasts, executes **only the first action** from the
    best sequence in Brax, and replans from the resulting real observation.
    """)


@app.cell(hide_code=True)
def _(mo):
    mpc_steps_slider = mo.ui.slider(5, 300, value=40, step=1, label="MPC control steps")
    mpc_run_button = mo.ui.run_button(label="Run true MPC rollout")
    mo.vstack(
        [
            mo.md(
                "Uses the shared rollout horizon, PPO-candidate count, and seed above."
            ),
            mo.hstack([mpc_steps_slider, mpc_run_button]),
        ]
    )
    return mpc_run_button, mpc_steps_slider


@app.cell(hide_code=True)
def _(balance_cost, jax, jnp, np, sample_policy_plan):
    def run_receding_horizon_mpc(
        env,
        reset,
        step,
        world_model,
        policy,
        *,
        horizon,
        candidates,
        control_steps,
        seed,
    ):
        """Replan at every real step and execute only each selected first action."""
        rng = jax.random.key(seed)
        rng, reset_rng = jax.random.split(rng)
        state = reset(reset_rng)
        initial_observation = np.asarray(state.obs, dtype=np.float64)
        actual_observations = [initial_observation]
        one_step_predictions = [initial_observation]
        selected_actions = []
        selected_scores = []
        forecasts = []
        rewards = []

        for _ in range(control_steps):
            current_observation = np.asarray(state.obs, dtype=np.float64)
            candidate_actions = []
            candidate_predictions = []
            candidate_scores = []
            for _ in range(candidates):
                rng, candidate_rng = jax.random.split(rng)
                actions, predictions = sample_policy_plan(
                    world_model,
                    policy,
                    current_observation,
                    horizon,
                    candidate_rng,
                )
                candidate_actions.append(actions)
                candidate_predictions.append(predictions)
                candidate_scores.append(
                    sum(balance_cost(prediction) for prediction in predictions[1:])
                )

            selected_index = int(np.argmin(candidate_scores))
            selected_plan = candidate_actions[selected_index]
            selected_forecast = candidate_predictions[selected_index]
            action = selected_plan[0]
            state = step(state, jnp.asarray(action, dtype=jnp.float32))

            selected_actions.append(action)
            selected_scores.append(candidate_scores[selected_index])
            forecasts.append(selected_forecast)
            one_step_predictions.append(selected_forecast[1])
            actual_observations.append(np.asarray(state.obs, dtype=np.float64))
            rewards.append(float(state.reward))

        actual_observations = np.asarray(actual_observations)
        one_step_predictions = np.asarray(one_step_predictions)
        return {
            "actual": actual_observations,
            "actions": np.asarray(selected_actions),
            "forecasts": np.asarray(forecasts),
            "one_step_error": np.linalg.norm(
                one_step_predictions - actual_observations, axis=1
            ),
            "one_step_predictions": one_step_predictions,
            "rewards": np.asarray(rewards),
            "selected_scores": np.asarray(selected_scores),
        }

    return (run_receding_horizon_mpc,)


@app.cell(hide_code=True)
def _(
    candidates_slider,
    env,
    horizon_slider,
    mo,
    mpc_run_button,
    mpc_steps_slider,
    reset,
    run_receding_horizon_mpc,
    seed_slider,
    step,
    stochastic_policy,
    world_model,
):
    mo.stop(mpc_run_button.value is False, mo.md("Press **Run true MPC rollout**."))
    mpc_trial = run_receding_horizon_mpc(
        env,
        reset,
        step,
        world_model,
        stochastic_policy,
        horizon=int(horizon_slider.value),
        candidates=int(candidates_slider.value),
        control_steps=int(mpc_steps_slider.value),
        seed=int(seed_slider.value),
    )
    return (mpc_trial,)


@app.cell(hide_code=True)
def _(mo, mpc_trial):
    mpc_summary = {
        "control_steps": len(mpc_trial["actions"]),
        "mean_one_step_error": float(mpc_trial["one_step_error"][1:].mean()),
        "total_reward": float(mpc_trial["rewards"].sum()),
    }
    mo.md(
        f"""### True MPC result

    - Executed first actions: `{mpc_summary["control_steps"]}`
    - Total real reward: `{mpc_summary["total_reward"]:.3f}`
    - Mean replan one-step error: `{mpc_summary["mean_one_step_error"]:.4f}`
    """
    )


@app.cell(hide_code=True, expand_output=True)
def _(mpc_trial, np, plt):
    mpc_time = np.arange(len(mpc_trial["actual"]))
    mpc_actual = mpc_trial["actual"]
    mpc_one_step = mpc_trial["one_step_predictions"]
    mpc_theta_actual = np.unwrap(np.arctan2(mpc_actual[:, 2], mpc_actual[:, 1]))
    mpc_theta_one_step = np.unwrap(np.arctan2(mpc_one_step[:, 2], mpc_one_step[:, 1]))
    mpc_figure, mpc_axes = plt.subplot_mosaic(
        [["cart", "angle"], ["error", "action"]],
        figsize=(12, 8),
        layout="constrained",
    )

    _mpc_forecast_stride = max(1, len(mpc_trial["forecasts"]) // 8)
    for _mpc_replan_step in range(0, len(mpc_trial["forecasts"]), _mpc_forecast_stride):
        _mpc_forecast = mpc_trial["forecasts"][_mpc_replan_step]
        _mpc_forecast_time = np.arange(
            _mpc_replan_step, _mpc_replan_step + len(_mpc_forecast)
        )
        mpc_axes["cart"].plot(
            _mpc_forecast_time,
            _mpc_forecast[:, 0],
            color="tab:orange",
            alpha=0.22,
            linewidth=1.5,
        )
        _mpc_forecast_theta = np.unwrap(
            np.arctan2(_mpc_forecast[:, 2], _mpc_forecast[:, 1])
        )
        mpc_axes["angle"].plot(
            _mpc_forecast_time,
            _mpc_forecast_theta,
            color="tab:orange",
            alpha=0.22,
            linewidth=1.5,
        )

    mpc_axes["cart"].plot(
        mpc_time, mpc_actual[:, 0], color="black", linewidth=2, label="actual"
    )
    mpc_axes["cart"].plot(
        mpc_time,
        mpc_one_step[:, 0],
        "--",
        color="tab:blue",
        label="replan one-step prediction",
    )
    mpc_axes["cart"].set(
        title="Cart position with selected-plan forecasts", xlabel="real MPC step"
    )
    mpc_axes["cart"].legend()

    mpc_axes["angle"].plot(
        mpc_time, mpc_theta_actual, color="black", linewidth=2, label="actual"
    )
    mpc_axes["angle"].plot(
        mpc_time,
        mpc_theta_one_step,
        "--",
        color="tab:blue",
        label="replan one-step prediction",
    )
    mpc_axes["angle"].set(
        title="Pole angle with selected-plan forecasts",
        xlabel="real MPC step",
        ylabel="radians",
    )
    mpc_axes["angle"].legend()

    mpc_axes["error"].plot(
        mpc_time, mpc_trial["one_step_error"], color="tab:red", linewidth=2
    )
    mpc_axes["error"].set(
        title="One-step model error after each replan",
        xlabel="real MPC step",
        ylabel="state L2 error",
    )

    mpc_axes["action"].step(
        mpc_time[:-1], mpc_trial["actions"][:, 0], where="post", color="tab:green"
    )
    mpc_axes["action"].set(
        title="Executed first action from each selected plan",
        xlabel="real MPC step",
        ylabel="force",
    )
    mpc_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### True MPC playback

    Black is the real CartPole after the first action from each selected MPC plan is
    executed. Transparent orange is the linear model's one-step prediction made by
    that same plan before the real action was applied.
    """)


@app.cell(hide_code=True)
def _(PolePlayback, mo, mpc_steps_slider):
    mpc_playback = mo.ui.anywidget(PolePlayback(maximum=int(mpc_steps_slider.value)))
    mpc_playback
    return (mpc_playback,)


@app.cell(hide_code=True)
def _(mpc_playback, mpc_trial, np, plt):
    mpc_animation_step = min(
        int(mpc_playback.value["step"]), len(mpc_trial["actual"]) - 1
    )
    _mpc_animation_actual = mpc_trial["actual"][mpc_animation_step]
    _mpc_animation_prediction = mpc_trial["one_step_predictions"][mpc_animation_step]

    mpc_animation_figure, mpc_animation_axis = plt.subplots(
        figsize=(7, 4), layout="constrained"
    )
    mpc_animation_axis.axhline(0.0, color="0.65", linewidth=2, zorder=0)
    for _mpc_state, _mpc_color, _mpc_alpha, _mpc_label in (
        (_mpc_animation_actual, "black", 1.0, "actual MPC state"),
        (_mpc_animation_prediction, "tab:orange", 0.55, "linear-model prediction"),
    ):
        _mpc_cart_position, _mpc_cos_theta, _mpc_sin_theta = _mpc_state[:3]
        _mpc_pole_tip = (_mpc_cart_position + _mpc_sin_theta, _mpc_cos_theta)
        mpc_animation_axis.plot(
            [_mpc_cart_position, _mpc_pole_tip[0]],
            [0.0, _mpc_pole_tip[1]],
            color=_mpc_color,
            alpha=_mpc_alpha,
            linewidth=4,
            label=_mpc_label,
        )
        mpc_animation_axis.plot(
            _mpc_cart_position,
            0.0,
            marker="s",
            color=_mpc_color,
            alpha=_mpc_alpha,
            markersize=12,
        )

    _mpc_animation_extent = max(
        1.5,
        np.abs(np.array([_mpc_animation_actual[0], _mpc_animation_prediction[0]])).max()
        + 1.25,
    )
    mpc_animation_axis.set(
        aspect="equal",
        title=f"True MPC: actual versus one-step model forecast, step {mpc_animation_step}",
        xlabel="cart position",
        xlim=(-_mpc_animation_extent, _mpc_animation_extent),
        ylim=(-1.2, 1.2),
    )
    mpc_animation_axis.legend(loc="upper right")
    mpc_animation_axis.text(
        0.02,
        0.03,
        f"state L2 error: {mpc_trial['one_step_error'][mpc_animation_step]:.3f}",
        transform=mpc_animation_axis.transAxes,
    )
    mpc_animation_figure


if __name__ == "__main__":
    app.run()
