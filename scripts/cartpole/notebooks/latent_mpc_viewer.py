# marimo cells display their last expression, which B018 flags as useless.
# ruff: noqa: B018

"""Browse closed-loop MPC results written by latent_mpc.py."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    import anywidget
    import imageio_ffmpeg
    import marimo as mo
    import matplotlib.pyplot as plt
    import mediapy
    import numpy as np
    import traitlets

    from world_model_playground.cartpole import FrameRenderer, load_cartpole

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    return (
        FrameRenderer,
        anywidget,
        json,
        load_cartpole,
        mediapy,
        mo,
        np,
        plt,
        traitlets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Closed-loop latent MPC

    `latent_mpc.py` runs the same batch of episodes under every controller: the
    PPO expert, random actions, and CEM model-predictive control through the
    true dynamics, the linear model, and the latent world model (scoring either
    the CartPole reward of probe-decoded latents or the distance to an upright
    goal latent). Each control step replans and executes only the first action.

    Pick a run, then an episode, to compare what each planner expected with
    what the real system did.
    """)


@app.cell
def _(mo):
    RESULTS_ROOT = (
        mo.notebook_dir().parents[2] / "artifacts" / "cartpole_world_model" / "mpc"
    )
    _runs = sorted(
        (path.parent for path in RESULTS_ROOT.glob("*/*/*/summary.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    run_picker = mo.ui.dropdown(
        options={str(path.relative_to(RESULTS_ROOT)): path for path in _runs},
        value=str(_runs[0].relative_to(RESULTS_ROOT)) if _runs else None,
        label="Run (newest first)",
    )
    run_picker
    return (run_picker,)


@app.cell
def _(json, mo, run_picker):
    mo.stop(run_picker.value is None, mo.md("No results yet: run `latent_mpc.py`."))
    run_dir = run_picker.value
    record = json.loads((run_dir / "summary.json").read_text())
    mo.vstack(
        [
            mo.md((run_dir / "results.md").read_text()),
            mo.accordion({"Parameters": mo.tree(record["params"])}),
        ]
    )
    return record, run_dir


@app.cell
def _(mo, record):
    reset_picker = mo.ui.dropdown(
        options=list(record["summary"]),
        value=next(iter(record["summary"])),
        label="Initial states",
    )
    reset_picker
    return (reset_picker,)


@app.cell
def _(mo, np, record, reset_picker, run_dir):
    results = {
        name: dict(np.load(run_dir / f"{reset_picker.value}_{name}.npz"))
        for name in record["summary"][reset_picker.value]
    }
    _figure = run_dir / f"{reset_picker.value}_rewards.png"
    mo.image(_figure.read_bytes()) if _figure.exists() else None
    return (results,)


@app.cell
def _(mo, record, results):
    _num_episodes = next(iter(results.values()))["rewards"].shape[0]
    controller_picker = mo.ui.dropdown(
        options=list(results),
        value="mpc-latent" if "mpc-latent" in results else next(iter(results)),
        label="Controller",
    )
    episode_picker = mo.ui.slider(
        0, _num_episodes - 1, value=0, label="Episode", show_value=True
    )
    mo.hstack([controller_picker, episode_picker], justify="start")
    return controller_picker, episode_picker


@app.cell(hide_code=True)
def _(episode_picker, np, plt, results):
    _figure, _axes = plt.subplots(1, 3, figsize=(16, 3.8), layout="constrained")
    for _name, _result in results.items():
        _obs = _result["observations"][episode_picker.value]
        _axes[0].plot(np.cumsum(_result["rewards"][episode_picker.value]), label=_name)
        _axes[1].plot(np.unwrap(np.arctan2(_obs[:, 2], _obs[:, 1])), label=_name)
        _axes[2].plot(_obs[:, 0], label=_name)
    _axes[0].set(title="Cumulative reward", xlabel="step")
    _axes[1].set(title="Pole angle θ (0 = upright)", xlabel="step", ylabel="rad")
    _axes[2].set(title="Cart position", xlabel="step")
    _axes[0].legend(fontsize=8)
    _figure


@app.cell(hide_code=True)
def _(anywidget, controller_picker, mo, results, traitlets):
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
          root.append(button, range, label);
          el.append(root);
          let timer = null;
          const save = (key, value) => { model.set(key, value); model.save_changes(); };
          const tick = () => {
            const next = model.get("step") + 1;
            save("step", next > model.get("maximum") ? 0 : next);
          };
          const syncTimer = () => {
            if (timer !== null) { clearInterval(timer); timer = null; }
            if (model.get("playing")) { timer = setInterval(tick, model.get("interval_ms")); }
          };
          const update = () => {
            range.max = String(model.get("maximum"));
            range.value = String(model.get("step"));
            label.textContent = `Step ${model.get("step")} / ${model.get("maximum")}`;
            button.textContent = model.get("playing") ? "Pause" : "Play";
          };
          range.addEventListener("input", () => save("step", Number(range.value)));
          button.addEventListener("click", () => save("playing", !model.get("playing")));
          model.on("change:step", update);
          model.on("change:maximum", update);
          model.on("change:playing", () => { update(); syncTimer(); });
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
        .pole-playback span { font-size: 0.85rem; white-space: nowrap; }
        @media (prefers-color-scheme: dark) { .pole-playback button { background: #f0f0f0; border-color: #f0f0f0; color: #151515; } }
        """
        step = traitlets.Int(0).tag(sync=True)
        maximum = traitlets.Int(50).tag(sync=True)
        playing = traitlets.Bool(False).tag(sync=True)
        interval_ms = traitlets.Int(200).tag(sync=True)

    _steps = results[controller_picker.value]["rewards"].shape[1]
    playback = mo.ui.anywidget(PolePlayback(maximum=_steps - 1))
    mo.vstack(
        [
            mo.md(
                "Black: real system. Orange: the plan chosen at this step, "
                "fading over its horizon. Blue: the latent probe's reading of "
                "the current frame."
            ),
            playback,
        ]
    )
    return (playback,)


@app.cell(hide_code=True, expand_output=True)
def _(controller_picker, episode_picker, np, playback, plt, results):
    def draw_pole(axis, state, **style):
        cart, cos_theta, sin_theta = state[:3]
        axis.plot([cart, cart + sin_theta], [0.0, cos_theta], linewidth=4, **style)
        axis.plot(cart, 0.0, marker="s", markersize=12, **style)

    _result = results[controller_picker.value]
    _episode = episode_picker.value
    _step = min(int(playback.value["step"]), _result["rewards"].shape[1] - 1)
    _figure, (_scene, _plan_axis) = plt.subplots(
        1, 2, figsize=(15, 4.5), width_ratios=[1.2, 1], layout="constrained"
    )
    _scene.axhline(0.0, color="0.7", linewidth=2, zorder=0)
    if "plan_observations" in _result:
        _plan = _result["plan_observations"][_episode, _step]
        for _index in range(0, len(_plan), 3):
            _alpha = 0.6 * (1 - _index / len(_plan)) + 0.05
            draw_pole(_scene, _plan[_index], color="tab:orange", alpha=_alpha)
    if "decoded_observations" in _result:
        draw_pole(
            _scene,
            _result["decoded_observations"][_episode, _step],
            color="tab:blue",
            alpha=0.5,
        )
    draw_pole(_scene, _result["observations"][_episode, _step], color="black")
    _scene.set(
        aspect="equal",
        xlim=(-2.4, 2.4),
        ylim=(-1.2, 1.2),
        title=f"{controller_picker.value}, episode {_episode}, step {_step}, "
        f"reward {_result['rewards'][_episode, _step]:.2f}",
    )

    _actual = _result["observations"][_episode]
    _theta = np.unwrap(np.arctan2(_actual[:, 2], _actual[:, 1]))
    _plan_axis.plot(_theta, color="black", label="actual θ")
    if "plan_observations" in _result:
        _plan = _result["plan_observations"][_episode, _step]
        _plan_theta = np.arctan2(_plan[:, 2], _plan[:, 1])
        # Align the plan's wrapped angles with the actual unwrapped angle.
        _plan_theta = (
            _theta[_step]
            + np.unwrap(
                np.concatenate(
                    (
                        [0.0],
                        _plan_theta - np.arctan2(_actual[_step, 2], _actual[_step, 1]),
                    )
                )
            )[1:]
        )
        _plan_axis.plot(
            np.arange(_step + 1, _step + 1 + len(_plan)),
            _plan_theta,
            color="tab:orange",
            label="plan θ",
        )
    _plan_axis.axvline(_step, color="0.6", linestyle="--")
    _plan_axis.set(
        title="Pole angle: actual and current plan", xlabel="step", ylabel="rad"
    )
    _plan_axis.legend()
    _figure


@app.cell(hide_code=True)
def _(mo):
    video_button = mo.ui.run_button(label="Render episode video")
    video_button
    return (video_button,)


@app.cell(hide_code=True)
def _(
    FrameRenderer,
    controller_picker,
    episode_picker,
    load_cartpole,
    mediapy,
    mo,
    np,
    results,
    video_button,
):
    mo.stop(not video_button.value)
    _qpos = results[controller_picker.value]["qpos"][episode_picker.value]
    _env = load_cartpole()
    with FrameRenderer(_env.mj_model, [(240, 320)], "fixed") as _renderer:
        _frames = _renderer.render(_qpos, np.zeros_like(_qpos), resolution=(240, 320))
    mo.video(mediapy.compress_video(_frames, fps=1 / _env.dt), loop=True, autoplay=True)


if __name__ == "__main__":
    app.run()
