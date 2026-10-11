# marimo cells display their last expression, which B018 flags as useless.
# ruff: noqa: B018

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json

    import imageio_ffmpeg
    import marimo as mo
    import matplotlib.pyplot as plt
    import mediapy
    import mujoco
    import numpy as np

    from world_model_playground.cartpole import load_cartpole

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    return json, load_cartpole, mediapy, mo, mujoco, np, plt


@app.cell
def title(mo):
    mo.md(r"""
    # CartPole transition viewer

    Visualize collected transitions $(o_t, a_t) \to o_{t+1}$ from
    `data/cartpole_trajectories`, both as MuJoCo renders and as observation
    vectors `[cart_pos, cos θ, sin θ, cart_vel, ang_vel]`.
    """)


@app.cell
def collection_picker(mo):
    DATA_DIR = mo.notebook_dir().parents[2] / "data" / "cartpole_trajectories"
    OBS_NAMES = ("cart_pos", "cos θ", "sin θ", "cart_vel", "ang_vel")

    _collections = sorted(
        (p for p in DATA_DIR.iterdir() if (p / "collection.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    collection_picker = mo.ui.dropdown(
        options=[p.name for p in _collections],
        value=_collections[0].name if _collections else None,
        label="Collection (newest first)",
    )
    collection_picker
    return DATA_DIR, OBS_NAMES, collection_picker


@app.cell
def collection_meta(DATA_DIR, collection_picker, json, mo):
    mo.stop(collection_picker.value is None, mo.md("Select a collection to start."))
    collection_dir = DATA_DIR / collection_picker.value
    collection = json.loads((collection_dir / "collection.json").read_text())
    mo.tree(collection, label="collection.json")
    return collection, collection_dir


@app.cell
def load_transitions(collection_dir, np):
    with np.load(collection_dir / "transitions.npz") as _npz:
        transitions = {_key: _npz[_key] for _key in _npz.files if _key != "metadata"}
    transitions["theta"] = np.arctan2(
        transitions["observations"][:, 2], transitions["observations"][:, 1]
    )
    transitions["next_theta"] = np.arctan2(
        transitions["next_observations"][:, 2], transitions["next_observations"][:, 1]
    )
    source_names = [str(_name) for _name in transitions["action_source_names"]]
    episode_ids = np.unique(transitions["episode_ids"])
    episode_lengths = np.bincount(transitions["episode_ids"])
    return episode_ids, episode_lengths, source_names, transitions


@app.cell
def load_images(collection_dir, mo, np):
    with (
        mo.status.spinner(title="Decompressing images (~4.6 GB)..."),
        np.load(collection_dir / "image_transitions.npz") as _npz,
    ):
        images = _npz["images"]
        next_images = _npz["next_images"]
    return images, next_images


@app.cell(hide_code=True)
def overview_md(mo):
    mo.md("""
    ## Dataset overview
    """)


@app.cell(hide_code=True)
def overview(
    SOURCE_COLORS,
    episode_ids,
    episode_lengths,
    np,
    plt,
    source_names,
    transitions,
):
    _fig, _axes = plt.subplot_mosaic(
        [
            ["sources", "actions", "episodes"],
            ["cart_pos", "theta", "cart_vel"],
            ["ang_vel", "reward", "."],
        ],
        figsize=(12, 8),
        layout="constrained",
    )
    _src = transitions["action_source"]
    _counts = np.bincount(_src, minlength=len(source_names))
    _axes["sources"].bar(source_names, _counts, color=SOURCE_COLORS)
    _axes["sources"].set_title("Action source")
    for _k, _name in enumerate(source_names):
        _axes["actions"].hist(
            transitions["actions"][_src == _k, 0],
            bins=50,
            range=(-1, 1),
            histtype="step",
            lw=1.5,
            color=SOURCE_COLORS[_k],
            label=_name,
            density=True,
        )
    _axes["actions"].set_title("$a_t$ by source (density)")
    _axes["actions"].legend(fontsize=8)
    _axes["episodes"].hist(episode_lengths[episode_ids], bins=20)
    _axes["episodes"].set_title(f"Episode length ({len(episode_ids)} episodes)")
    _features = {
        "cart_pos": transitions["observations"][:, 0],
        "theta": transitions["theta"],
        "cart_vel": transitions["observations"][:, 3],
        "ang_vel": transitions["observations"][:, 4],
        "reward": transitions["rewards"],
    }
    for _key, _values in _features.items():
        _axes[_key].hist(_values, bins=80)
        _axes[_key].set_title("θ = atan2(sin, cos)" if _key == "theta" else _key)
        _axes[_key].set_yscale("log")
    _fig


@app.cell(hide_code=True)
def camera_model(collection, load_cartpole, mujoco, np):
    _env = load_cartpole(full_circle_reset=collection["full_circle_reset"])
    cartpole_model = _env.mj_model
    CONTROL_DT = float(_env.dt)
    _data = mujoco.MjData(cartpole_model)
    mujoco.mj_forward(cartpole_model, _data)
    _cam = cartpole_model.camera(collection["camera"]).id
    _pole_geom = cartpole_model.geom("pole_1")

    CAMERA_POS = _data.cam_xpos[_cam].copy()
    CAMERA_ROT = (
        _data.cam_xmat[_cam].reshape(3, 3).copy()
    )  # columns: camera x, y, z axes in world
    CAMERA_FOVY = float(cartpole_model.cam_fovy[_cam])
    PIVOT = _data.xpos[
        cartpole_model.body("pole_1").id
    ].copy()  # hinge position at cart_pos = 0
    POLE_LENGTH = 2.0 * float(
        _pole_geom.size[1]
    )  # capsule from pivot along the pole z-axis

    def pole_segment(obs):
        """World-frame (pivot, tip) of the pole implied by observation rows."""
        obs = np.atleast_2d(obs)
        pivot = np.tile(PIVOT, (len(obs), 1))
        pivot[:, 0] += obs[:, 0]
        axis = np.stack(
            [obs[:, 2], np.zeros(len(obs)), obs[:, 1]], axis=-1
        )  # body z = (sin θ, 0, cos θ)
        return pivot, pivot + POLE_LENGTH * axis

    def project(points, height, width):
        """Pinhole projection of world points into image pixel coordinates (u, v)."""
        cam = (np.asarray(points) - CAMERA_POS) @ CAMERA_ROT
        depth = -cam[..., 2]  # MuJoCo cameras look along -z
        focal = 0.5 * height / np.tan(np.deg2rad(CAMERA_FOVY) / 2)
        u = 0.5 * width + focal * cam[..., 0] / depth - 0.5
        v = 0.5 * height - focal * cam[..., 1] / depth - 0.5
        return np.stack([u, v], axis=-1)

    return CONTROL_DT, cartpole_model, pole_segment, project


@app.cell(hide_code=True)
def draw_helpers(np, pole_segment, project):
    SOURCE_COLORS = ("tab:blue", "tab:orange", "tab:red")

    def draw_frame(ax, image, obs, title, action=None, overlay=True):
        """Shows a rendered frame, optionally with the pole implied by `obs` and the applied action."""
        height, width = image.shape[:2]
        ax.imshow(image)
        if overlay:
            pivot, tip = pole_segment(obs)
            (pu, pv), (tu, tv) = project(np.concatenate([pivot, tip]), height, width)
            ax.plot(
                [pu, tu], [pv, tv], color="red", lw=1.5, ls="--", label="pole from obs"
            )
            ax.plot(pu, pv, "o", color="red", ms=4)
        if action is not None:
            pivot, _ = pole_segment(obs)
            pu, pv = project(pivot, height, width)[0]
            ax.annotate(
                "",
                xy=(pu + 60 * action, pv + 25),
                xytext=(pu, pv + 25),
                arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 2},
            )
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    return SOURCE_COLORS, draw_frame


@app.cell(hide_code=True)
def explorer_md(mo):
    mo.md(r"""
    ## Transition explorer

    Pick a transition $(o_t, a_t) \to o_{t+1}$. The red dashed line is the pole
    reconstructed from the observation and projected into the camera, so it should
    sit on the rendered pole. The yellow arrow is the action applied to the cart.
    """)


@app.cell(hide_code=True)
def explorer_episode(episode_ids, episode_lengths, mo):
    episode_picker = mo.ui.dropdown(
        options={f"{_e} ({episode_lengths[_e]} steps)": int(_e) for _e in episode_ids},
        value=f"{episode_ids[0]} ({episode_lengths[episode_ids[0]]} steps)",
        label="Episode",
    )
    overlay_toggle = mo.ui.checkbox(value=True, label="Overlay pole from observation")
    timeline_window = mo.ui.slider(
        20, 1000, value=200, step=20, label="Timeline half-window", show_value=True
    )
    clip_seconds = mo.ui.slider(
        1, 20, value=3, label="Clip length (s)", show_value=True
    )
    clip_button = mo.ui.run_button(label="Render clip from t")
    return (
        clip_button,
        clip_seconds,
        episode_picker,
        overlay_toggle,
        timeline_window,
    )


@app.cell(hide_code=True)
def explorer_controls(episode_picker, mo, np, overlay_toggle, transitions):
    episode_rows = np.flatnonzero(transitions["episode_ids"] == episode_picker.value)
    episode_rows = episode_rows[np.argsort(transitions["timesteps"][episode_rows])]
    step_slider = mo.ui.slider(
        0,
        len(episode_rows) - 1,
        value=0,
        label="t",
        include_input=True,
        full_width=True,
    )
    mo.hstack(
        [episode_picker, overlay_toggle, step_slider], widths=[1, 1, 4], align="center"
    )
    return episode_rows, step_slider


@app.cell(hide_code=True)
def explorer_view(
    OBS_NAMES,
    draw_frame,
    episode_rows,
    images,
    mo,
    next_images,
    np,
    overlay_toggle,
    plt,
    source_names,
    step_slider,
    transitions,
):
    transition_index = int(episode_rows[step_slider.value])
    _o = transitions["observations"][transition_index]
    _o_next = transitions["next_observations"][transition_index]
    _a = float(transitions["actions"][transition_index, 0])
    _source = source_names[transitions["action_source"][transition_index]]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    draw_frame(
        _axes[0],
        images[transition_index],
        _o,
        f"$o_t$  (t = {transitions['timesteps'][transition_index]})",
        action=_a,
        overlay=overlay_toggle.value,
    )
    draw_frame(
        _axes[1],
        next_images[transition_index],
        _o_next,
        "$o_{t+1}$",
        overlay=overlay_toggle.value,
    )
    _fig.suptitle(
        f"$a_t$ = {_a:+.3f} ({_source})   ·   $r_{{t+1}}$ = {transitions['rewards'][transition_index]:.3f}"
        f"   ·   done = {bool(transitions['dones'][transition_index])}   ·   row {transition_index}"
    )

    _names = (*OBS_NAMES, "θ (atan2)")
    _ot = np.append(_o, transitions["theta"][transition_index])
    _ot1 = np.append(_o_next, transitions["next_theta"][transition_index])
    _table = mo.ui.table(
        [
            {
                "feature": _n,
                "o_t": f"{_x:+.4f}",
                "o_t+1": f"{_y:+.4f}",
                "Δ": f"{_y - _x:+.4f}",
            }
            for _n, _x, _y in zip(_names, _ot, _ot1)
        ],
        selection=None,
        pagination=False,
        show_column_summaries=False,
    )
    mo.vstack([_fig, _table])
    return (transition_index,)


@app.cell(hide_code=True)
def timeline_md(mo):
    mo.md(r"""
    ### Episode timeline

    Observations and actions around the selected $t$ (dashed line), with actions
    colored by source. The clip plays the stored `images` from $t$ onward.
    """)


@app.cell(hide_code=True)
def timeline(
    SOURCE_COLORS,
    episode_rows,
    mo,
    np,
    plt,
    source_names,
    step_slider,
    timeline_window,
    transition_index,
    transitions,
):
    _t = step_slider.value
    _window = episode_rows[
        max(0, _t - timeline_window.value) : _t + timeline_window.value + 1
    ]
    _steps = transitions["timesteps"][_window]
    _series = {
        "θ (unwrapped)": np.unwrap(transitions["theta"][_window]),
        "cart_pos": transitions["observations"][_window, 0],
        "cart_vel": transitions["observations"][_window, 3],
        "ang_vel": transitions["observations"][_window, 4],
    }
    _fig, _axes = plt.subplots(
        len(_series) + 1, 1, figsize=(12, 7), sharex=True, layout="constrained"
    )
    for _ax, (_name, _values) in zip(_axes, _series.items()):
        _ax.plot(_steps, _values, lw=1)
        _ax.set_ylabel(_name)
    _src = transitions["action_source"][_window]
    for _k, _name in enumerate(source_names):
        _mask = _src == _k
        _axes[-1].scatter(
            _steps[_mask],
            transitions["actions"][_window[_mask], 0],
            s=6,
            color=SOURCE_COLORS[_k],
            label=_name,
        )
    _axes[-1].set_ylabel("$a_t$")
    _axes[-1].set_xlabel("timestep")
    _axes[-1].legend(fontsize=8, ncols=3, loc="upper right")
    for _ax in _axes:
        _ax.axvline(
            transitions["timesteps"][transition_index], color="k", ls="--", lw=1
        )
    mo.vstack([timeline_window, _fig])


@app.cell(hide_code=True)
def clip(
    CONTROL_DT,
    clip_button,
    clip_seconds,
    episode_rows,
    images,
    mediapy,
    mo,
    step_slider,
):
    _n = int(clip_seconds.value / CONTROL_DT)
    _rows = episode_rows[step_slider.value : step_slider.value + _n]
    _controls = mo.hstack([clip_seconds, clip_button], justify="start")
    mo.vstack(
        [
            _controls,
            mo.video(
                mediapy.compress_video(images[_rows], fps=1 / CONTROL_DT),
                loop=True,
                autoplay=True,
            )
            if clip_button.value
            else mo.md(
                f"_Click to render {len(_rows)} frames starting at t = {step_slider.value}._"
            ),
        ]
    )


@app.cell(hide_code=True)
def checks_md(mo):
    mo.md(r"""
    ## Consistency checks

    1. **Chaining**: within an episode, $o_{t+1}$ of row $t$ must equal $o_t$ of row $t+1$, and likewise for images.
    2. **Overlay**: the pole reconstructed from the observation, projected into the `fixed` camera, must match the render.
    3. **Re-render**: rendering MuJoCo from `qpos = [cart_pos, atan2(sin θ, cos θ)]` must reproduce the stored image.
       The residual is compared to the change between consecutive frames, so "small" has a reference scale.
    """)


@app.cell(hide_code=True)
def check_chaining(episode_ids, images, mo, next_images, np, transitions):
    _order = np.lexsort((transitions["timesteps"], transitions["episode_ids"]))
    _same_episode = (
        transitions["episode_ids"][_order[1:]]
        == transitions["episode_ids"][_order[:-1]]
    )
    chain_current, chain_next = _order[:-1][_same_episode], _order[1:][_same_episode]

    _obs_gap = np.abs(
        transitions["next_observations"][chain_current]
        - transitions["observations"][chain_next]
    ).max(axis=1)
    _step_gap = (
        transitions["timesteps"][chain_next] - transitions["timesteps"][chain_current]
    )
    _image_match = np.fromiter(
        (
            np.array_equal(next_images[_i], images[_j])
            for _i, _j in zip(chain_current, chain_next)
        ),
        dtype=bool,
        count=len(chain_current),
    )
    _ok = lambda _flag: "✅" if _flag else "❌"
    mo.md(f"""
    | check | result | |
    |---|---|---|
    | consecutive pairs within episodes | {len(chain_current)} | |
    | timestep increments by 1 | {int((_step_gap == 1).sum())} / {len(_step_gap)} | {_ok((_step_gap == 1).all())} |
    | `next_observations[t] == observations[t+1]` (max abs gap) | {_obs_gap.max():.2e} | {_ok(_obs_gap.max() < 1e-6)} |
    | `next_images[t] == images[t+1]` | {int(_image_match.sum())} / {len(_image_match)} | {_ok(_image_match.all())} |
    | `done` transitions | {int(transitions["dones"].sum())} | {"⚠️ single never-ending episode" if len(episode_ids) == 1 else ""} |
    """)


@app.cell(hide_code=True)
def check_overlay(draw_frame, images, np, plt, transitions):
    # Rows whose θ is closest to evenly spaced targets, so the gallery covers the full circle.
    _targets = np.linspace(-np.pi, np.pi, 8, endpoint=False) + np.pi / 8
    _gallery = list(
        dict.fromkeys(
            int(np.argmin(np.abs(np.angle(np.exp(1j * (transitions["theta"] - _x))))))
            for _x in _targets
        )
    )  # angles missing from the data map to the same row, so drop duplicates
    _fig, _axes = plt.subplots(2, 4, figsize=(14, 5.5), layout="constrained")
    for _ax, _row in zip(_axes.flat, _gallery):
        draw_frame(
            _ax,
            images[_row],
            transitions["observations"][_row],
            f"row {_row} · θ = {transitions['theta'][_row]:+.2f}",
        )
    for _ax in _axes.flat[len(_gallery) :]:
        _ax.set_axis_off()
    _fig.suptitle("Overlay check across pole angles (red = pole from observation)")
    _fig


@app.cell(hide_code=True)
def check_rerender(
    cartpole_model,
    collection,
    images,
    mujoco,
    next_images,
    np,
    transitions,
):
    def render_from_obs(rows, lag_correction=False):
        """Re-renders frames from observations by setting the slider and hinge qpos.

        MJX computes `xmat` (hence cos θ, sin θ) before integrating qpos, so the observed angle lags
        one physics step behind; `lag_correction` advances it by ang_vel * timestep.
        """
        _data = mujoco.MjData(cartpole_model)
        _height, _width = images.shape[1:3]
        with mujoco.Renderer(cartpole_model, height=_height, width=_width) as _renderer:
            _frames = []
            for _row in rows:
                _o = transitions["observations"][_row]
                _theta = np.arctan2(_o[2], _o[1]) + (
                    _o[4] * cartpole_model.opt.timestep if lag_correction else 0.0
                )
                _data.qpos[:] = (_o[0], _theta)
                mujoco.mj_forward(cartpole_model, _data)
                _renderer.update_scene(_data, camera=collection["camera"])
                _frames.append(_renderer.render())
        return np.stack(_frames)

    rerender_rows = np.sort(
        np.random.default_rng(0).choice(len(images), size=512, replace=False)
    )
    rerender_frames = render_from_obs(rerender_rows)
    rerender_residual = np.abs(
        rerender_frames.astype(np.int16) - images[rerender_rows]
    ).mean(axis=(1, 2, 3))
    rerender_residual_corrected = np.abs(
        render_from_obs(rerender_rows, lag_correction=True).astype(np.int16)
        - images[rerender_rows]
    ).mean(axis=(1, 2, 3))
    frame_change = np.abs(
        next_images[rerender_rows].astype(np.int16) - images[rerender_rows]
    ).mean(axis=(1, 2, 3))
    return (
        frame_change,
        rerender_frames,
        rerender_residual,
        rerender_residual_corrected,
        rerender_rows,
    )


@app.cell(hide_code=True)
def check_rerender_view(
    cartpole_model,
    frame_change,
    images,
    mo,
    np,
    plt,
    rerender_frames,
    rerender_residual,
    rerender_residual_corrected,
    rerender_rows,
    transitions,
):
    _worst = int(np.argmax(rerender_residual))
    _fig, _axes = plt.subplot_mosaic(
        [["hist", "stored", "rendered", "diff"]],
        figsize=(15, 3.4),
        width_ratios=[1.3, 1, 1, 1],
        layout="constrained",
    )
    _bins = np.logspace(
        -4, np.log10(max(rerender_residual.max(), frame_change.max()) + 1e-3), 50
    )
    _axes["hist"].hist(
        np.maximum(rerender_residual, 1e-4),
        bins=_bins,
        alpha=0.7,
        label="re-render vs stored",
    )
    _axes["hist"].hist(
        np.maximum(rerender_residual_corrected, 1e-4),
        bins=_bins,
        alpha=0.7,
        label="… with θ + ang_vel·dt",
    )
    _axes["hist"].hist(
        np.maximum(frame_change, 1e-4), bins=_bins, alpha=0.7, label="frame t vs t+1"
    )
    _axes["hist"].set_xscale("log")
    _axes["hist"].set_xlabel("mean |Δ pixel| (0-255)")
    _axes["hist"].legend(fontsize=8)
    _row = rerender_rows[_worst]
    _axes["stored"].imshow(images[_row])
    _axes["stored"].set_title(f"stored, row {_row} (worst)")
    _axes["rendered"].imshow(rerender_frames[_worst])
    _axes["rendered"].set_title("re-rendered from obs")
    _axes["diff"].imshow(
        np.abs(rerender_frames[_worst].astype(np.int16) - images[_row]).sum(-1),
        cmap="magma",
    )
    _axes["diff"].set_title(f"|diff|, mean = {rerender_residual[_worst]:.3f}")
    for _k in ("stored", "rendered", "diff"):
        _axes[_k].set_axis_off()
    mo.vstack(
        [
            mo.md(
                f"Re-render residual over {len(rerender_rows)} random rows: median **{np.median(rerender_residual):.4f}**, "
                f"max **{rerender_residual.max():.4f}**, versus a median consecutive-frame change of "
                f"**{np.median(frame_change):.4f}**. "
                f"Correlation of the residual with |ang_vel|: **{np.corrcoef(np.abs(transitions['observations'][rerender_rows, 4]), rerender_residual)[0, 1]:.3f}**. "
                f"Advancing θ by `ang_vel · timestep` brings the max residual to **{rerender_residual_corrected.max():.4f}**: "
                "the observed angle is computed before the integration step, so it lags the image (and `cart_pos`) by one physics step "
                f"({cartpole_model.opt.timestep * 1e3:.0f} ms)."
            ),
            _fig,
        ]
    )


@app.cell(hide_code=True)
def dynamics_md(mo):
    mo.md(r"""
    ## Dynamics view

    How $o_{t+1} - o_t$ depends on $a_t$ and on the state: this is what a world model has to learn,
    and the coverage by action source shows how much of it the dataset actually explores.
    """)


@app.cell(hide_code=True)
def dynamics(SOURCE_COLORS, np, plt, source_names, transitions):
    _delta = transitions["next_observations"] - transitions["observations"]
    _a = transitions["actions"][:, 0]
    _src = transitions["action_source"]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for _k in np.argsort(
        -np.bincount(_src)
    ):  # draw the most common source first so rarer ones stay visible
        _mask = _src == _k
        _kw = {
            "s": 3,
            "alpha": 0.4,
            "color": SOURCE_COLORS[_k],
            "label": source_names[_k],
            "rasterized": True,
        }
        _axes[0].scatter(_a[_mask], _delta[_mask, 3], **_kw)
        _axes[1].scatter(_a[_mask], _delta[_mask, 4], **_kw)
        _axes[2].scatter(
            transitions["theta"][_mask], transitions["observations"][_mask, 4], **_kw
        )
    _axes[0].set(
        xlabel="$a_t$", ylabel="Δ cart_vel", title="Cart acceleration vs action"
    )
    _axes[1].set(
        xlabel="$a_t$", ylabel="Δ ang_vel", title="Pole acceleration vs action"
    )
    _axes[2].set(xlabel="θ", ylabel="ang_vel", title="Phase portrait coverage")
    _axes[2].legend(markerscale=4, fontsize=8)
    _fig


if __name__ == "__main__":
    app.run()
