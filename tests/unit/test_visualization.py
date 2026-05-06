import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go

from code.prep_data.pipeline import TrialBatch, build_trial_batch
from code.models.conventional_speaker import ConventionFit
from code.analysis.visualization import (
    compute_tsne_coords,
    compute_per_game_tsne_coords,
    plot_overview,
    plot_per_game_overview,
    plot_game,
)


def _make_batch_and_fit(n_games=3, n_images=4, n_reps=3, D=8):
    rng = np.random.default_rng(42)
    n_total = n_games * n_images * n_reps
    utt = rng.standard_normal((n_total, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n_total, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    game_ids = np.repeat(np.arange(n_games), n_images * n_reps)
    image_ids = np.tile(np.repeat(np.arange(n_images), n_reps), n_games)
    rep_nums = np.tile(np.arange(1, n_reps + 1), n_games * n_images)
    target_idx = (image_ids % 12).astype(np.int32)

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.array(target_idx),
        selected_idx=jnp.array(target_idx),
        listener_ids=jnp.zeros(n_total, dtype=jnp.int32),
        game_ids=jnp.array(game_ids, dtype=jnp.int32),
        image_ids=jnp.array(image_ids, dtype=jnp.int32),
        rep_num=jnp.array(rep_nums, dtype=jnp.int32),
        n_listeners=1,
        n_games=n_games,
        n_images=n_images,
    )

    fit = ConventionFit(
        speaker_m_loc=rng.standard_normal((n_games, n_images, D)).astype(np.float32),
        speaker_m_scale=np.ones((n_games, n_images), dtype=np.float32) * 0.2,
        listener_m_loc=rng.standard_normal((n_games, n_images, D)).astype(np.float32),
        listener_m_scale=np.ones((n_games, n_images), dtype=np.float32) * 0.2,
        mu_i=rng.standard_normal((n_images, D)).astype(np.float32),
        sigma_game=0.2,
        sigma_min=0.5,
        sigma_max=1.5,
    )
    return batch, fit


def test_compute_tsne_coords_columns():
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    assert set(df.columns) >= {"x", "y", "kind", "game_id", "image_id", "rep_num", "correct"}


def test_compute_tsne_coords_kinds():
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4, n_reps=3)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    assert set(df["kind"].unique()) == {"utterance", "convention", "prototype"}


def test_compute_tsne_coords_prototype_count():
    n_images = 4
    batch, fit = _make_batch_and_fit(n_games=3, n_images=n_images, n_reps=3)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    assert len(df[df["kind"] == "prototype"]) == n_images


def test_compute_tsne_coords_convention_count():
    n_games, n_images = 3, 4
    batch, fit = _make_batch_and_fit(n_games=n_games, n_images=n_images, n_reps=3)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    assert len(df[df["kind"] == "convention"]) == n_games * n_images


def test_compute_tsne_coords_deduplicates_utterances():
    # Two listeners for the same (game, image, rep) → only one point in output
    rng = np.random.default_rng(0)
    D = 8
    shared_utt = rng.standard_normal((1, D)).astype(np.float32)
    shared_utt /= np.linalg.norm(shared_utt)
    utt = np.tile(shared_utt, (2, 1))  # same utterance, two listeners
    imgs = rng.standard_normal((2, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(2, dtype=jnp.int32),
        selected_idx=jnp.zeros(2, dtype=jnp.int32),
        listener_ids=jnp.array([0, 1], dtype=jnp.int32),
        game_ids=jnp.zeros(2, dtype=jnp.int32),
        image_ids=jnp.zeros(2, dtype=jnp.int32),
        rep_num=jnp.ones(2, dtype=jnp.int32),
        n_listeners=2,
        n_games=1,
        n_images=1,
    )
    fit = ConventionFit(
        speaker_m_loc=rng.standard_normal((1, 1, D)).astype(np.float32),
        speaker_m_scale=np.ones((1, 1), dtype=np.float32),
        listener_m_loc=rng.standard_normal((1, 1, D)).astype(np.float32),
        listener_m_scale=np.ones((1, 1), dtype=np.float32),
        mu_i=rng.standard_normal((1, D)).astype(np.float32),
        sigma_game=0.2, sigma_min=0.5, sigma_max=1.5,
    )
    df = compute_tsne_coords(fit, batch, perplexity=2, seed=0)
    assert len(df[df["kind"] == "utterance"]) == 1


def test_plot_overview_returns_figure():
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)
    assert isinstance(fig, go.Figure)


def test_plot_overview_has_dropdown():
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)
    assert len(fig.layout.updatemenus) == 2
    # image menu: 1 (all) + n_images (4) = 5
    assert len(fig.layout.updatemenus[0].buttons) == 5
    # game menu: 1 (all) + n_games (3) = 4
    assert len(fig.layout.updatemenus[1].buttons) == 4


def test_plot_game_returns_figure():
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=1)
    assert isinstance(fig, go.Figure)


def test_plot_game_title_includes_game_id():
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=2)
    assert "2" in fig.layout.title.text


def test_plot_game_white_background():
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=0)
    assert fig.layout.plot_bgcolor == "white"


def test_plot_overview_white_background():
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)
    assert fig.layout.plot_bgcolor == "white"


def test_plot_game_bigger():
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=0)
    assert fig.layout.width >= 1000
    assert fig.layout.height >= 750


def test_plot_overview_bigger():
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)
    assert fig.layout.width >= 1100
    assert fig.layout.height >= 800


def test_plot_game_legend_includes_correct_incorrect():
    """Legend must explain the circle/x symbol encoding."""
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=0)
    legend_names = [t.name for t in fig.data if t.showlegend]
    assert any("correct" in (n or "").lower() for n in legend_names), legend_names
    assert any("incorrect" in (n or "").lower() for n in legend_names), legend_names


def test_plot_overview_legend_includes_utterances():
    """Grey background dots should have a legend entry."""
    batch, fit = _make_batch_and_fit()
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)
    legend_names = [t.name for t in fig.data if t.showlegend]
    assert any("utterance" in (n or "").lower() for n in legend_names), legend_names


def test_plot_game_has_arrow_annotations():
    """Consecutive rep pairs must produce arrow annotations (not bare lines)."""
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4, n_reps=3)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=1)
    assert len(fig.layout.annotations) > 0, "Expected arrow annotations between reps"


def test_plot_game_rep1_has_halo():
    """Rep 1 dots must have a visible outline ring; later reps must not."""
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4, n_reps=3)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_game(df, game_id=0)

    rep1_ring_found = False
    later_ring_found = False
    for trace in fig.data:
        if trace.name is None or not trace.name.startswith("image"):
            continue
        if trace.marker is None or trace.marker.line is None:
            continue
        widths = trace.marker.line.width
        if widths is None or not hasattr(widths, "__iter__"):
            continue
        for w, row in zip(widths, trace.customdata):
            rep_num = int(row[0])
            if rep_num == 1 and w > 0:
                rep1_ring_found = True
            if rep_num > 1 and w > 0:
                later_ring_found = True

    assert rep1_ring_found, "No rep_num=1 dot has a halo ring"
    assert not later_ring_found, "Later reps should not have a halo ring"


def test_plot_overview_game_button_includes_annotations():
    """Game dropdown buttons must carry arrow annotations in their layout args."""
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4, n_reps=3)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)
    # game buttons are in updatemenus[1]; index 0 = "all games", index 1 = first game
    game_button = fig.layout.updatemenus[1].buttons[1]
    assert len(game_button.args) >= 2, "Game button should have layout update args"
    layout_args = game_button.args[1]
    assert "annotations" in layout_args, layout_args.keys()
    assert len(layout_args["annotations"]) > 0


def test_plot_overview_image_filter_hides_other_prototypes():
    """
    Each image has its own prototype trace.  'All images' shows all n_images prototype
    traces; selecting image K shows exactly 1.
    """
    n_images = 4
    batch, fit = _make_batch_and_fit(n_games=3, n_images=n_images)
    df = compute_tsne_coords(fit, batch, perplexity=5, seed=0)
    fig = plot_overview(df)

    def _count_visible_proto_traces(button):
        visible = button.args[0]["visible"]
        return sum(
            1 for idx, trace in enumerate(fig.data)
            if visible[idx]
            and getattr(trace.marker, "symbol", None) == "diamond"
            and trace.x is not None
            and len(trace.x) > 0
            and trace.x[0] is not None  # exclude dummy legend traces
        )

    all_button = fig.layout.updatemenus[0].buttons[0]
    assert _count_visible_proto_traces(all_button) == n_images

    # buttons 1..n_images select individual images; pick image 2 (index 3)
    image_button = fig.layout.updatemenus[0].buttons[3]
    assert _count_visible_proto_traces(image_button) == 1


# ---------------------------------------------------------------------------
# compute_per_game_tsne_coords
# ---------------------------------------------------------------------------

def test_per_game_tsne_coords_columns():
    batch, fit = _make_batch_and_fit()
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    assert set(df.columns) >= {"x", "y", "kind", "game_id", "image_id", "rep_num", "correct"}


def test_per_game_tsne_coords_all_games_present():
    n_games = 3
    batch, fit = _make_batch_and_fit(n_games=n_games, n_images=4)
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    utt_game_ids = set(df[df["kind"] == "utterance"]["game_id"].unique())
    assert utt_game_ids == set(range(n_games))


def test_per_game_tsne_coords_has_utterance_convention_prototype():
    batch, fit = _make_batch_and_fit(n_games=2, n_images=3)
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    assert set(df["kind"].unique()) == {"utterance", "convention", "prototype"}


def test_per_game_tsne_coords_deduplicates_within_game():
    """Two listeners for the same (game, image, rep) → only one utterance point per game."""
    rng = np.random.default_rng(0)
    D = 8
    n = 4  # 2 games × 2 (game_id, image_id, rep_num) keys, each with 2 listeners
    utt = rng.standard_normal((n, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(n, dtype=jnp.int32),
        selected_idx=jnp.zeros(n, dtype=jnp.int32),
        listener_ids=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
        game_ids=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        image_ids=jnp.zeros(n, dtype=jnp.int32),
        rep_num=jnp.ones(n, dtype=jnp.int32),
        n_listeners=2, n_games=2, n_images=1,
    )
    fit = ConventionFit(
        speaker_m_loc=rng.standard_normal((2, 1, D)).astype(np.float32),
        speaker_m_scale=np.ones((2, 1), dtype=np.float32),
        listener_m_loc=rng.standard_normal((2, 1, D)).astype(np.float32),
        listener_m_scale=np.ones((2, 1), dtype=np.float32),
        mu_i=rng.standard_normal((1, D)).astype(np.float32),
        sigma_game=0.2, sigma_min=0.5, sigma_max=1.5,
    )
    df = compute_per_game_tsne_coords(fit, batch, perplexity=2, seed=0)
    # Each game should have exactly 1 utterance (the duplicate listener row is dropped)
    for g in [0, 1]:
        n_utts = len(df[(df["kind"] == "utterance") & (df["game_id"] == g)])
        assert n_utts == 1, f"game {g}: expected 1 utterance, got {n_utts}"


# ---------------------------------------------------------------------------
# plot_per_game_overview
# ---------------------------------------------------------------------------

def test_plot_per_game_overview_returns_figure():
    batch, fit = _make_batch_and_fit()
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    fig = plot_per_game_overview(df)
    assert isinstance(fig, go.Figure)


def test_plot_per_game_overview_has_one_subplot_per_game():
    n_games = 3
    batch, fit = _make_batch_and_fit(n_games=n_games)
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    fig = plot_per_game_overview(df)
    n_xaxes = sum(1 for k in fig.to_dict()["layout"] if k.startswith("xaxis"))
    assert n_xaxes >= n_games


def test_plot_per_game_overview_has_arrows():
    batch, fit = _make_batch_and_fit(n_games=3, n_images=4, n_reps=3)
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    fig = plot_per_game_overview(df)
    assert len(fig.layout.annotations) > 0


def test_plot_per_game_overview_white_background():
    batch, fit = _make_batch_and_fit()
    df = compute_per_game_tsne_coords(fit, batch, perplexity=3, seed=0)
    fig = plot_per_game_overview(df)
    assert fig.layout.plot_bgcolor == "white"
