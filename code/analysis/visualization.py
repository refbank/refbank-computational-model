import logging
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from code.models.conventional_speaker import ConventionFit
from code.prep_data.pipeline import TrialBatch

_log = logging.getLogger(__name__)

# 12-color qualitative palette, one per image slot
_IMAGE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


def compute_tsne_coords(
    fit: ConventionFit,
    batch: TrialBatch,
    role: Literal["speaker", "listener"] = "speaker",
    perplexity: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Projects all utterance embeddings, convention points m_{g,i}, and global
    prototypes mu_i to 2D via t-SNE on the combined embedding matrix.

    Returns DataFrame with columns:
      x, y          — 2D t-SNE coordinates
      kind          — "utterance", "convention", or "prototype"
      game_id       — int; -1 for prototypes
      image_id      — int
      rep_num       — int; 0 for convention points and prototypes
      correct       — bool; True if selected_idx == target_idx (utterances only)

    Utterances are deduplicated by (game_id, image_id, rep_num) since multiple
    listeners share the same utterance embedding.
    """
    utt_emb = np.array(batch.utterance_emb)
    game_ids = np.array(batch.game_ids)
    image_ids = np.array(batch.image_ids)
    rep_nums = np.array(batch.rep_num)
    correct = np.array(batch.selected_idx) == np.array(batch.target_idx)

    seen: set = set()
    utt_indices: list[int] = []
    utt_meta: list[dict] = []
    for n in range(len(utt_emb)):
        key = (int(game_ids[n]), int(image_ids[n]), int(rep_nums[n]))
        if key not in seen:
            seen.add(key)
            utt_indices.append(n)
            utt_meta.append({
                "kind": "utterance",
                "game_id": int(game_ids[n]),
                "image_id": int(image_ids[n]),
                "rep_num": int(rep_nums[n]),
                "correct": bool(correct[n]),
            })

    m_loc = fit.speaker_m_loc if role == "speaker" else fit.listener_m_loc
    conv_meta: list[dict] = []
    for g in range(batch.n_games):
        for i in range(batch.n_images):
            conv_meta.append({
                "kind": "convention",
                "game_id": g,
                "image_id": i,
                "rep_num": 0,
                "correct": False,
            })

    proto_meta: list[dict] = [
        {"kind": "prototype", "game_id": -1, "image_id": i, "rep_num": 0, "correct": False}
        for i in range(batch.n_images)
    ]

    utt_vecs = utt_emb[utt_indices]
    conv_vecs = m_loc[:batch.n_games].reshape(-1, m_loc.shape[-1])
    proto_vecs = fit.mu_i

    all_vecs = np.vstack([utt_vecs, conv_vecs, proto_vecs]).astype(np.float64)

    coords = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, n_jobs=1
    ).fit_transform(all_vecs)

    all_meta = utt_meta + conv_meta + proto_meta
    df = pd.DataFrame(all_meta)
    df["x"] = coords[:, 0].astype(np.float32)
    df["y"] = coords[:, 1].astype(np.float32)
    return df


def _arrow_annotations(utts: pd.DataFrame) -> list[dict]:
    """
    Build Plotly annotation dicts for rep-to-rep arrows within each image group.
    Each annotation is an arrow from rep t to rep t+1, colored by image.
    """
    annotations = []
    for img_id, grp in utts.groupby("image_id"):
        grp_sorted = grp.sort_values("rep_num")
        if len(grp_sorted) < 2:
            continue
        color = _IMAGE_COLORS[int(img_id) % len(_IMAGE_COLORS)]
        pts = list(zip(grp_sorted["x"].tolist(), grp_sorted["y"].tolist()))
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            annotations.append(dict(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor=color,
                text="",
            ))
    return annotations


def plot_overview(tsne_df: pd.DataFrame) -> go.Figure:
    """
    Interactive overview of all convention trajectories.

    All utterance embeddings are shown as grey dots by default. Two dropdowns
    let you filter independently:
      - Image dropdown: highlights utterances for one image across all games,
        shows only that image's prototype diamond.
      - Game dropdown: highlights utterances for one game, colored by image,
        with rep-to-rep arrows.

    Call compute_tsne_coords first; pass its output here. t-SNE is slow, so
    compute once and call this function multiple times to experiment.
    """
    utts = tsne_df[tsne_df["kind"] == "utterance"]
    convs = tsne_df[tsne_df["kind"] == "convention"]
    protos = tsne_df[tsne_df["kind"] == "prototype"]

    n_images = int(utts["image_id"].max()) + 1
    n_games = int(utts["game_id"].max()) + 1

    traces: list[go.BaseTraceType] = []

    # Trace 0: grey background (all utterances)
    traces.append(go.Scatter(
        x=utts["x"].tolist(),
        y=utts["y"].tolist(),
        mode="markers",
        marker=dict(color="lightgrey", size=4, opacity=0.6),
        hovertemplate="game=%{customdata[0]} img=%{customdata[1]} rep=%{customdata[2]}<extra></extra>",
        customdata=utts[["game_id", "image_id", "rep_num"]].values,
        name="utterances",
        showlegend=True,
    ))

    # Traces 1..n_images: one prototype diamond per image (individually toggleable)
    for i in range(n_images):
        proto_i = protos[protos["image_id"] == i]
        color = _IMAGE_COLORS[i % len(_IMAGE_COLORS)]
        traces.append(go.Scatter(
            x=proto_i["x"].tolist(),
            y=proto_i["y"].tolist(),
            mode="markers",
            marker=dict(symbol="diamond", size=14, color=color, line=dict(width=2, color="black")),
            hovertemplate=f"prototype img={i}<extra></extra>",
            name=f"proto {i}",
            showlegend=False,
        ))

    # Legend entries (always visible, no data)
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="diamond", size=14, color="grey", line=dict(width=2, color="black")),
        name="prototype μᵢ (◆)", showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="star", size=12, color="grey", line=dict(width=1, color="black")),
        name="convention m(g,i) (★)", showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=9, color="grey"),
        name="correct (○)", showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="x", size=9, color="grey"),
        name="incorrect (✗)", showlegend=True,
    ))

    # N_ALWAYS: bg + n_images proto traces + 4 legend dummies
    N_ALWAYS = 1 + n_images + 4
    t_proto_start = 1
    t_legend_start = 1 + n_images
    t_img_dots_start = N_ALWAYS
    t_img_conv_start = N_ALWAYS + n_images
    t_game_dots_start = N_ALWAYS + 2 * n_images

    # Per-image traces (hidden by default)
    for i in range(n_images):
        img_utts = utts[utts["image_id"] == i]
        color = _IMAGE_COLORS[i % len(_IMAGE_COLORS)]
        traces.append(go.Scatter(
            x=img_utts["x"].tolist(),
            y=img_utts["y"].tolist(),
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.8),
            hovertemplate=f"img={i} game=%{{customdata[0]}} rep=%{{customdata[1]}} correct=%{{customdata[2]}}<extra></extra>",
            customdata=img_utts[["game_id", "rep_num", "correct"]].values,
            name=f"image {i}",
            visible=False,
            showlegend=True,
        ))

    for i in range(n_images):
        img_convs = convs[convs["image_id"] == i]
        color = _IMAGE_COLORS[i % len(_IMAGE_COLORS)]
        traces.append(go.Scatter(
            x=img_convs["x"].tolist(),
            y=img_convs["y"].tolist(),
            mode="markers",
            marker=dict(symbol="star", size=10, color=color, line=dict(width=1, color="black")),
            hovertemplate=f"convention img={i} game=%{{customdata[0]}}<extra></extra>",
            customdata=img_convs[["game_id"]].values,
            name=f"conv img {i}",
            visible=False,
            showlegend=False,
        ))

    # Per-game traces (hidden by default)
    for g in range(n_games):
        game_utts = utts[utts["game_id"] == g]
        colors = [_IMAGE_COLORS[i % len(_IMAGE_COLORS)] for i in game_utts["image_id"]]
        rep_widths = [3 if int(r) == 1 else 0 for r in game_utts["rep_num"]]
        traces.append(go.Scatter(
            x=game_utts["x"].tolist(),
            y=game_utts["y"].tolist(),
            mode="markers",
            marker=dict(
                color=colors, size=6, opacity=0.9,
                line=dict(width=rep_widths, color="rgba(0,0,0,0.8)"),
            ),
            hovertemplate=f"game={g} img=%{{customdata[0]}} rep=%{{customdata[1]}} correct=%{{customdata[2]}}<extra></extra>",
            customdata=game_utts[["image_id", "rep_num", "correct"]].values,
            name=f"game {g}",
            visible=False,
            showlegend=False,
        ))

    n_traces = len(traces)

    def _vis(show_img: int | None, show_game: int | None) -> list[bool]:
        v = [False] * n_traces
        v[0] = True  # background
        for j in range(4):
            v[t_legend_start + j] = True  # legend dummies always on
        if show_img is None:
            for i in range(n_images):
                v[t_proto_start + i] = True  # all prototypes
        else:
            v[t_proto_start + show_img] = True  # only selected prototype
            v[t_img_dots_start + show_img] = True
            v[t_img_conv_start + show_img] = True
        if show_game is not None:
            v[t_game_dots_start + show_game] = True
        return v

    game_annotations: dict[int, list[dict]] = {
        g: _arrow_annotations(utts[utts["game_id"] == g]) for g in range(n_games)
    }

    image_buttons: list[dict] = [dict(
        label="— all images —", method="update",
        args=[{"visible": _vis(None, None)}, {"annotations": []}],
    )]
    for i in range(n_images):
        image_buttons.append(dict(
            label=f"image {i}", method="update",
            args=[{"visible": _vis(show_img=i, show_game=None)}, {"annotations": []}],
        ))

    game_buttons: list[dict] = [dict(
        label="— all games —", method="update",
        args=[{"visible": _vis(None, None)}, {"annotations": []}],
    )]
    for g in range(n_games):
        game_buttons.append(dict(
            label=f"game {g}", method="update",
            args=[{"visible": _vis(None, g)}, {"annotations": game_annotations[g]}],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=image_buttons, direction="down", showactive=True,
                x=0.0, xanchor="left", y=1.12, yanchor="bottom",
            ),
            dict(
                buttons=game_buttons, direction="down", showactive=True,
                x=0.18, xanchor="left", y=1.12, yanchor="bottom",
            ),
        ],
        title="Convention trajectories (t-SNE)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=900,
        margin=dict(t=120, b=20, l=20, r=20),
    )
    return fig


def compute_per_game_tsne_coords(
    fit: ConventionFit,
    batch: TrialBatch,
    role: Literal["speaker", "listener"] = "speaker",
    perplexity: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Like compute_tsne_coords, but runs a separate t-SNE per game so each game's
    utterances fill the 2D space independently.

    Each game's t-SNE includes: that game's utterances, that game's convention
    points m_{g,i}, and all global prototypes mu_i.

    Returns a DataFrame with the same columns as compute_tsne_coords.
    Coordinates are not comparable across games (each game has its own 2D space).
    """
    utt_emb = np.array(batch.utterance_emb)
    game_ids = np.array(batch.game_ids)
    image_ids = np.array(batch.image_ids)
    rep_nums = np.array(batch.rep_num)
    correct = np.array(batch.selected_idx) == np.array(batch.target_idx)

    m_loc = fit.speaker_m_loc if role == "speaker" else fit.listener_m_loc
    proto_vecs = fit.mu_i
    n_images = proto_vecs.shape[0]

    rng = np.random.default_rng(seed)
    all_rows: list[pd.DataFrame] = []

    for g in range(batch.n_games):
        if g % 10 == 0:
            _log.info("  per-game t-SNE: %d / %d games", g, batch.n_games)
        mask = game_ids == g
        seen: set = set()
        utt_indices: list[int] = []
        utt_meta: list[dict] = []
        for n in np.where(mask)[0]:
            key = (int(image_ids[n]), int(rep_nums[n]))
            if key not in seen:
                seen.add(key)
                utt_indices.append(n)
                utt_meta.append({
                    "kind": "utterance",
                    "game_id": g,
                    "image_id": int(image_ids[n]),
                    "rep_num": int(rep_nums[n]),
                    "correct": bool(correct[n]),
                })

        conv_meta = [
            {"kind": "convention", "game_id": g, "image_id": i, "rep_num": 0, "correct": False}
            for i in range(n_images)
        ]
        proto_meta = [
            {"kind": "prototype", "game_id": g, "image_id": i, "rep_num": 0, "correct": False}
            for i in range(n_images)
        ]

        utt_vecs = utt_emb[utt_indices]
        conv_vecs = m_loc[g]
        all_vecs = np.vstack([utt_vecs, conv_vecs, proto_vecs]).astype(np.float64)

        n_pts = len(all_vecs)
        p = min(perplexity, max(2, (n_pts - 1) // 3))
        game_seed = int(rng.integers(0, 2**31))
        coords_2d = TSNE(
            n_components=2, perplexity=p, random_state=game_seed, n_jobs=1
        ).fit_transform(all_vecs)

        meta = utt_meta + conv_meta + proto_meta
        df_g = pd.DataFrame(meta)
        df_g["x"] = coords_2d[:, 0].astype(np.float32)
        df_g["y"] = coords_2d[:, 1].astype(np.float32)
        all_rows.append(df_g)

    _log.info("  per-game t-SNE: done (%d games)", batch.n_games)
    return pd.concat(all_rows, ignore_index=True)


def plot_per_game_overview(tsne_df: pd.DataFrame) -> go.Figure:
    """
    Interactive per-game t-SNE view with a game dropdown.

    Each game's t-SNE was computed independently so its utterances fill the
    full 2D space. A dropdown selects which game to display; one game is shown
    at a time so the coordinates are always meaningful.
    Use compute_per_game_tsne_coords as input.
    """
    n_games = int(tsne_df[tsne_df["kind"] == "utterance"]["game_id"].max()) + 1
    n_images = int(tsne_df["image_id"].max()) + 1

    # Legend dummies — always visible, one per symbol type
    traces: list[go.BaseTraceType] = [
        go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(symbol="diamond", size=12, color="grey", line=dict(width=2, color="black")),
            name="prototype μᵢ (◆)", showlegend=True),
        go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(symbol="star", size=12, color="grey", line=dict(width=1, color="black")),
            name="convention m(g,i) (★)", showlegend=True),
        go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(symbol="circle", size=9, color="grey"),
            name="correct (○)", showlegend=True),
        go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(symbol="x", size=9, color="grey"),
            name="incorrect (✗)", showlegend=True),
    ]
    N_LEGEND = len(traces)
    # Per-game block: n_images utterance traces + n_images convention traces + 1 prototype trace
    TRACES_PER_GAME = n_images + n_images + 1

    game_annotations: dict[int, list[dict]] = {}

    for g in range(n_games):
        gdf = tsne_df[tsne_df["game_id"] == g]
        utts = gdf[gdf["kind"] == "utterance"]
        convs = gdf[gdf["kind"] == "convention"]
        protos = gdf[gdf["kind"] == "prototype"]
        visible = g == 0

        for i in range(n_images):
            color = _IMAGE_COLORS[i % len(_IMAGE_COLORS)]
            img_utts = utts[utts["image_id"] == i].sort_values("rep_num")
            rep_nums = img_utts["rep_num"].tolist()
            traces.append(go.Scatter(
                x=img_utts["x"].tolist(),
                y=img_utts["y"].tolist(),
                mode="markers",
                marker=dict(
                    color=color, size=6, opacity=0.9,
                    symbol=img_utts["correct"].map(lambda c: "circle" if c else "x").tolist(),
                    line=dict(width=[3 if r == 1 else 0 for r in rep_nums], color="rgba(0,0,0,0.8)"),
                ),
                hovertemplate=f"img={i} rep=%{{customdata[0]}} correct=%{{customdata[1]}}<extra></extra>",
                customdata=img_utts[["rep_num", "correct"]].values,
                name=f"image {i}",
                showlegend=(g == 0),
                legendgroup=f"image {i}",
                visible=visible,
            ))

        for i in range(n_images):
            gc = convs[convs["image_id"] == i]
            color = _IMAGE_COLORS[i % len(_IMAGE_COLORS)]
            traces.append(go.Scatter(
                x=gc["x"].tolist(),
                y=gc["y"].tolist(),
                mode="markers",
                marker=dict(symbol="star", size=12, color=color, line=dict(width=1, color="black")),
                name=f"conv {i}",
                showlegend=False,
                legendgroup=f"image {i}",
                visible=visible,
            ))

        proto_colors = [_IMAGE_COLORS[int(i) % len(_IMAGE_COLORS)] for i in protos["image_id"]]
        traces.append(go.Scatter(
            x=protos["x"].tolist(),
            y=protos["y"].tolist(),
            mode="markers",
            marker=dict(symbol="diamond", size=12, color=proto_colors, line=dict(width=1.5, color="black")),
            name="prototypes",
            showlegend=False,
            visible=visible,
        ))

        game_annotations[g] = _arrow_annotations(utts)

    n_traces = len(traces)

    def _vis(show_game: int) -> list[bool]:
        v = [False] * n_traces
        for j in range(N_LEGEND):
            v[j] = True
        start = N_LEGEND + show_game * TRACES_PER_GAME
        for j in range(TRACES_PER_GAME):
            v[start + j] = True
        return v

    game_buttons = [
        dict(
            label=f"game {g}", method="update",
            args=[
                {"visible": _vis(g)},
                {"annotations": game_annotations[g],
                 "title": f"Per-game convention trajectories (t-SNE): game {g}"},
            ],
        )
        for g in range(n_games)
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(
            buttons=game_buttons, direction="down", showactive=True,
            x=0.0, xanchor="left", y=1.12, yanchor="bottom",
        )],
        title="Per-game convention trajectories (t-SNE): game 0",
        annotations=game_annotations[0],
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1100,
        height=850,
        margin=dict(t=120, b=20, l=20, r=20),
    )
    return fig


def plot_game(tsne_df: pd.DataFrame, game_id: int) -> go.Figure:
    """
    Detailed trajectory plot for a single game.

    Shows all other utterances as grey background. For the selected game:
    utterances are colored by image_id, with arrows connecting consecutive
    reps for each image. Convention points m_{g,i} are shown as stars.
    Prototype mu_i are shown as diamonds. Correct selections are circles;
    incorrect are x markers.

    Call compute_tsne_coords first and reuse its output across multiple
    plot_game calls (one per game of interest).
    """
    utts = tsne_df[tsne_df["kind"] == "utterance"]
    convs = tsne_df[tsne_df["kind"] == "convention"]
    protos = tsne_df[tsne_df["kind"] == "prototype"]

    game_utts = utts[utts["game_id"] == game_id]
    game_convs = convs[convs["game_id"] == game_id]
    other_utts = utts[utts["game_id"] != game_id]

    n_images = int(utts["image_id"].max()) + 1
    traces: list[go.BaseTraceType] = []

    # Grey background
    traces.append(go.Scatter(
        x=other_utts["x"].tolist(),
        y=other_utts["y"].tolist(),
        mode="markers",
        marker=dict(color="lightgrey", size=3, opacity=0.4),
        hoverinfo="skip",
        name="other games",
        showlegend=False,
    ))

    # Prototypes
    traces.append(go.Scatter(
        x=protos["x"].tolist(),
        y=protos["y"].tolist(),
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=14,
            color=[_IMAGE_COLORS[i % len(_IMAGE_COLORS)] for i in protos["image_id"]],
            line=dict(width=2, color="black"),
        ),
        hovertemplate="prototype img=%{customdata[0]}<extra></extra>",
        customdata=protos[["image_id"]].values,
        name="prototype μᵢ (◆)",
        showlegend=True,
    ))

    # Legend entries explaining symbol encoding
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="star", size=12, color="grey", line=dict(width=1, color="black")),
        name="convention m(g,i) (★)",
        showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=9, color="grey"),
        name="correct (○)",
        showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="x", size=9, color="grey"),
        name="incorrect (✗)",
        showlegend=True,
    ))

    # One trace per image: dots (arrows added as annotations below)
    for i in range(n_images):
        img_utts = game_utts[game_utts["image_id"] == i].sort_values("rep_num")
        color = _IMAGE_COLORS[i % len(_IMAGE_COLORS)]

        # Utterance dots — rep 1 gets a halo ring
        rep_nums = img_utts["rep_num"].tolist()
        traces.append(go.Scatter(
            x=img_utts["x"].tolist(),
            y=img_utts["y"].tolist(),
            mode="markers",
            marker=dict(
                color=color,
                size=[5 + r for r in rep_nums],
                symbol=img_utts["correct"].map(lambda c: "circle" if c else "x").tolist(),
                line=dict(
                    width=[3 if r == 1 else 0 for r in rep_nums],
                    color="rgba(0,0,0,0.8)",
                ),
            ),
            hovertemplate=f"img={i} rep=%{{customdata[0]}} correct=%{{customdata[1]}}<extra></extra>",
            customdata=img_utts[["rep_num", "correct"]].values,
            name=f"image {i}",
            showlegend=True,
        ))

        # Convention point m_{g,i}
        gc = game_convs[game_convs["image_id"] == i]
        if len(gc) > 0:
            traces.append(go.Scatter(
                x=gc["x"].tolist(),
                y=gc["y"].tolist(),
                mode="markers",
                marker=dict(symbol="star", size=14, color=color, line=dict(width=1.5, color="black")),
                hovertemplate=f"convention m(g={game_id}, img={i})<extra></extra>",
                name=f"m(g,{i})",
                showlegend=False,
            ))

    fig = go.Figure(data=traces)

    # Add rep-to-rep arrows as annotations
    for ann in _arrow_annotations(game_utts):
        fig.add_annotation(**ann)

    fig.update_layout(
        title=f"Game {game_id} — convention trajectories (t-SNE)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1100,
        height=850,
        margin=dict(t=60, b=20, l=20, r=20),
        legend=dict(title="image"),
    )
    return fig


_TAB_CSS = """
body { font-family: sans-serif; margin: 0; padding: 0; }
.tab-bar { display: flex; border-bottom: 2px solid #ccc; padding: 0 12px; background: #f8f8f8; }
.tab-btn {
    padding: 10px 20px; cursor: pointer; border: 1px solid transparent;
    border-bottom: none; margin-bottom: -2px; background: transparent;
    font-size: 14px; color: #555;
}
.tab-btn.active { background: white; border-color: #ccc; color: #000; border-radius: 4px 4px 0 0; }
.tab-pane { display: none; padding: 8px; }
.tab-pane.active { display: block; }
"""

_TAB_JS = """
function showTab(btn, name) {
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(name).classList.add('active');
    btn.classList.add('active');
}
"""


def combined_html(overview_fig: go.Figure, per_game_fig: go.Figure) -> str:
    """
    Combine the global overview and per-game overview figures into a single
    self-contained HTML file with two tabs.

    Plotly JS is embedded (no CDN dependency), so the file works offline.
    """
    overview_div = overview_fig.to_html(full_html=False, include_plotlyjs="inline")
    per_game_div = per_game_fig.to_html(full_html=False, include_plotlyjs=False)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Convention trajectories</title>
  <style>{_TAB_CSS}</style>
</head>
<body>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="showTab(this, 'tab-overview')">Overview</button>
    <button class="tab-btn" onclick="showTab(this, 'tab-per-game')">Per-game</button>
  </div>
  <div id="tab-overview" class="tab-pane active">{overview_div}</div>
  <div id="tab-per-game" class="tab-pane">{per_game_div}</div>
  <script>{_TAB_JS}</script>
</body>
</html>"""
