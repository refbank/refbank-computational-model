import math
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from code.models.conventional_speaker import ConventionFit
from code.prep_data.pipeline import TrialBatch

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
                x=0.0, xanchor="left", y=1.02, yanchor="bottom",
            ),
            dict(
                buttons=game_buttons, direction="down", showactive=True,
                x=0.18, xanchor="left", y=1.02, yanchor="bottom",
            ),
        ],
        title="Convention trajectories (t-SNE)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=900,
        margin=dict(t=80, b=20, l=20, r=20),
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
