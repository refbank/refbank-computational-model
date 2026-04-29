from typing import Literal

import numpy as np
import pandas as pd

from code.models.conventional_speaker import ConventionFit
from code.prep_data.pipeline import TrialBatch


def sequential_kalman_means(
    fit: ConventionFit,
    batch: TrialBatch,
    role: Literal["speaker", "listener"],
) -> dict[tuple[int, int], np.ndarray]:
    """
    For each (game_id, image_id) pair, returns array of shape (n_reps, D):
    the posterior mean m after processing each rep in chronological order.

    Uses closed-form Gaussian updates with prior N(mu_i, sigma_game^2 * I)
    and per-trial emission variance from compute_sigma2_t.

    For both roles, s_t is binary correctness (selected_idx == target_idx).
    sigma_min, sigma_max are taken from fit.
    """
    utt = np.array(batch.utterance_emb)
    rep_num = np.array(batch.rep_num)
    game_ids = np.array(batch.game_ids)
    image_ids = np.array(batch.image_ids)
    selected_idx = np.array(batch.selected_idx)
    target_idx = np.array(batch.target_idx)

    # Binary correctness used for both roles (self-contained; no ListenerFit needed)
    s_t = (selected_idx == target_idx).astype(np.float32)

    sigma_min = fit.sigma_min
    sigma_max = fit.sigma_max
    sigma_game_sq = fit.sigma_game ** 2
    eps = 0.01

    result = {}
    for g, i in sorted(set(zip(game_ids.tolist(), image_ids.tolist()))):
        mask = (game_ids == g) & (image_ids == i)
        order = np.argsort(rep_num[mask])
        utts_sorted = utt[mask][order]
        s_sorted = s_t[mask][order]

        mu = fit.mu_i[i].copy().astype(np.float64)
        P = float(sigma_game_sq)
        means = np.empty((len(utts_sorted), utt.shape[1]), dtype=np.float32)

        for t in range(len(utts_sorted)):
            s = float(np.clip(s_sorted[t], eps, 1.0))
            sigma2 = sigma_min + (sigma_max - sigma_min) * (1.0 - s) / s
            sigma2 = max(sigma2, 1e-6)
            K = P / (P + sigma2)
            mu = mu + K * (utts_sorted[t].astype(np.float64) - mu)
            P = P * sigma2 / (P + sigma2)
            means[t] = mu.astype(np.float32)

        result[(g, i)] = means

    return result


def step_sizes_over_reps(
    fit: ConventionFit,
    batch: TrialBatch,
    role: Literal["speaker", "listener"] = "speaker",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: game_id, image_id, rep_num, step_size.
    step_size at rep r = ||E[m]^(r) - E[m]^(r-1)||_2 via sequential Kalman means.
    Pairs with fewer than 2 reps are omitted.
    """
    kalman_means = sequential_kalman_means(fit, batch, role)
    rep_num_arr = np.array(batch.rep_num)
    game_ids_arr = np.array(batch.game_ids)
    image_ids_arr = np.array(batch.image_ids)

    rows = []
    for (g, i), means in kalman_means.items():
        if len(means) < 2:
            continue
        mask = (game_ids_arr == g) & (image_ids_arr == i)
        sorted_reps = np.sort(rep_num_arr[mask])
        for t in range(1, len(means)):
            rows.append({
                "game_id": g,
                "image_id": i,
                "rep_num": int(sorted_reps[t]),
                "step_size": float(np.linalg.norm(means[t] - means[t - 1])),
            })

    return pd.DataFrame(rows, columns=["game_id", "image_id", "rep_num", "step_size"])


def semantic_displacement_after_error(
    fit: ConventionFit,
    batch: TrialBatch,
    success_threshold: float = 0.5,
    role: Literal["speaker", "listener"] = "speaker",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: game_id, image_id, rep_num, displacement, prev_success.
    displacement[t] = ||z(u_t) - z(u_{t-1})||_2 (data-level utterance embedding distance).
    prev_success = s_{t-1} >= success_threshold, where s is binary correctness.
    Pairs with fewer than 2 reps are omitted.
    """
    utt = np.array(batch.utterance_emb)
    rep_num = np.array(batch.rep_num)
    game_ids = np.array(batch.game_ids)
    image_ids = np.array(batch.image_ids)
    selected_idx = np.array(batch.selected_idx)
    target_idx = np.array(batch.target_idx)

    s_t = (selected_idx == target_idx).astype(np.float32)

    rows = []
    for g, i in sorted(set(zip(game_ids.tolist(), image_ids.tolist()))):
        mask = (game_ids == g) & (image_ids == i)
        order = np.argsort(rep_num[mask])
        utts_sorted = utt[mask][order]
        s_sorted = s_t[mask][order]
        reps_sorted = rep_num[mask][order]

        if len(utts_sorted) < 2:
            continue

        for t in range(1, len(utts_sorted)):
            rows.append({
                "game_id": g,
                "image_id": i,
                "rep_num": int(reps_sorted[t]),
                "displacement": float(np.linalg.norm(utts_sorted[t] - utts_sorted[t - 1])),
                "prev_success": bool(s_sorted[t - 1] >= success_threshold),
            })

    return pd.DataFrame(rows, columns=["game_id", "image_id", "rep_num", "displacement", "prev_success"])
