import json
import os
import subprocess
import sys
import importlib
import re
import zipfile
import urllib.request
import urllib.parse
from pathlib import Path

def _ensure_packages_installed(requirements: list[tuple[str, str]]) -> None:
    for module_name, pip_name in requirements:
        try:
            importlib.import_module(module_name)
            continue
        except ModuleNotFoundError:
            pass

        cmd = [sys.executable, "-m", "pip", "install", "--user", "--disable-pip-version-check", pip_name]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = (
                f"Failed to auto-install required package '{pip_name}' (import name '{module_name}').\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
            raise RuntimeError(msg)
        importlib.invalidate_caches()
        importlib.import_module(module_name)


_ensure_packages_installed(
    [
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
        ("rastermap", "rastermap"),
    ]
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture


def _safe_zscore_rows(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    z = (x - mean) / std
    return np.nan_to_num(z)


def _nearest_index(time_s: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(time_s - float(t))))


def _uniform_time_indices(n_time: int, max_time: int) -> np.ndarray:
    n_time = int(n_time)
    max_time = int(max_time)
    if max_time <= 0 or n_time <= max_time:
        return np.arange(n_time, dtype=int)
    return np.unique(np.linspace(0, n_time - 1, max_time, dtype=int))


def _compute_raster_order(x_cells_time: np.ndarray, random_state: int = 0) -> tuple[np.ndarray, str]:
    from rastermap import Rastermap  # type: ignore

    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5)
    model.fit(x_cells_time)
    order = getattr(model, "isort", None)
    if order is None:
        order = np.argsort(np.asarray(model.embedding).reshape(-1))
    return np.asarray(order, dtype=int), "rastermap"


def save_activity_matrix_rastermap(
    x_cells_time: np.ndarray,
    dt_s: float,
    out_path: Path,
    onset_times_s: list[float],
    random_state: int = 0,
) -> dict:
    x_cells_time = np.asarray(x_cells_time, dtype=np.float32)
    x_z = _safe_zscore_rows(x_cells_time)
    order, method = _compute_raster_order(x_z, random_state=int(random_state))
    x_plot = x_z[order]

    vmin = float(np.nanpercentile(x_plot, 5))
    vmax = float(np.nanpercentile(x_plot, 95))

    time_s = np.arange(x_plot.shape[1], dtype=float) * float(dt_s)
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    ax.imshow(
        x_plot,
        cmap="gray_r",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=[float(time_s[0]), float(time_s[-1]), float(x_plot.shape[0]), 0.0],
    )

    onset_times_s = [float(x) for x in onset_times_s]
    if onset_times_s:
        ax.scatter(
            onset_times_s,
            [-0.02] * len(onset_times_s),
            marker="^",
            s=60,
            color="magenta",
            edgecolors="none",
            linewidths=0.0,
            label="Hunting onset",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=10,
        )
        ax.legend(frameon=False, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Astroglia (sorted)")
    ax.set_title(f"Activity matrix rastermap ({method})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {
        "method": method,
        "n_cells": int(x_cells_time.shape[0]),
        "n_time": int(x_cells_time.shape[1]),
        "dt_s": float(dt_s),
    }


def extract_centered_windows(trace_1d: np.ndarray, time_s: np.ndarray, centers_s, window_s: float) -> np.ndarray:
    time_s = np.asarray(time_s).astype(float)
    trace_1d = np.asarray(trace_1d).astype(float)
    centers_s = [float(c) for c in centers_s]
    if time_s.size < 3:
        return np.zeros((0, 0), dtype=float)

    dt = float(np.median(np.diff(time_s)))
    half_n = int(np.round(float(window_s) / dt))
    win_len = 2 * half_n + 1
    if win_len <= 1:
        return np.zeros((0, 0), dtype=float)

    windows = []
    for c in centers_s:
        idx = _nearest_index(time_s, c)
        start = idx - half_n
        end = idx + half_n
        if start < 0 or end >= trace_1d.size:
            continue
        windows.append(trace_1d[start : end + 1])
    if not windows:
        return np.zeros((0, win_len), dtype=float)
    return np.vstack(windows)


def benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = int(pvals.size)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    thresholds = float(alpha) * (np.arange(1, m + 1) / m)
    is_below = p_sorted <= thresholds
    if not np.any(is_below):
        mask_sorted = np.zeros(m, dtype=bool)
    else:
        k = int(np.max(np.where(is_below)[0]))
        mask_sorted = np.zeros(m, dtype=bool)
        mask_sorted[: k + 1] = True
    mask = np.zeros(m, dtype=bool)
    mask[order] = mask_sorted
    return mask


def detect_significant_points_shuffle_exact(
    data: np.ndarray,
    alpha: float = 0.05,
    two_tailed: bool = True,
    apply_fdr: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    n, t = data.shape
    significance = np.zeros((n, t), dtype=int)
    p_values = np.zeros((n, t), dtype=float)

    for i in range(n):
        x = data[i]
        xs = np.sort(x)
        m = float(xs.size)
        left = np.searchsorted(xs, x, side="left").astype(float)
        right = np.searchsorted(xs, x, side="right").astype(float)
        percentile = (left + 0.5 * (right - left)) / m

        if two_tailed:
            p = 2.0 * np.minimum(percentile, 1.0 - percentile)
            p_values[i] = p
            q_low = float(np.quantile(xs, float(alpha) / 2.0))
            q_high = float(np.quantile(xs, 1.0 - float(alpha) / 2.0))
            sig = np.zeros(t, dtype=int)
            hit = p < float(alpha)
            sig[hit & (x >= q_high)] = 1
            sig[hit & (x <= q_low)] = -1
            significance[i] = sig
        else:
            p_high = (m - right) / m
            p_low = left / m
            p = np.minimum(p_high, p_low)
            p_values[i] = p
            sig = np.zeros(t, dtype=int)
            hit = p < float(alpha)
            sig[hit & (x >= float(np.quantile(xs, 1.0 - float(alpha))))] = 1
            sig[hit & (x <= float(np.quantile(xs, float(alpha))))] = -1
            significance[i] = sig

        if apply_fdr:
            mask = benjamini_hochberg(p_values[i], float(alpha))
            med = float(np.median(xs))
            row = np.zeros(t, dtype=int)
            row[mask & (x > med)] = 1
            row[mask & (x < med)] = -1
            significance[i] = row

    return significance, p_values


def _find_nearest_indices(time_stamps: np.ndarray, start: float, end: float) -> tuple[int, int]:
    time_stamps = np.asarray(time_stamps, dtype=float)
    start_idx = int(np.searchsorted(time_stamps, float(start), side="left"))
    end_idx = int(np.searchsorted(time_stamps, float(end), side="left")) - 1
    return start_idx, end_idx


def _gmm_intersection_threshold_1d(values: np.ndarray, random_seed: int = 0) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return float(np.quantile(v, 0.25)) if v.size > 0 else 0.0

    eps = np.finfo(float).eps
    x = np.log10(v + eps).reshape(-1, 1)
    if np.unique(x).size < 10:
        return float(np.quantile(v, 0.25))

    gmm = GaussianMixture(n_components=2, random_state=int(random_seed))
    gmm.fit(x)
    weights = gmm.weights_.reshape(-1)
    means = gmm.means_.reshape(-1)
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    order = np.argsort(means)
    w1, w2 = float(weights[order][0]), float(weights[order][1])
    m1, m2 = float(means[order][0]), float(means[order][1])
    s1, s2 = float(stds[order][0]), float(stds[order][1])

    if s1 <= 0 or s2 <= 0 or not np.isfinite(s1) or not np.isfinite(s2):
        return float(10 ** ((m1 + m2) / 2.0))

    a = (1.0 / (2.0 * s2 * s2)) - (1.0 / (2.0 * s1 * s1))
    b = (m1 / (s1 * s1)) - (m2 / (s2 * s2))
    c = (m2 * m2) / (2.0 * s2 * s2) - (m1 * m1) / (2.0 * s1 * s1) + np.log((w2 / s2) / (w1 / s1))

    roots = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots = [float(-c / b)]
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            r1 = float((-b + np.sqrt(disc)) / (2.0 * a))
            r2 = float((-b - np.sqrt(disc)) / (2.0 * a))
            roots = [r1, r2]

    if roots:
        lo = min(m1, m2)
        hi = max(m1, m2)
        between = [r for r in roots if lo <= r <= hi]
        x_star = between[0] if between else min(roots, key=lambda r: abs(r - (m1 + m2) / 2.0))
    else:
        x_star = (m1 + m2) / 2.0

    return float(10 ** x_star)


def find_hunting_related_component(
    H_by_set: dict,
    time_s: np.ndarray,
    hunting_onsets_by_set: dict,
    window_s: float = 10.0,
    n_shuffles: int = 500,
    random_seed: int = 42,
) -> dict:
    rng = np.random.default_rng(int(random_seed))
    set_names = list(H_by_set.keys())
    n_components = int(next(iter(H_by_set.values())).shape[0])

    dt = float(np.median(np.diff(time_s)))
    half_n = int(np.round(float(window_s) / dt))
    center_slice = slice(max(0, half_n - 2), half_n + 3)

    zscores = np.full(n_components, np.nan, dtype=float)
    pvals = np.full(n_components, np.nan, dtype=float)
    obs_stats = np.full(n_components, np.nan, dtype=float)

    for k in range(n_components):
        observed_windows = []
        for s in set_names:
            centers = hunting_onsets_by_set.get(s, [])
            if not centers:
                continue
            windows = extract_centered_windows(H_by_set[s][k], time_s, centers, window_s=window_s)
            if windows.size == 0:
                continue
            observed_windows.append(windows)

        if not observed_windows:
            continue

        observed_windows = np.vstack(observed_windows)
        obs_avg = np.nanmean(observed_windows, axis=0)
        obs_stat = float(np.nanmean(obs_avg[center_slice]))
        obs_stats[k] = obs_stat

        null_stats = []
        for _ in range(int(n_shuffles)):
            shuffled_windows = []
            for s in set_names:
                centers = hunting_onsets_by_set.get(s, [])
                if not centers:
                    continue
                trace = H_by_set[s][k]
                shift = int(rng.integers(0, trace.size))
                trace_shifted = np.roll(trace, shift)
                windows = extract_centered_windows(trace_shifted, time_s, centers, window_s=window_s)
                if windows.size == 0:
                    continue
                shuffled_windows.append(windows)
            if not shuffled_windows:
                continue
            shuffled_windows = np.vstack(shuffled_windows)
            null_avg = np.nanmean(shuffled_windows, axis=0)
            null_stats.append(float(np.nanmean(null_avg[center_slice])))

        null_stats = np.asarray(null_stats, dtype=float)
        if null_stats.size < 10:
            continue

        p = float((1.0 + np.sum(null_stats >= obs_stat)) / (1.0 + null_stats.size))
        mu = float(np.mean(null_stats))
        sigma = float(np.std(null_stats)) if float(np.std(null_stats)) > 0 else np.nan
        z = (obs_stat - mu) / sigma if np.isfinite(sigma) else np.nan

        pvals[k] = p
        zscores[k] = z

    best_idx = int(np.nanargmax(zscores))
    return {
        "best_component_index": best_idx,
        "best_zscore": float(zscores[best_idx]),
        "best_pvalue": float(pvals[best_idx]),
        "zscores": zscores,
        "pvalues": pvals,
        "obs_stats": obs_stats,
    }


def compute_energy_ratio(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    W_scaled = W * norms.T
    energies = np.abs(W_scaled)
    total = np.sum(energies, axis=1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return energies / total


def gmm_select_high_component(values_1d: np.ndarray, random_seed: int = 0) -> dict:
    x = np.asarray(values_1d, dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=int(random_seed))
    gmm.fit(x)
    labels = gmm.predict(x)
    means = gmm.means_.reshape(-1)
    high_label = int(np.argmax(means))
    selected = labels == high_label
    return {
        "selected_mask": selected,
        "labels": labels,
        "means": means,
        "weights": gmm.weights_.reshape(-1),
        "covariances": gmm.covariances_.reshape(-1),
    }


def find_hunting_tp_by_correlation(
    H_by_set: dict,
    time_s: np.ndarray,
    convergence_periods: dict,
    fish_name: str,
    set_order: list[str],
) -> dict:
    time_s = np.asarray(time_s, dtype=float)
    n_components = int(next(iter(H_by_set.values())).shape[0])

    binary_by_set = {}
    for s in set_order:
        b = np.zeros(time_s.size, dtype=float)
        for start_frame, end_frame in convergence_periods.get(fish_name, {}).get(s, []):
            t0 = float(start_frame) / 300.0
            t1 = float(end_frame) / 300.0
            i0 = int(np.searchsorted(time_s, t0, side="left"))
            i1 = int(np.searchsorted(time_s, t1, side="right"))
            i0 = max(0, min(i0, time_s.size - 1))
            i1 = max(0, min(i1, time_s.size))
            b[i0:i1] = 1.0
        binary_by_set[s] = b

    binary = np.concatenate([binary_by_set[s] for s in set_order], axis=0)
    H = np.concatenate([H_by_set[s] for s in set_order], axis=1)

    b = (binary - binary.mean()) / (binary.std() if binary.std() > 0 else 1.0)
    H_z = (H - H.mean(axis=1, keepdims=True)) / H.std(axis=1, keepdims=True)
    H_z = np.nan_to_num(H_z)
    cor = (H_z * b).mean(axis=1)

    best_idx = int(np.argmax(cor))
    top10 = np.argsort(cor)[::-1][: min(10, n_components)]
    return {
        "best_component_index": best_idx,
        "correlations": cor,
        "top10": [(int(i), float(cor[i])) for i in top10],
    }


def find_hunting_tp_by_shuffle_significance(
    *,
    H_by_set: dict,
    time_stamps: np.ndarray,
    convergence_periods: dict,
    fish_name: str,
    set_order: list[str],
    window_size_s: float,
    alpha: float,
    apply_fdr: bool,
    gmm_seed: int = 0,
) -> dict:
    H_concatenated = np.concatenate([H_by_set[s] for s in set_order], axis=1)
    t1 = int(H_by_set[set_order[0]].shape[1])
    t2 = int(H_by_set[set_order[1]].shape[1])
    t3 = int(H_by_set[set_order[2]].shape[1])
    offsets = {
        set_order[0]: 0,
        set_order[1]: t1,
        set_order[2]: t1 + t2,
        set_order[3]: t1 + t2 + t3,
    }

    significance, p_values = detect_significant_points_shuffle_exact(
        H_concatenated, alpha=float(alpha), two_tailed=True, apply_fdr=bool(apply_fdr)
    )

    event_mask = np.zeros(H_concatenated.shape[1], dtype=bool)
    for set_name in set_order:
        current = convergence_periods.get(fish_name, {}).get(set_name, [])
        for start_frame, _end_frame in current:
            onset_s = float(start_frame) / 300.0
            start = onset_s - float(window_size_s)
            end = onset_s + float(window_size_s)
            s_idx, e_idx = _find_nearest_indices(time_stamps, start, end)
            s_idx = max(0, s_idx)
            e_idx = min(int(time_stamps.shape[0]) - 1, e_idx)
            off = int(offsets[set_name])
            event_mask[off + s_idx : off + e_idx + 1] = True

    bg_mask = ~event_mask
    n = int(H_concatenated.shape[0])
    if np.count_nonzero(bg_mask) > 1:
        std_bg = H_concatenated[:, bg_mask].std(axis=1, ddof=1)
    else:
        std_bg = np.zeros(n, dtype=float)
    if np.count_nonzero(event_mask) > 0:
        amp_avg = np.mean(np.abs(H_concatenated[:, event_mask]), axis=1)
    else:
        amp_avg = np.zeros(n, dtype=float)
    ratios = std_bg / np.maximum(amp_avg, 1e-12)
    ratio_threshold = _gmm_intersection_threshold_1d(ratios[np.isfinite(ratios)], random_seed=int(gmm_seed))

    low_var_flags = ratios <= float(ratio_threshold)
    sig_in_event = (significance == 1) & event_mask
    has_sig_flags = sig_in_event.any(axis=1)
    sig_counts = sig_in_event.sum(axis=1).astype(float)

    flagged = low_var_flags & has_sig_flags
    candidates = np.where(flagged)[0]
    if candidates.size == 0:
        candidates = np.where(has_sig_flags)[0]
    if candidates.size == 0:
        candidates = np.arange(n, dtype=int)

    scores = sig_counts / np.maximum(ratios, 1e-12)
    best_idx = int(candidates[np.argmax(scores[candidates])])
    top10 = np.argsort(scores)[::-1][: min(10, n)]

    return {
        "best_component_index": best_idx,
        "ratio_threshold": float(ratio_threshold),
        "ratios": ratios,
        "low_var_flags": low_var_flags,
        "has_sig_flags": has_sig_flags,
        "flagged_indices": np.where(flagged)[0],
        "scores": scores,
        "top10": [(int(i), float(scores[i])) for i in top10],
        "alpha": float(alpha),
        "apply_fdr": bool(apply_fdr),
        "window_size_s": float(window_size_s),
        "event_mask": event_mask,
        "significance": significance,
        "p_values": p_values,
    }


def _collect_event_centers(
    *,
    fish_name: str,
    set_name: str,
    time_s: np.ndarray,
    convergence_periods: dict,
    passivity_periods: dict,
    bouts_info: dict,
    stimulus_periods: list,
    stimulus_labels: list,
    window_s: float,
) -> dict:
    centers = {"hunting": [], "passivity": [], "locomotion": [], "loom": [], "struggle": []}

    conv_frames = convergence_periods.get(fish_name, {}).get(set_name, [])
    for start_frame, _end_frame in conv_frames:
        centers["hunting"].append(float(start_frame) / 300.0)

    pass_list = passivity_periods.get(fish_name, {}).get(set_name, [])
    for start_s, _end_s in pass_list:
        centers["passivity"].append(float(start_s))

    prey_periods = [p for p, s in zip(stimulus_periods, stimulus_labels) if str(s).strip().lower() == "prey"]
    omr_periods = [p for p, s in zip(stimulus_periods, stimulus_labels) if str(s).strip().lower() == "omr"]
    loom_periods = [p for p, s in zip(stimulus_periods, stimulus_labels) if str(s).strip().lower() == "loom"]

    set_payload = bouts_info.get(fish_name, {}).get(set_name, {})
    bout_frames = np.asarray(set_payload.get("bout", []), dtype=float)
    bout_labels = [str(x).strip().lower() for x in set_payload.get("behaviour", [])]

    escape_periods = []
    struggle_periods = []
    if bout_frames.size > 0:
        bout_sec = bout_frames / 300.0
        for (b0, b1), lbl in zip(bout_sec, bout_labels):
            if lbl == "escape":
                escape_periods.append([float(b0), float(b1)])
            if "struggle" in lbl:
                struggle_periods.append([float(b0), float(b1)])

    for start_s, end_s in loom_periods:
        loom_start = float(start_s)
        loom_end = float(end_s)
        overlaps = False
        for esc0, esc1 in escape_periods:
            if max(loom_start, esc0) < min(loom_end, esc1):
                overlaps = True
                break
        if not overlaps:
            for st0, st1 in struggle_periods:
                if max(loom_start, st0) < min(loom_end, st1):
                    overlaps = True
                    break
        if overlaps:
            continue
        centers["loom"].append(loom_start)

    for st0, _st1 in struggle_periods:
        if st0 - window_s < float(time_s[0]) or st0 + window_s > float(time_s[-1]):
            continue
        centers["struggle"].append(float(st0))

    if bout_frames.size > 0 and omr_periods:
        bout_sec = bout_frames / 300.0
        omr_swim_bouts = []
        for (b0, b1), lbl in zip(bout_sec, bout_labels):
            is_omr_swim = ("turn left" in lbl) or ("turn right" in lbl) or ("slow swim" in lbl)
            if not is_omr_swim:
                continue
            for omr0, omr1 in omr_periods:
                if float(omr0) <= float(b0) <= float(omr1):
                    omr_swim_bouts.append([float(b0), float(b1)])
                    break

        omr_swim_bouts.sort(key=lambda v: v[0])
        cluster_start = None
        cluster_end = None
        for b0, b1 in omr_swim_bouts:
            if cluster_start is None:
                cluster_start = b0
                cluster_end = b1
            elif b0 - cluster_end < float(window_s):
                cluster_end = max(cluster_end, b1)
            else:
                centers["locomotion"].append(float(cluster_start))
                cluster_start = b0
                cluster_end = b1
        if cluster_start is not None:
            centers["locomotion"].append(float(cluster_start))

    return centers


def compute_event_average_trace(
    X: np.ndarray,
    time_s: np.ndarray,
    selected_cells_mask: np.ndarray,
    centers_s,
    window_s: float,
    error_mode: str = "sem",
) -> dict:
    time_s = np.asarray(time_s, dtype=float)
    selected_cells_mask = np.asarray(selected_cells_mask, dtype=bool)
    X_sel = X[selected_cells_mask]
    Xz = _safe_zscore_rows(X_sel)

    if time_s.size < 3:
        return {"mean": np.array([np.nan]), "error": np.array([np.nan]), "time": np.array([0.0]), "n_events": 0}

    dt = float(np.median(np.diff(time_s)))
    half_n = int(np.round(float(window_s) / dt))
    win_len = 2 * half_n + 1
    if win_len <= 1:
        return {"mean": np.array([np.nan]), "error": np.array([np.nan]), "time": np.array([0.0]), "n_events": 0}

    per_event = []
    for c in centers_s:
        idx = _nearest_index(time_s, float(c))
        start = idx - half_n
        end = idx + half_n
        if start < 0 or end >= time_s.size:
            continue
        slice_x = Xz[:, start : end + 1]
        per_event.append(np.nanmean(slice_x, axis=0))

    if not per_event:
        return {
            "mean": np.full(win_len, np.nan),
            "error": np.full(win_len, np.nan),
            "time": np.linspace(-window_s, window_s, win_len),
            "n_events": 0,
        }

    per_event = np.vstack(per_event)
    mean_trace = np.nanmean(per_event, axis=0)
    if error_mode == "std":
        err = np.nanstd(per_event, axis=0)
    else:
        n = np.sum(~np.isnan(per_event), axis=0)
        std = np.nanstd(per_event, axis=0)
        err = np.divide(std, np.sqrt(n), out=np.full_like(std, np.nan), where=n > 0)

    return {
        "mean": mean_trace,
        "error": err,
        "time": np.linspace(-window_s, window_s, mean_trace.size),
        "n_events": int(per_event.shape[0]),
    }


def main():
    project_dir = Path(__file__).resolve().parent
    if str(os.environ.get("TEST_DROPBOX_DOWNLOAD", "0")).strip() == "1":
        test_dropbox_demo_data_download(DROPBOX_DEMO_FOLDER_URL)
        return

    fish_name = str(os.environ.get("FISH_NAME", "20250807-F2")).strip()
    data_dir = project_dir / f"demo_data_F1_{fish_name}"
    out_dir = project_dir / f"demo_outputs_{fish_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        download_dropbox_demo_data(project_dir, DROPBOX_DEMO_FOLDER_URL)

    set_order = [
        "prey-loom-omr-set001",
        "prey-loom-omr-set002",
        "prey-loom-omr-set003",
        "prey-loom-omr-set004",
    ]

    prefix = f"F1_demo_{fish_name}_"
    time_s = np.load(data_dir / f"{prefix}t.npy")
    X_by_set = {
        s: np.load(data_dir / f"{prefix}normalized_masked_dfof_{s}.npy") for s in set_order
    }

    bouts_info = json.loads((data_dir / f"{prefix}bouts_info.json").read_text(encoding="utf-8"))
    passivity = json.loads((data_dir / f"{prefix}events_passivity.json").read_text(encoding="utf-8"))
    convergence = json.loads(
        (data_dir / f"{prefix}events_convergence_periods.json").read_text(encoding="utf-8")
    )
    stim = json.loads((data_dir / f"{prefix}stimulus_periods.json").read_text(encoding="utf-8"))
    stimulus_periods = stim["periods"]
    stimulus_labels = stim["stimuli"]

    X_full = np.concatenate([X_by_set[s] for s in set_order], axis=1)
    dt_s = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    t_lengths_x = {s: int(X_by_set[s].shape[1]) for s in set_order}
    offsets_x = {}
    acc_x = 0
    for s in set_order:
        offsets_x[s] = acc_x
        acc_x += t_lengths_x[s]
    onset_times_s = []
    for s in set_order:
        for start_frame, _end_frame in convergence.get(fish_name, {}).get(s, []):
            onset_s = float(start_frame) / 300.0
            idx_set = _nearest_index(time_s, onset_s)
            idx = int(offsets_x[s]) + int(idx_set)
            if 0 <= idx < int(X_full.shape[1]):
                onset_times_s.append(float(idx) * dt_s)
    onset_times_s = sorted(set(onset_times_s))
    rastermap_info = save_activity_matrix_rastermap(
        X_full,
        dt_s=dt_s,
        out_path=out_dir / "activity_matrix_rastermap.png",
        onset_times_s=onset_times_s,
        random_state=0,
    )

    n_components = 90
    recompute_nmf = str(os.environ.get("RECOMPUTE_NMF", "0")).strip() == "1"
    if recompute_nmf:
        nmf = NMF(n_components=n_components, init="random", random_state=42, max_iter=2000)
        W = nmf.fit_transform(X_full)
        H = nmf.components_
        H_splits = np.split(H, 4, axis=1)
        H_by_set = {s: H_splits[i] for i, s in enumerate(set_order)}
    else:
        W = np.load(data_dir / f"{prefix}Spatial_components_full_sets_90.npy")
        H_by_set = {
            s: np.load(data_dir / f"{prefix}Temporal_components_full_sets_{s}_90.npy") for s in set_order
        }
        H = np.concatenate([H_by_set[s] for s in set_order], axis=1)

    shuffle_alpha = float(os.environ.get("SHUFFLE_ALPHA", "0.05"))
    shuffle_apply_fdr = str(os.environ.get("SHUFFLE_APPLY_FDR", "0")).strip() == "1"
    tp_window_s = float(os.environ.get("TP_WINDOW_S", "10"))
    hunting_tp = find_hunting_tp_by_shuffle_significance(
        H_by_set=H_by_set,
        time_stamps=time_s,
        convergence_periods=convergence,
        fish_name=fish_name,
        set_order=set_order,
        window_size_s=tp_window_s,
        alpha=shuffle_alpha,
        apply_fdr=shuffle_apply_fdr,
        gmm_seed=0,
    )
    tp_idx = int(hunting_tp["best_component_index"])

    fe = compute_energy_ratio(W, H)
    energy_ratio = fe[:, tp_idx]

    gmm_out = gmm_select_high_component(energy_ratio, random_seed=0)
    selective_mask = gmm_out["selected_mask"]

    np.save(out_dir / "energy_ratio.npy", energy_ratio)
    np.save(out_dir / "gmm_selected_mask.npy", selective_mask.astype(bool))

    plt.figure(figsize=(10, 3))
    plt.hist(energy_ratio, bins=100, color="lightgray", density=False)
    plt.title(f"{fish_name} energy ratio distribution (GMM 2-comp)")
    plt.xlabel("Energy ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "energy_ratio_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.hist(hunting_tp["ratios"], bins=60, color="lightgray", density=False)
    plt.axvline(hunting_tp["ratio_threshold"], color="red", lw=2)
    plt.title(f"{fish_name} hunting ratio threshold (GMM intersection)")
    plt.xlabel("ratio = std(background) / mean(|event|)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "tp_low_variability_ratio_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 3))
    time_concat_s = np.arange(H.shape[1]) * dt_s
    plt.plot(time_concat_s, H[tp_idx], color="black", lw=2)
    t_lengths = {s: int(H_by_set[s].shape[1]) for s in set_order}
    offsets = {}
    acc = 0
    for s in set_order:
        offsets[s] = acc
        acc += t_lengths[s]
    onset_indices = []
    for s in set_order:
        for start_frame, _end_frame in convergence.get(fish_name, {}).get(s, []):
            onset_s = float(start_frame) / 300.0
            idx_set = _nearest_index(time_s, onset_s)
            idx = int(offsets[s]) + int(idx_set)
            if 0 <= idx < int(H.shape[1]):
                onset_indices.append(idx)
    onset_indices = sorted(set(onset_indices))
    if onset_indices:
        onset_times_s = [time_concat_s[i] for i in onset_indices]
        ax = plt.gca()
        ax.scatter(
            onset_times_s,
            [-0.06] * len(onset_indices),
            marker="^",
            s=90,
            color="magenta",
            edgecolors="none",
            linewidths=0.0,
            label="Hunting onset",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=10,
        )
        ax.legend(frameon=False, loc="upper right")
    plt.title(f"{fish_name} hunting-related temporal component")
    plt.xlabel("Time (s)")
    plt.ylabel("Component amplitude")
    plt.tight_layout()
    plt.savefig(out_dir / "hunting_tp_trace.png", dpi=200)
    plt.close()

    window_s = 30.0
    trace_color = "#4AC9FF"
    panels = ["hunting", "passivity", "locomotion", "loom", "struggle"]
    titles = ["Hunting", "Passivity", "Locomotion", "Loom sensory", "Struggle"]

    traces = {k: [] for k in panels}
    errors = {k: [] for k in panels}
    ns = {k: 0 for k in panels}

    for s in set_order:
        centers = _collect_event_centers(
            fish_name=fish_name,
            set_name=s,
            time_s=time_s,
            convergence_periods=convergence,
            passivity_periods=passivity,
            bouts_info=bouts_info,
            stimulus_periods=stimulus_periods,
            stimulus_labels=stimulus_labels,
            window_s=window_s,
        )
        for k in panels:
            res = compute_event_average_trace(
                X=X_by_set[s],
                time_s=time_s,
                selected_cells_mask=selective_mask,
                centers_s=centers[k],
                window_s=window_s,
                error_mode="sem",
            )
            traces[k].append(res["mean"])
            errors[k].append(res["error"])
            ns[k] += int(res["n_events"])

    for k in panels:
        traces[k] = np.vstack(traces[k])
        errors[k] = np.vstack(errors[k])

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    for ax, k, title in zip(axes, panels, titles):
        mean_trace = np.nanmean(traces[k], axis=0)
        mean_error = np.nanmean(errors[k], axis=0)
        time_axis = np.linspace(-window_s, window_s, mean_trace.size)
        ax.fill_between(time_axis, mean_trace - mean_error, mean_trace + mean_error, color="gray", alpha=0.15)
        ax.plot(time_axis, mean_trace, color=trace_color, lw=3)
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.set_title(f"{title}")
        ax.set_xlim(-window_s, window_s)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("z-scored dF/F")
    plt.tight_layout()
    plt.savefig(out_dir / "event_average_5_conditions.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

