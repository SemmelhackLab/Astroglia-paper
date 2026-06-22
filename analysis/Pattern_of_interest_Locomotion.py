"""
Locomotion-specific significance analysis (shuffle) post-processing.

This script aggregates locomotion candidate patterns (selected elsewhere), computes
event-aligned average traces for each selected temporal pattern, and performs a
simple hierarchical clustering of the resulting average traces.

Data root configuration:
- Environment variable ASTROCYTE_DATA_ROOT (default: D:/2p astrocyte)

Key inputs (relative to ASTROCYTE_DATA_ROOT):
- behaviour/bouts_info.json
- imaging/Registered 10-fish/{fish_name}/t.npy
- imaging/Registered 10-fish/{fish_name}/Temporal_components_full_sets/
    Temporal_components_{set_name}_{component_number}.npy

Key input (analysis output from earlier steps):
- candidates_json_path: locomotion_candidate_patterns.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ASTROCYTE_DATA_ROOT = os.environ.get("ASTROCYTE_DATA_ROOT", r"D:/2p astrocyte")
BEHAVIOUR_DIR = Path(ASTROCYTE_DATA_ROOT) / "behaviour"
IMAGING_DIR = Path(ASTROCYTE_DATA_ROOT) / "imaging"
REGISTERED_FISH_DIR = IMAGING_DIR / "Registered 10-fish"

BOUTS_INFO_PATH = BEHAVIOUR_DIR / "bouts_info.json"

DEFAULT_CANDIDATES_JSON_PATH = (
    IMAGING_DIR
    / "pattern_of_interest_significance_analysis (shuffle)"
    / "summaries"
    / "locomotion_candidate_patterns.json"
)
DEFAULT_EVENT_AVG_OUT_DIR = (
    IMAGING_DIR
    / "pattern_of_interest_significance_analysis (shuffle)"
    / "summaries"
    / "locomotion_event_average_traces"
)
DEFAULT_CLUSTER_OUT_DIR = (
    IMAGING_DIR
    / "pattern_of_interest_significance_analysis (shuffle)"
    / "summaries"
    / "locomotion_clustering"
)

DEFAULT_FPS = 300.0
DEFAULT_MIN_INTERVAL_S = 20.0
DEFAULT_POST_END_S = 1.0
DEFAULT_MIN_GAP_S = 5.0
DEFAULT_RESAMPLE_POINTS = 200


def load_bouts_info(json_path: str | Path = BOUTS_INFO_PATH) -> dict:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def normalize_bout_behavior(raw_behavior: object) -> str:
    if raw_behavior is None:
        return "OMR"
    s = str(raw_behavior).strip().lower()
    if "j-turn right" in s:
        return "J-turn right"
    if "escape" in s:
        return "escape"
    if "struggle" in s:
        return "struggle"
    return "OMR"


def get_locomotion_events_by_set(
    bouts_info: dict,
    fish_name: str,
    *,
    fps: float = DEFAULT_FPS,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
) -> dict[str, list[tuple[float, float, str]]]:
    fish_dict = bouts_info.get(fish_name, {})
    results: dict[str, list[tuple[float, float, str]]] = {}
    for set_name, payload in fish_dict.items():
        bouts = payload.get("bout", [])
        behaviours = payload.get("behaviour", None)
        events: list[tuple[float, float, str]] = []
        for idx, bout in enumerate(bouts):
            if not (isinstance(bout, (list, tuple)) and len(bout) >= 1):
                continue
            start_raw = bout[0]
            end_raw = bout[1] if (isinstance(bout, (list, tuple)) and len(bout) >= 2) else bout[0]
            if start_raw is None or end_raw is None:
                continue
            raw_b = behaviours[idx] if isinstance(behaviours, list) and idx < len(behaviours) else None
            behaviour = normalize_bout_behavior(raw_b)
            events.append((float(start_raw), float(end_raw), behaviour))
        if len(events) == 0:
            continue
        times = [t for s, e, _ in events for t in (s, e)]
        times.sort()
        scale = (1.0 / float(fps)) if float(np.median(times)) > 1000 else 1.0
        events_sec = [(s * scale, e * scale, b) for s, e, b in events]
        events_sec.sort(key=lambda x: x[0])
        filtered: list[tuple[float, float, str]] = []
        last_kept_by_behavior: dict[str, float] = {}
        for start_sec, end_sec, b in events_sec:
            last = last_kept_by_behavior.get(b, None)
            if last is None or (start_sec - last) >= float(min_interval_s):
                filtered.append((float(start_sec), float(end_sec), b))
                last_kept_by_behavior[b] = float(start_sec)
        results[set_name] = filtered
    return results


def find_nearest_indices(time_stamps: np.ndarray, start: float, end: float) -> tuple[int, int]:
    time_stamps = np.asarray(time_stamps, dtype=float)
    start_idx = int(np.searchsorted(time_stamps, float(start), side="left"))
    end_idx = int(np.searchsorted(time_stamps, float(end), side="left")) - 1
    start_idx = max(0, min(start_idx, time_stamps.size - 1))
    end_idx = max(0, min(end_idx, time_stamps.size - 1))
    return start_idx, end_idx


def filter_event_windows_by_min_gap(windows: list[tuple[float, float]], min_gap_s: float) -> list[tuple[float, float]]:
    if len(windows) == 0:
        return []
    windows_sorted = sorted(windows, key=lambda x: float(x[0]))
    keep = [True] * len(windows_sorted)
    for i in range(len(windows_sorted) - 1):
        s0, e0 = float(windows_sorted[i][0]), float(windows_sorted[i][1])
        s1, _ = float(windows_sorted[i + 1][0]), float(windows_sorted[i + 1][1])
        if (e0 + float(min_gap_s)) > s1:
            keep[i] = False
            keep[i + 1] = False
    return [w for w, k in zip(windows_sorted, keep) if k]


def collect_resampled_segments_for_windows(
    trace: np.ndarray,
    time_stamps: np.ndarray,
    windows: list[tuple[float, float]],
    *,
    resample_points: int,
) -> list[np.ndarray]:
    trace = np.asarray(trace, dtype=float).ravel()
    time_stamps = np.asarray(time_stamps, dtype=float).ravel()
    segments: list[np.ndarray] = []
    for start, end in windows:
        s_idx, e_idx = find_nearest_indices(time_stamps, float(start), float(end))
        if e_idx <= s_idx:
            continue
        y = trace[s_idx : e_idx + 1].astype(float, copy=True)
        if y.size < 2:
            continue
        y = y - y[0]
        t = time_stamps[s_idx : e_idx + 1].astype(float, copy=False)
        dur = float(t[-1] - t[0])
        if dur <= 1e-12:
            continue
        x = (t - t[0]) / dur
        xq = np.linspace(0.0, 1.0, int(resample_points))
        seg = np.interp(xq, x, y)
        segments.append(seg)
    return segments


def compute_event_average_trace(segments: list[np.ndarray]) -> np.ndarray | None:
    if len(segments) == 0:
        return None
    X = np.asarray(segments, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return None
    return np.nanmean(X, axis=0)


def plot_event_average_trace(
    avg_trace: np.ndarray,
    *,
    fish_name: str,
    component_number: int,
    pattern_index: int,
    n_events: int,
    out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(avg_trace, dtype=float)
    x = np.linspace(0.0, 1.0, y.size)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color="black", linewidth=2)
    plt.xlabel("Normalized time (0=start, 1=end)", fontsize=12)
    plt.ylabel("Amplitude (baseline-corrected)", fontsize=12)
    plt.title(f"{fish_name} comp={component_number} pattern={pattern_index} n_events={n_events}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)
    plt.close()


def silhouette_score_from_distance_matrix(D: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = labels.size
    if n <= 2:
        return 0.0
    uniq = np.unique(labels)
    if uniq.size <= 1:
        return 0.0
    s_vals = np.zeros(n, dtype=float)
    for i in range(n):
        li = labels[i]
        same = labels == li
        same[i] = False
        a = float(np.mean(D[i, same])) if np.any(same) else 0.0
        b = np.inf
        for lj in uniq:
            if lj == li:
                continue
            mask = labels == lj
            if not np.any(mask):
                continue
            bj = float(np.mean(D[i, mask]))
            b = min(b, bj)
        if not np.isfinite(b):
            s_vals[i] = 0.0
        else:
            denom = max(a, b)
            s_vals[i] = 0.0 if denom <= 1e-12 else (b - a) / denom
    return float(np.mean(s_vals))


def load_locomotion_candidates(json_path: str | Path) -> list[dict]:
    json_path = Path(json_path)
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results: list[dict] = []
    if not isinstance(payload, dict):
        return results
    for fish_name, comps in payload.items():
        if not isinstance(comps, dict):
            continue
        for comp_key, entry in comps.items():
            try:
                component_number = int(comp_key)
            except Exception:
                continue
            if isinstance(entry, list):
                selected = entry
            elif isinstance(entry, dict):
                selected = entry.get("selected_indices", []) or entry.get("selected_patterns", [])
                if isinstance(selected, list) and len(selected) > 0 and isinstance(selected[0], dict):
                    selected = [
                        x.get("pattern_index")
                        for x in selected
                        if isinstance(x, dict) and x.get("pattern_index") is not None
                    ]
            else:
                selected = []
            selected = [int(x) for x in selected if x is not None]
            if len(selected) == 0:
                continue
            results.append(
                {"fish_name": str(fish_name), "component_number": int(component_number), "selected_indices": selected}
            )
    return results


@dataclass(frozen=True)
class PatternMeta:
    fish_name: str
    component_number: int
    pattern_index: int
    n_events: int


def run_locomotion_event_average_and_clustering(
    *,
    candidates_json_path: str | Path = DEFAULT_CANDIDATES_JSON_PATH,
    out_traces_dir: str | Path = DEFAULT_EVENT_AVG_OUT_DIR,
    out_cluster_dir: str | Path = DEFAULT_CLUSTER_OUT_DIR,
    resample_points: int = DEFAULT_RESAMPLE_POINTS,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    fps: float = DEFAULT_FPS,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    post_end_s: float = DEFAULT_POST_END_S,
) -> None:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    bouts_info = load_bouts_info()
    entries = load_locomotion_candidates(candidates_json_path)
    if len(entries) == 0:
        return

    all_avg: list[np.ndarray] = []
    meta: list[PatternMeta] = []

    out_traces_dir = Path(out_traces_dir)
    out_cluster_dir = Path(out_cluster_dir)
    out_cluster_dir.mkdir(parents=True, exist_ok=True)

    for item in entries:
        fish_name = item["fish_name"]
        component_number = int(item["component_number"])
        selected_indices = [int(x) for x in item["selected_indices"]]

        time_stamps_path = REGISTERED_FISH_DIR / fish_name / "t.npy"
        if not time_stamps_path.exists():
            continue
        time_stamps = np.load(str(time_stamps_path))

        events_by_set = get_locomotion_events_by_set(bouts_info, fish_name, fps=fps, min_interval_s=min_interval_s)
        windows_by_set: dict[str, list[tuple[float, float]]] = {}
        for set_name, events in events_by_set.items():
            windows = [(float(s), float(e) + float(post_end_s)) for s, e, _ in events]
            windows = filter_event_windows_by_min_gap(windows, min_gap_s=float(min_gap_s))
            if len(windows) > 0:
                windows_by_set[set_name] = windows
        if len(windows_by_set) == 0:
            continue

        H_by_set: dict[str, np.ndarray] = {}
        for set_name in windows_by_set.keys():
            H_path = (
                REGISTERED_FISH_DIR
                / fish_name
                / "Temporal_components_full_sets"
                / f"Temporal_components_{set_name}_{component_number}.npy"
            )
            if not H_path.exists():
                continue
            try:
                H_by_set[set_name] = np.load(str(H_path))
            except Exception:
                continue
        if len(H_by_set) == 0:
            continue

        for pattern_index in selected_indices:
            segments: list[np.ndarray] = []
            for set_name, windows in windows_by_set.items():
                H = H_by_set.get(set_name, None)
                if H is None:
                    continue
                if pattern_index < 0 or pattern_index >= H.shape[0]:
                    continue
                trace = H[int(pattern_index), :]
                segments.extend(
                    collect_resampled_segments_for_windows(
                        trace,
                        time_stamps,
                        windows,
                        resample_points=int(resample_points),
                    )
                )

            avg = compute_event_average_trace(segments)
            if avg is None:
                continue

            out_path = out_traces_dir / fish_name / f"component_{component_number}" / f"pattern_{int(pattern_index)}.png"
            plot_event_average_trace(
                avg,
                fish_name=fish_name,
                component_number=component_number,
                pattern_index=int(pattern_index),
                n_events=int(len(segments)),
                out_path=out_path,
            )

            all_avg.append(np.asarray(avg, dtype=float))
            meta.append(
                PatternMeta(
                    fish_name=str(fish_name),
                    component_number=int(component_number),
                    pattern_index=int(pattern_index),
                    n_events=int(len(segments)),
                )
            )

    if len(all_avg) < 2:
        return

    X = np.vstack(all_avg)
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_std = np.std(X, axis=1, keepdims=True)
    row_std = np.where(row_std <= 1e-12, 1e-12, row_std)
    Xz = (X - row_mean) / row_std

    peak = np.max(X, axis=1)
    trough = np.min(X, axis=1)
    ptp = peak - trough
    auc = np.trapz(X, axis=1) / max(1, X.shape[1] - 1)
    t_peak = np.argmax(X, axis=1) / max(1, X.shape[1] - 1)
    t_trough = np.argmin(X, axis=1) / max(1, X.shape[1] - 1)
    d_energy = np.mean(np.diff(X, axis=1) ** 2, axis=1) if X.shape[1] > 1 else np.zeros(X.shape[0])
    scalars = np.vstack([ptp, peak, trough, auc, t_peak, t_trough, d_energy]).T
    scal_mean = scalars.mean(axis=0, keepdims=True)
    scal_std = scalars.std(axis=0, keepdims=True)
    scal_std = np.where(scal_std <= 1e-12, 1e-12, scal_std)
    scal_z = (scalars - scal_mean) / scal_std

    F = np.hstack([Xz, scal_z])
    F_mean = F.mean(axis=0, keepdims=True)
    F_std = F.std(axis=0, keepdims=True)
    F_std = np.where(F_std <= 1e-12, 1e-12, F_std)
    Fz = (F - F_mean) / F_std

    d_condensed = pdist(Fz, metric="euclidean")
    D = squareform(d_condensed)
    Z = linkage(d_condensed, method="ward")

    Fc = Fz - Fz.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Fc, full_matrices=False)
    coords = Fc @ vt.T
    coords2 = coords[:, :2] if coords.shape[1] >= 2 else np.hstack([coords[:, :1], np.zeros((coords.shape[0], 1))])

    max_k = min(8, Fz.shape[0] - 1)
    if max_k < 2:
        return

    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=int(k), criterion="maxclust")
        score = silhouette_score_from_distance_matrix(D, labels)

        plt.figure(figsize=(12, 9))
        uniq = np.unique(labels)
        cmap = cm.get_cmap("tab10", int(max(uniq.size, 1)))
        for ci, c in enumerate(uniq):
            mask = labels == c
            plt.scatter(coords2[mask, 0], coords2[mask, 1], s=60, color=cmap(ci), label=f"Cluster {int(c)}")
        for i, m in enumerate(meta):
            plt.text(coords2[i, 0], coords2[i, 1], f"{m.fish_name}:{m.pattern_index}", fontsize=8)
        plt.xlabel("PC1", fontsize=12)
        plt.ylabel("PC2", fontsize=12)
        plt.title(f"Locomotion pattern clustering (k={k}, silhouette={score:.3g})", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()
        plt.savefig(str(out_cluster_dir / f"locomotion_pattern_clusters_k{k}.png"), dpi=300)
        plt.close()

        rows = [
            {
                "fish_name": m.fish_name,
                "component_number": m.component_number,
                "pattern_index": m.pattern_index,
                "n_events": m.n_events,
                "cluster": int(labels[i]),
                "k": int(k),
                "silhouette": float(score),
            }
            for i, m in enumerate(meta)
        ]
        pd.DataFrame(rows).to_csv(str(out_cluster_dir / f"locomotion_pattern_clusters_k{k}.csv"), index=False, encoding="utf-8-sig")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=Path(__file__).name)
    p.add_argument("--candidates-json", type=str, default=str(DEFAULT_CANDIDATES_JSON_PATH))
    p.add_argument("--out-traces-dir", type=str, default=str(DEFAULT_EVENT_AVG_OUT_DIR))
    p.add_argument("--out-cluster-dir", type=str, default=str(DEFAULT_CLUSTER_OUT_DIR))
    p.add_argument("--resample-points", type=int, default=int(DEFAULT_RESAMPLE_POINTS))
    p.add_argument("--min-gap-s", type=float, default=float(DEFAULT_MIN_GAP_S))
    p.add_argument("--fps", type=float, default=float(DEFAULT_FPS))
    p.add_argument("--min-interval-s", type=float, default=float(DEFAULT_MIN_INTERVAL_S))
    p.add_argument("--post-end-s", type=float, default=float(DEFAULT_POST_END_S))
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_locomotion_event_average_and_clustering(
        candidates_json_path=args.candidates_json,
        out_traces_dir=args.out_traces_dir,
        out_cluster_dir=args.out_cluster_dir,
        resample_points=int(args.resample_points),
        min_gap_s=float(args.min_gap_s),
        fps=float(args.fps),
        min_interval_s=float(args.min_interval_s),
        post_end_s=float(args.post_end_s),
    )


if __name__ == "__main__":
    main()

