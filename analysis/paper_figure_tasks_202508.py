"""
Paper figure generation utilities (20250807/20250808 fish only).

This module consolidates plotting code that was previously scattered across:
- paper plots.py
- new_paper_plots.py

Goals (for GitHub / submission):
- A single, runnable entry point for figure generation.
- Explicit, reproducible I/O contracts (inputs, outputs, expected folder layout).
- A data packaging step that collects the minimal derived data needed to re-run the plots.

Scope restriction:
- Only fish IDs starting with "20250807" or "20250808" are included.

Code provenance (original locations → consolidated entry points here):
- new_paper_plots.py:L51-318 → hunting_overlap_and_energy_ratio_distributions_single_fish_from_pack
- new_paper_plots.py:L958-1468 → Fig1_C_from_pack
- new_paper_plots.py:L2385-2979 → Fig2_energy_ratio_hiearchy_from_pack
- paper plots.py:L2383-2765 → event_average_hunting_prey_sensory_multi_fish_from_pack
- paper plots.py:L2768-3160 → event_average_loom_escape_multi_fish_from_pack
- paper plots.py:L3163-3881 → event_average_locomotion_rois_across_events_multi_fish_from_pack + concat_event_average_vertical
- paper plots.py:L3980-4049 → make_figures: multi-fish 5-condition + vertical concatenation
- paper plots.py:L5556-5590 → make_figures: Misha example TP panel set + vertical concatenation
- paper plots.py:L5632-5677 → make_figures: Hunting example TP panels across fish + vertical concatenation

Two workflows are supported:

1) Package data (run once on a machine that has the original dataset):
   python paper_figure_tasks_202508.py package-data --out <PACKAGE_DIR> --data-root <ASTROCYTE_DATA_ROOT>

2) Generate figures (run on reviewer machine using only PACKAGE_DIR):
   python paper_figure_tasks_202508.py make-figures --data <PACKAGE_DIR> --out <FIG_OUT_DIR>

Reproducibility bundle for reviewers:
- Data package: output of `package-data` (a single folder tree).
- Environment: d:/PycharmProjects/Astrocyte/environment.yml

Environment variables:
- ASTROCYTE_DATA_ROOT: original dataset root (default: D:/2p astrocyte)

Package directory layout produced by `package-data`:
<PACKAGE_DIR>/
  behaviour/
    bouts_info.json
    passivity/passivity.json
    convergence periods/convergence_periods.json
    10 fish data/stimulus_info.xlsx
    tail_angles/
      {fish_name}/
        {set_id}_tail_angles.npz  (t_tail_sec, averaged_tail_angle_deg)
  imaging/
    Registered 10-fish/
      {fish_name}/
        t.npy
        Spatial_components_full_sets/Spatial_components_{K}.npy
        Temporal_components_full_sets/Temporal_components_prey-loom-omr-set00{1..4}_{K}.npy
        intersection_masks_full_sets_Energy ratio_time_corrected/{K}/component_*_intersection_*.npy
        normalized_masked_dfof_prey-loom-omr-set00{1..4}.npy
    p_vals_final/
      {fish_name}_{mode}_inbrain.npy
      {fish_name}_{mode}_p_vals.npy

Figures produced by `make-figures`:
- Example temporal patterns (TPs) panels for multiple fish and vertical concatenations.
- Multi-fish event-triggered averages (5-condition and 2-condition panels) and vertical concatenations.
- Single-fish overlap scatterplots (Hunting vs Loom/Passivity/Struggle/Locomotion).

Notes on reproducibility:
- This script does not attempt to regenerate upstream preprocessing (NMF, masks, p-values).
  Instead it packages the derived arrays required for the plots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


FISH_ID_PATTERN = re.compile(r"^(20250807|20250808)-F\d+$")


Hunting_dict = {
    "20250807-F1": "90_component_29_intersection_3_4",
    "20250807-F2": "90_component_50_intersection_1_2",
    "20250807-F3": "90_component_76_intersection_2_3",
    "20250807-F4": "90_component_61_intersection_2_3",
    "20250808-F1": "70_component_30_intersection_2_3",
}

Passivity_dict = {
    "20250807-F1": "90_component_43_intersection_2_3",
    "20250807-F3": "90_component_7_intersection_1_2",
    "20250807-F4": "90_component_37_intersection_2_3",
    "20250808-F1": "70_component_23_intersection_1_2",
}

Struggle_dict = {
    "20250807-F2": "90_component_16_intersection_3_4",
    "20250807-F3": "90_component_18_intersection_1_2",
    "20250808-F1": "70_component_7_intersection_2_3",
}

Loom_dict = {
    "20250807-F1": "90_component_73_intersection_2_3",
    "20250807-F2": "90_component_9_intersection_2_3",
    "20250807-F3": "90_component_42_intersection_2_3",
    "20250807-F4": "90_component_42_intersection_1_2",
    "20250808-F1": "70_component_9_intersection_2_3",
}

locomtion_1_dict = {
    "20250807-F1": "90_component_84_intersection_3_4",
    "20250807-F2": "90_component_18_intersection_3_4",
    "20250807-F3": "90_component_16_intersection_3_4",
    "20250807-F4": "90_component_26_intersection_2_3",
    "20250808-F1": "70_component_35_intersection_2_3",
}

locomtion_2_dict = {
    "20250807-F1": "90_component_26_intersection_3_4",
    "20250807-F2": "90_component_68_intersection_2_3",
    "20250807-F3": "90_component_26_intersection_2_3",
    "20250807-F4": "90_component_13_intersection_3_4",
    "20250808-F1": "70_component_11_intersection_2_3",
}


def _data_root_from_env(explicit: Optional[str] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("ASTROCYTE_DATA_ROOT", r"D:/2p astrocyte"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_allowed_fish_id(fish_name: str) -> bool:
    return bool(FISH_ID_PATTERN.match(str(fish_name)))


def _filter_bouts_info(bouts_info: dict) -> dict:
    out: dict = {}
    for fish_name, payload in (bouts_info or {}).items():
        if not _is_allowed_fish_id(str(fish_name)):
            continue
        if not isinstance(payload, dict):
            continue
        out[str(fish_name)] = payload
    return out


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _save_json(path: Path, payload: dict) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_component_spec(spec: str) -> Tuple[int, int, str]:
    parts = str(spec).split("_")
    if len(parts) < 6:
        raise ValueError(f"Invalid component spec: {spec!r}")
    component_number = int(parts[0])
    if parts[1] != "component":
        raise ValueError(f"Invalid component spec: {spec!r}")
    component_index = int(parts[2])
    intersection_mask_name = "component_" + str(component_index) + "_" + "_".join(parts[3:])
    return component_number, component_index, intersection_mask_name


@dataclass(frozen=True)
class PackPaths:
    root: Path

    @property
    def behaviour(self) -> Path:
        return self.root / "behaviour"

    @property
    def imaging(self) -> Path:
        return self.root / "imaging"

    @property
    def registered(self) -> Path:
        return self.imaging / "Registered 10-fish"

    @property
    def pvals(self) -> Path:
        return self.imaging / "p_vals_final"

    def fish_dir(self, fish_name: str) -> Path:
        return self.registered / fish_name

    def tail_npz(self, fish_name: str, set_id: str) -> Path:
        return self.behaviour / "tail_angles" / fish_name / f"{set_id}_tail_angles.npz"


def get_stimulus_from_pack(pack: PackPaths, fish_name: str) -> Tuple[List[List[float]], List[str]]:
    stim_xlsx = pack.behaviour / "10 fish data" / "stimulus_info.xlsx"
    df = pd.read_excel(stim_xlsx, engine="openpyxl")
    start = df["start"].tolist()
    end = df["end"].tolist()
    stimuli = df["stimuli"].tolist()
    periods = [[float(s), float(e)] for s, e in zip(start, end)]
    return periods, [str(x) for x in stimuli]


def find_nearest_indices(time_stamps: np.ndarray, start: float, end: float) -> Tuple[int, int]:
    ts = np.asarray(time_stamps, dtype=float).ravel()
    s_idx = int(np.searchsorted(ts, float(start), side="left"))
    e_idx = int(np.searchsorted(ts, float(end), side="left")) - 1
    s_idx = max(0, min(s_idx, ts.size - 1))
    e_idx = max(0, min(e_idx, ts.size - 1))
    return s_idx, e_idx


def slice_matrix_by_time(X: np.ndarray, time_stamps: np.ndarray, period: Sequence[float]) -> np.ndarray:
    start, end = float(period[0]), float(period[1])
    s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
    if e_idx < s_idx:
        return X[:, :0].copy()
    return np.asarray(X[:, s_idx : e_idx + 1], dtype=float).copy()


def concat_example_tps_vertical(image_paths: Sequence[str | Path], out_path: str | Path, gap: int = 20) -> Optional[str]:
    image_paths = [str(p) for p in image_paths if p is not None]
    images: List[Image.Image] = []
    for p in image_paths:
        img = Image.open(p)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        images.append(img)
    if len(images) == 0:
        return None
    target_w = max([im.size[0] for im in images])
    padded: List[Image.Image] = []
    for im in images:
        w, h = im.size
        if w == target_w:
            padded.append(im)
        else:
            canvas = Image.new("RGB", (target_w, h), (255, 255, 255))
            canvas.paste(im, (0, 0))
            padded.append(canvas)
    gap = int(gap) if gap is not None else 0
    gap = max(gap, 0)
    total_h = sum([im.size[1] for im in padded]) + max(0, (len(padded) - 1)) * gap
    canvas = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    y = 0
    for im in padded:
        canvas.paste(im, (0, y))
        y += im.size[1] + gap
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    return str(out_path)


def concat_event_average_vertical(
    image_paths: Sequence[str | Path],
    out_path: str | Path,
    legend_items: Optional[Sequence[Tuple[str, str]]] = None,
) -> Optional[str]:
    image_paths = [str(p) for p in image_paths if p is not None]

    def _crop_white_margins(im: Image.Image, pad: int = 28, white_thr: int = 245) -> Image.Image:
        arr = np.asarray(im)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return im
        mask = np.any(arr[:, :, :3] < int(white_thr), axis=2)
        if not np.any(mask):
            return im
        ys, xs = np.where(mask)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
        x0 = 0
        x1 = int(arr.shape[1])
        return im.crop((x0, y0, x1, y1))

    images: List[Image.Image] = []
    for p in image_paths:
        img = Image.open(p)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        img = _crop_white_margins(img)
        images.append(img)

    if len(images) == 0:
        return None

    target_w = min([im.size[0] for im in images])
    resized: List[Image.Image] = []
    for im in images:
        w, h = im.size
        if w == target_w:
            resized.append(im)
        else:
            new_h = int(round(h * (target_w / float(w))))
            resized.append(im.resize((target_w, new_h), Image.Resampling.LANCZOS))

    gap_h = 14
    legend_w = 0
    legend_font_size = 120
    legend_line_len = 280
    legend_line_w = 28
    legend_gap_y = int(round(legend_font_size * 1.4))
    legend_y0 = int(round(legend_font_size * 0.35))
    legend_text_y_offset = int(round(legend_font_size * 0.35))
    if legend_items:
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size=legend_font_size)
        except Exception:
            try:
                font = ImageFont.truetype(r"C:\Windows\Fonts\calibri.ttf", size=legend_font_size)
            except Exception:
                font = ImageFont.load_default()
        dummy = Image.new("RGB", (10, 10), (255, 255, 255))
        draw_dummy = ImageDraw.Draw(dummy)
        max_text_w = 0
        for label, _ in legend_items:
            bbox = draw_dummy.textbbox((0, 0), str(label), font=font)
            max_text_w = max(max_text_w, int(bbox[2] - bbox[0]))
        legend_w = 40 + legend_line_len + 40 + max_text_w + 60

    total_h = sum([im.size[1] for im in resized]) + gap_h * (len(resized) - 1)
    if legend_items:
        legend_needed_h = legend_y0 + (len(legend_items) - 1) * legend_gap_y + legend_font_size + 60
        total_h = max(int(total_h), int(legend_needed_h))

    canvas = Image.new("RGB", (target_w + legend_w, total_h), (255, 255, 255))
    y = 0
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.size[1] + gap_h

    if legend_items:
        draw = ImageDraw.Draw(canvas)
        x0 = target_w + 40
        y0 = legend_y0
        for idx, (label, color) in enumerate(legend_items):
            yy = y0 + idx * legend_gap_y
            draw.line((x0, yy, x0 + legend_line_len, yy), fill=str(color), width=legend_line_w)
            draw.text((x0 + legend_line_len + 40, yy - legend_text_y_offset), str(label), fill="black", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    return str(out_path)


def _load_filtered_intersection_mask(pack: PackPaths, fish_name: str, mode: str, intersection_mask: np.ndarray, p_thr: float = 0.05) -> np.ndarray:
    intersection_mask = np.asarray(intersection_mask, dtype=bool).reshape(-1)
    inbrain = np.load(str(pack.pvals / f"{fish_name}_{mode}_inbrain.npy")).astype(bool).reshape(-1)
    pvals = np.load(str(pack.pvals / f"{fish_name}_{mode}_p_vals.npy")).reshape(-1)

    inter_inbrain = intersection_mask & inbrain
    new_p_vals: List[float] = []
    p_count = 0
    for i in range(inbrain.shape[0]):
        if bool(inbrain[i]):
            if bool(intersection_mask[i]):
                new_p_vals.append(float(pvals[p_count]))
            p_count += 1

    out = np.zeros_like(inter_inbrain, dtype=bool)
    k = 0
    thr = float(p_thr)
    for i in range(inter_inbrain.shape[0]):
        if bool(inter_inbrain[i]):
            out[i] = float(new_p_vals[k]) < thr
            k += 1
    return out


def _safe_minmax(arrs: Sequence[np.ndarray | None]) -> Tuple[float, float]:
    vals = np.concatenate([np.ravel(a) for a in arrs if a is not None]) if arrs else np.array([], dtype=float)
    if vals.size == 0:
        return -1.0, 1.0
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return -1.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def _summarize_trace_group(
    trace_group: Sequence[np.ndarray],
    trace_weights: Sequence[float],
    *,
    target_len: int,
    error_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_len <= 0:
        target_len = 1
    nan_vec = np.full(target_len, np.nan, dtype=float)
    nan_err = np.full(target_len, np.nan, dtype=float)
    if not trace_group:
        return nan_vec, nan_err, np.empty((0, target_len), dtype=float)
    if len(trace_group) != len(trace_weights):
        raise ValueError("Internal error: trace_group and trace_weights length mismatch.")

    event_means: List[np.ndarray] = []
    for trace in trace_group:
        event_mean = np.nanmean(np.asarray(trace, dtype=float), axis=0)
        current_len = int(event_mean.shape[0])
        if current_len < target_len:
            pad_width = target_len - current_len
            event_mean = np.pad(event_mean, (0, pad_width), mode="constant", constant_values=np.nan)
        elif current_len > target_len:
            event_mean = event_mean[:target_len]
        event_means.append(event_mean)

    event_means_mat = np.vstack(event_means) if event_means else np.empty((0, target_len), dtype=float)
    weights = np.asarray(list(trace_weights), dtype=float).reshape(-1)
    valid = ~np.isnan(event_means_mat)
    weighted = weights[:, None] * np.where(valid, event_means_mat, 0.0)
    denom = np.sum(weights[:, None] * valid, axis=0)
    averaged_trace = np.divide(
        np.sum(weighted, axis=0),
        denom,
        out=np.full(target_len, np.nan, dtype=float),
        where=denom > 0,
    )

    diff = np.where(valid, event_means_mat - averaged_trace[None, :], 0.0)
    var = np.divide(
        np.sum(weights[:, None] * diff**2, axis=0),
        denom,
        out=np.full(target_len, np.nan, dtype=float),
        where=denom > 0,
    )
    std_trace = np.sqrt(var)
    if str(error_mode).lower() == "sem":
        sum_w2 = np.sum((weights[:, None] ** 2) * valid, axis=0)
        n_eff = np.divide(
            denom**2,
            sum_w2,
            out=np.full(target_len, np.nan, dtype=float),
            where=sum_w2 > 0,
        )
        error_trace = np.divide(
            std_trace,
            np.sqrt(n_eff),
            out=np.full(target_len, np.nan, dtype=float),
            where=n_eff > 0,
        )
    else:
        error_trace = std_trace

    return averaged_trace, error_trace, event_means_mat


def Example_TPs_from_pack(
    pack: PackPaths,
    *,
    fish_name: str,
    set_id: Sequence[str] | str,
    time_range: Optional[Sequence[Sequence[float]] | Sequence[float]] = None,
    component_number: int,
    component_index: int,
    mode: str,
    notable_label: Optional[Sequence[str]] = None,
    show_scale_bar: bool = True,
    out_path: str | Path,
) -> str:
    if isinstance(set_id, str):
        set_ids = [set_id]
    else:
        set_ids = list(set_id)

    parsed_ranges: List[Tuple[float, float]] = []
    if time_range is None:
        pass
    elif isinstance(time_range, (list, tuple)):
        if len(time_range) > 0 and isinstance(time_range[0], (list, tuple)):
            for tr in time_range:
                parsed_ranges.append((float(tr[0]), float(tr[1])))
        elif len(time_range) == 2:
            parsed_ranges.append((float(time_range[0]), float(time_range[1])))

    plot_segments_specs: List[dict] = []
    if len(set_ids) == 1:
        curr_set = set_ids[0]
        if not parsed_ranges:
            plot_segments_specs.append({"set_id": curr_set, "range": None})
        else:
            for rng in parsed_ranges:
                plot_segments_specs.append({"set_id": curr_set, "range": rng})
    else:
        if not parsed_ranges:
            for s in set_ids:
                plot_segments_specs.append({"set_id": s, "range": None})
        else:
            for i, s in enumerate(set_ids):
                rng = parsed_ranges[i] if i < len(parsed_ranges) else None
                plot_segments_specs.append({"set_id": s, "range": rng})

    if notable_label is None:
        notable_labels = ["no"] * len(plot_segments_specs)
    else:
        if len(notable_label) != len(plot_segments_specs):
            raise ValueError("notable_label length must match plotted segments.")
        notable_labels = [str(x) for x in notable_label]

    needed_set_indices = set()
    for s in set_ids:
        s_str = str(s)
        if "001" in s_str:
            needed_set_indices.add(1)
        elif "002" in s_str:
            needed_set_indices.add(2)
        elif "003" in s_str:
            needed_set_indices.add(3)
        elif "004" in s_str:
            needed_set_indices.add(4)
    if not needed_set_indices:
        needed_set_indices = {1}

    H_sets: Dict[int, np.ndarray] = {}
    for set_idx in sorted(needed_set_indices):
        H_sets[set_idx] = np.load(
            str(
                pack.fish_dir(fish_name)
                / "Temporal_components_full_sets"
                / f"Temporal_components_prey-loom-omr-set{set_idx:03d}_{component_number}.npy"
            )
        )

    H = np.concatenate([H_sets[k] for k in sorted(H_sets.keys())], axis=1)
    H_min = np.min(H, axis=1, keepdims=True)
    H_max = np.max(H, axis=1, keepdims=True)
    H_range = H_max - H_min
    H_range[H_range == 0] = 1
    H_norm = (H - H_min) / H_range

    split_points = []
    cum = 0
    for k in sorted(H_sets.keys()):
        cum += int(H_sets[k].shape[1])
        split_points.append(cum)
    split_points = split_points[:-1]
    H_norm_parts = np.split(H_norm, split_points, axis=1) if split_points else [H_norm]
    H_norm_by_set = {k: part for k, part in zip(sorted(H_sets.keys()), H_norm_parts)}

    convergence_periods_all = _load_json(pack.behaviour / "convergence periods" / "convergence_periods.json")
    passivity_json = _load_json(pack.behaviour / "passivity" / "passivity.json")
    periods, stimuli = get_stimulus_from_pack(pack, fish_name)
    looming_periods = []
    for idx, stimulus_item in enumerate(stimuli):
        if str(stimulus_item).strip().lower() == "loom" and idx < len(periods):
            looming_periods.append(periods[idx])

    desired_notable_order = {
        "hunting": 0,
        "none": 0,
        "no": 0,
        "omr": 1,
        "locomotion": 1,
        "passivity": 2,
        "struggle": 3,
        "loom": 4,
    }

    def _notable_order_index(label: str) -> int:
        return desired_notable_order.get(str(label).strip().lower(), 999)

    processed_segments = []
    for spec, curr_label in zip(plot_segments_specs, notable_labels):
        curr_set_id = str(spec["set_id"])
        if "001" in curr_set_id:
            set_idx = 1
        elif "002" in curr_set_id:
            set_idx = 2
        elif "003" in curr_set_id:
            set_idx = 3
        elif "004" in curr_set_id:
            set_idx = 4
        else:
            set_idx = 1

        selected_temporal_component = H_norm_by_set.get(set_idx, H_norm_parts[0])[int(component_index)]
        t_img = np.arange(len(selected_temporal_component)) / 3.0
        max_t = float(t_img[-1]) if len(t_img) else 0.0
        if spec["range"] is None:
            x_min, x_max = 0.0, max_t
        else:
            x_min, x_max = float(spec["range"][0]), float(spec["range"][1])

        current_convergence_periods = convergence_periods_all.get(fish_name, {}).get(curr_set_id, [])
        passivity = np.asarray(passivity_json.get(fish_name, {}).get(curr_set_id, []), dtype=float)

        bouts_info = _load_json(pack.behaviour / "bouts_info.json")
        bouts = np.asarray(bouts_info.get(fish_name, {}).get(curr_set_id, {}).get("bout", []), dtype=float)
        behavior = np.asarray(bouts_info.get(fish_name, {}).get(curr_set_id, {}).get("behaviour", []), dtype=object)

        processed_segments.append(
            {
                "set_id": curr_set_id,
                "range": (x_min, x_max),
                "t_img": t_img,
                "selected_temporal_component": selected_temporal_component,
                "current_convergence_periods": current_convergence_periods,
                "passivity": passivity,
                "bouts": bouts,
                "behavior": behavior,
                "notable_label": str(curr_label),
            }
        )

    processed_segments = [
        seg
        for _, seg in sorted(
            enumerate(processed_segments),
            key=lambda t: (_notable_order_index(t[1].get("notable_label")), t[0]),
        )
    ]

    base_font = 25
    durations = []
    for seg in processed_segments:
        s, e = seg["range"]
        dur = abs(float(e) - float(s))
        durations.append(dur if dur > 1e-6 else 1e-6)

    y_ticks = [0.0, 0.5, 1.0]
    top_margin = 0.92
    bottom_with_scale = 0.30
    bottom_no_scale = 0.06
    frac_with = float(top_margin - bottom_with_scale)
    frac_no = float(top_margin - bottom_no_scale)
    base_fig_h = 3.1
    if show_scale_bar:
        fig_h = base_fig_h
        bottom_margin = bottom_with_scale
    else:
        fig_h = base_fig_h * (frac_with / frac_no) if frac_no > 0 else base_fig_h
        bottom_margin = bottom_no_scale

    fig = plt.figure(figsize=(24, fig_h))
    gs = fig.add_gridspec(1, len(processed_segments), width_ratios=durations, wspace=0.32)
    axs = []
    for c in range(len(processed_segments)):
        sharey = axs[0] if c > 0 else None
        ax = fig.add_subplot(gs[0, c], sharey=sharey)
        axs.append(ax)

    for i, seg_data in enumerate(processed_segments):
        x_min, x_max = seg_data["range"]
        t_img = seg_data["t_img"]
        selected_temporal_component = seg_data["selected_temporal_component"]
        current_convergence_periods = seg_data["current_convergence_periods"]
        passivity = seg_data["passivity"]
        bouts = seg_data["bouts"]
        behavior = seg_data["behavior"]
        curr_notable_label = str(seg_data["notable_label"]).strip().lower()
        ax = axs[i]

        ax.plot(t_img, selected_temporal_component, color="black", lw=3, zorder=3)
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", which="major", labelsize=base_font)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.set_xticks([])

        start_color = "#0e5fab" if curr_notable_label == "locomotion" else None
        mode_lower = str(mode).strip().lower()
        is_hunting_subplot = (mode_lower == "hunting") and (curr_notable_label in {"hunting", "none", "no"})

        if is_hunting_subplot:
            try:
                jt_starts = []
                jt_ends = []
                for bout_item, behavior_item in zip(bouts, behavior):
                    behavior_label = str(behavior_item).lower()
                    is_j_turn = ("j-turn" in behavior_label) or ("j turn" in behavior_label) or ("j_turn" in behavior_label)
                    if not is_j_turn:
                        continue
                    bs = float(bout_item[0]) / 300.0
                    be = float(bout_item[1]) / 300.0
                    if be < x_min or bs > x_max:
                        continue
                    jt_starts.append(bs)
                    jt_ends.append(be)
                if jt_starts and jt_ends:
                    span_start = float(np.min(jt_starts))
                    span_end = float(np.max(jt_ends))
                    if span_end > span_start:
                        ax.axvspan(span_start, span_end, color="magenta", alpha=1, linewidth=0, zorder=1)
            except Exception:
                pass
        else:
            try:
                for cp in current_convergence_periods:
                    cs = float(cp[0]) / 300.0
                    if x_min <= cs <= x_max:
                        ax.axvline(cs, color="magenta", alpha=0.9, linewidth=1, zorder=1)
            except Exception:
                pass

        show_passivity = (mode_lower == "passivity") or (curr_notable_label == "passivity")
        if show_passivity and curr_notable_label != "locomotion":
            try:
                for cp in passivity:
                    cs, ce = float(cp[0]), float(cp[1])
                    visible_start = max(cs, x_min)
                    visible_end = min(ce, x_max)
                    if visible_end > visible_start:
                        ax.axvspan(visible_start, visible_end, color=(start_color or "#05830D"), alpha=0.5, linewidth=0, zorder=1)
            except Exception:
                pass

        try:
            if curr_notable_label == "struggle":
                for bout_item, behavior_item in zip(bouts, behavior):
                    if "struggle" not in str(behavior_item).lower():
                        continue
                    bs, be = float(bout_item[0]) / 300.0, float(bout_item[1]) / 300.0
                    visible_start = max(bs, x_min)
                    visible_end = min(be, x_max)
                    if visible_end > visible_start:
                        ax.axvspan(visible_start, visible_end, color=(start_color or "purple"), alpha=1, linewidth=0, zorder=1)
            elif curr_notable_label == "loom":
                for period in looming_periods:
                    ls, le = float(period[0]), float(period[1])
                    visible_start = max(ls, x_min)
                    visible_end = min(le, x_max)
                    if visible_end > visible_start:
                        ax.axvspan(visible_start, visible_end, color=(start_color or "orange"), alpha=0.5, linewidth=0, zorder=1)
            elif curr_notable_label == "locomotion":
                for bout_item, behavior_item in zip(bouts, behavior):
                    behavior_label = str(behavior_item).lower()
                    is_locomotion = ("turn left" in behavior_label) or ("turn right" in behavior_label) or ("slow swim" in behavior_label)
                    if not is_locomotion:
                        continue
                    bs, be = float(bout_item[0]) / 300.0, float(bout_item[1]) / 300.0
                    visible_start = max(bs, x_min)
                    visible_end = min(be, x_max)
                    if visible_end > visible_start:
                        ax.axvspan(visible_start, visible_end, color=(start_color or "brown"), alpha=1, linewidth=0, zorder=1)
        except Exception:
            pass

        if i > 0:
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.spines["left"].set_visible(False)
        if len(processed_segments) > 1 and i < len(processed_segments) - 1:
            ax.spines["right"].set_visible(False)

        if i == 0 and show_scale_bar:
            try:
                from matplotlib.transforms import blended_transform_factory

                scale_len = 10.0
                if (x_max - x_min) < scale_len:
                    scale_len = max(0.0, float(x_max - x_min))
                if scale_len > 0:
                    trans = blended_transform_factory(ax.transData, ax.transAxes)
                    y_bar = -0.18
                    x0 = float(x_min)
                    x1 = float(x_min + scale_len)
                    ax.plot([x0, x1], [y_bar, y_bar], transform=trans, color="black", linewidth=2.5, clip_on=False)
                    ax.text((x0 + x1) / 2, y_bar - 0.06, f"{scale_len:g}s", transform=trans, ha="center", va="top", fontsize=base_font + 4)
            except Exception:
                pass

    fig.subplots_adjust(left=0.10, right=0.99, bottom=float(bottom_margin), top=float(top_margin), wspace=0.32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, facecolor="white")
    plt.close(fig)
    return str(out_path)


def hunting_overlap_and_energy_ratio_distributions_single_fish_from_pack(
    pack: PackPaths,
    *,
    fish_name: str,
    output_dir: str | Path,
    locomotion_er_mode: str = "max",
    default_xlim: Tuple[float, float] = (0.0, 0.06),
    default_ylim: Tuple[float, float] = (0.0, 0.06),
    axis_limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    plot_grey_points: bool = True,
    grey_fraction: float = 1.0,
    grey_alpha: float = 0.9,
    grey_size: float = 14.0,
    dpi: int = 300,
) -> Dict[str, dict]:
    hunting_entry = Hunting_dict.get(fish_name)
    loom_entry = Loom_dict.get(fish_name)
    passivity_entry = Passivity_dict.get(fish_name)
    struggle_entry = Struggle_dict.get(fish_name)
    locomotion_1 = {"20250807-F3": "90_component_16_intersection_3_4"}.get(fish_name)
    locomotion_2 = {"20250807-F3": "90_component_26_intersection_2_3"}.get(fish_name)

    if hunting_entry is None or loom_entry is None or passivity_entry is None or struggle_entry is None:
        raise ValueError(f"Missing dict entry for fish_name={fish_name}")
    if locomotion_1 is None or locomotion_2 is None:
        raise ValueError(f"Missing locomotion entries for fish_name={fish_name}")

    def parse_entry(entry: str) -> Tuple[int, int, str]:
        comp_num = int(entry.split("_")[0])
        comp_idx = int(entry.split("_")[2])
        mask_name = entry.split(f"{comp_num}_")[1]
        return comp_num, comp_idx, mask_name

    hunting_comp_num, hunting_idx, hunting_mask_name = parse_entry(hunting_entry)
    loom_comp_num, loom_idx, loom_mask_name = parse_entry(loom_entry)
    pass_comp_num, pass_idx, pass_mask_name = parse_entry(passivity_entry)
    struggle_comp_num, struggle_idx, struggle_mask_name = parse_entry(struggle_entry)
    loco1_comp_num, loco1_idx, loco1_mask_name = parse_entry(locomotion_1)
    loco2_comp_num, loco2_idx, loco2_mask_name = parse_entry(locomotion_2)

    comp_nums = {hunting_comp_num, loom_comp_num, pass_comp_num, struggle_comp_num, loco1_comp_num, loco2_comp_num}
    if len(comp_nums) != 1:
        raise ValueError(f"All patterns must use the same component_number, got {sorted(comp_nums)}")
    component_number = comp_nums.pop()

    fish_dir = pack.fish_dir(fish_name)
    W = np.load(str(fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{component_number}.npy"))
    H_parts = []
    for set_id in ("prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set003", "prey-loom-omr-set004"):
        H_parts.append(
            np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{component_number}.npy"))
        )
    H = np.concatenate(tuple(H_parts), axis=1)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    W_scaled = W * norms.T
    component_energies = np.abs(W_scaled)
    total_energies = np.maximum(np.sum(component_energies, axis=1, keepdims=True), 1e-12)
    FE = component_energies / total_energies

    hunting_er = FE[:, hunting_idx]
    loom_er = FE[:, loom_idx]
    pass_er = FE[:, pass_idx]
    struggle_er = FE[:, struggle_idx]
    loco1_er = FE[:, loco1_idx]
    loco2_er = FE[:, loco2_idx]

    if str(locomotion_er_mode).lower() == "max":
        locomotion_er = np.maximum(loco1_er, loco2_er)
    elif str(locomotion_er_mode).lower() == "sum":
        locomotion_er = loco1_er + loco2_er
    else:
        raise ValueError("locomotion_er_mode must be 'max' or 'sum'")

    def load_mask(mask_name: str) -> np.ndarray:
        p = fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(component_number) / f"{mask_name}.npy"
        return np.load(str(p)).astype(bool)

    hunting_mask = load_mask(hunting_mask_name)
    loom_mask = load_mask(loom_mask_name)
    pass_mask = load_mask(pass_mask_name)
    struggle_mask = load_mask(struggle_mask_name)
    loco1_mask = load_mask(loco1_mask_name)
    loco2_mask = load_mask(loco2_mask_name)
    locomotion_mask = loco1_mask | loco2_mask

    if not (hunting_mask.shape[0] == W.shape[0] == FE.shape[0]):
        raise ValueError("Mask length does not match number of ROIs")

    palette = {
        "Hunting": "#4AC9FF",
        "Loom": "orange",
        "Passivity": "#05830D",
        "Struggle": "purple",
        "Locomotion": "#0e5fab",
        "Overlap": "black",
    }

    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.sum(a & b))
        union = int(np.sum(a | b))
        return float(inter / union) if union else 0.0

    def dice(a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.sum(a & b))
        denom = int(np.sum(a) + np.sum(b))
        return float(2 * inter / denom) if denom else 0.0

    def plot_pair(other_name: str, other_mask: np.ndarray, other_er: np.ndarray) -> dict:
        overlap_mask = hunting_mask & other_mask
        hunting_only_mask = hunting_mask & ~other_mask
        other_only_mask = other_mask & ~hunting_mask
        grey_mask = ~(hunting_mask | other_mask)

        x_h = hunting_er[hunting_only_mask]
        y_h = other_er[hunting_only_mask]
        x_o = hunting_er[other_only_mask]
        y_o = other_er[other_only_mask]
        x_i = hunting_er[overlap_mask]
        y_i = other_er[overlap_mask]

        n_h = int(np.sum(hunting_only_mask))
        n_o = int(np.sum(other_only_mask))
        n_i = int(np.sum(overlap_mask))
        n_h_all = int(np.sum(hunting_mask))
        n_o_all = int(np.sum(other_mask))

        J = jaccard(hunting_mask, other_mask)
        D = dice(hunting_mask, other_mask)

        local_xlim = default_xlim
        local_ylim = default_ylim
        if axis_limits and other_name in axis_limits:
            spec = axis_limits[other_name] or {}
            if spec.get("xlim") is not None:
                local_xlim = spec["xlim"]
            if spec.get("ylim") is not None:
                local_ylim = spec["ylim"]

        fig, ax_scatter = plt.subplots(figsize=(7.8, 7.2), dpi=int(dpi))

        if plot_grey_points and np.any(grey_mask):
            if not (0.0 < float(grey_fraction) <= 1.0):
                raise ValueError("grey_fraction must be in (0, 1]")
            grey_idx = np.where(grey_mask)[0]
            n_grey = int(grey_idx.size)
            k = n_grey if float(grey_fraction) >= 1.0 else max(1, int(np.floor(n_grey * float(grey_fraction))))
            rng = np.random.default_rng(0)
            chosen = grey_idx if k >= n_grey else rng.choice(grey_idx, size=k, replace=False)
            ax_scatter.scatter(
                hunting_er[chosen],
                other_er[chosen],
                c="grey",
                s=float(grey_size),
                alpha=float(grey_alpha),
                label=f"Not selected ({n_grey})",
                linewidths=0,
            )
        if x_o.size:
            ax_scatter.scatter(x_o, y_o, c=palette[other_name], s=float(grey_size), alpha=0.9, label=f"{other_name} only ({n_o})")
        if x_h.size:
            ax_scatter.scatter(x_h, y_h, c=palette["Hunting"], s=float(grey_size), alpha=0.9, label=f"Hunting only ({n_h})")
        if x_i.size:
            ax_scatter.scatter(x_i, y_i, c=palette["Overlap"], s=float(grey_size), alpha=0.9, label=f"Overlap ({n_i})")

        tick_label_size = 19
        text_size = tick_label_size + 2
        legend_size = text_size
        ax_scatter.set_xlabel("Hunting energy ratio", fontsize=text_size)
        ax_scatter.set_ylabel(f"{other_name} energy ratio", fontsize=text_size)
        ax_scatter.tick_params(axis="both", which="major", labelsize=tick_label_size)
        ax_scatter.set_xlim(*local_xlim)
        ax_scatter.set_ylim(*local_ylim)
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)

        j_percent = int(np.round(J * 100))
        from matplotlib.lines import Line2D

        handles, labels = ax_scatter.get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", marker=None, color="none"))
        labels.append(f"Jaccard overlap ratio: {j_percent}%")
        leg = ax_scatter.legend(handles, labels, fontsize=legend_size, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, loc="upper right")
        if leg.get_texts():
            leg.get_texts()[-1].set_fontstyle("italic")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fish_name}_Hunting_vs_{other_name}_K{component_number}.png"
        fig.savefig(str(out_path), dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)

        return {
            "other": other_name,
            "component_number": int(component_number),
            "hunting_index": int(hunting_idx),
            "other_indices": None,
            "n_hunting": int(n_h_all),
            "n_other": int(n_o_all),
            "n_overlap": int(n_i),
            "jaccard": float(J),
            "dice": float(D),
            "figure_path": str(out_path),
        }

    results = {
        "Hunting_vs_Loom": plot_pair("Loom", loom_mask, loom_er),
        "Hunting_vs_Passivity": plot_pair("Passivity", pass_mask, pass_er),
        "Hunting_vs_Struggle": plot_pair("Struggle", struggle_mask, struggle_er),
        "Hunting_vs_Locomotion": plot_pair("Locomotion", locomotion_mask, locomotion_er),
    }
    results["Hunting_vs_Locomotion"]["other_indices"] = [int(loco1_idx), int(loco2_idx)]
    results["Hunting_vs_Loom"]["other_indices"] = [int(loom_idx)]
    results["Hunting_vs_Passivity"]["other_indices"] = [int(pass_idx)]
    results["Hunting_vs_Struggle"]["other_indices"] = [int(struggle_idx)]
    return results


def _summarize_trace_group(
    trace_group: List[np.ndarray],
    trace_weights: List[float],
    target_len: int,
    error_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(trace_group) == 0:
        nan_vec = np.full(target_len, np.nan, dtype=float)
        return nan_vec, nan_vec.copy(), np.empty((0, target_len), dtype=float)
    if len(trace_group) != len(trace_weights):
        raise ValueError("trace_group and trace_weights length mismatch.")

    event_means = []
    for trace in trace_group:
        event_mean = np.nanmean(trace, axis=0)
        if event_mean.shape[0] < target_len:
            event_mean = np.pad(event_mean, (0, target_len - event_mean.shape[0]), mode="constant", constant_values=np.nan)
        elif event_mean.shape[0] > target_len:
            event_mean = event_mean[:target_len]
        event_means.append(event_mean)
    event_means = np.vstack(event_means)

    weights = np.asarray(trace_weights, dtype=float)
    valid = ~np.isnan(event_means)
    weighted = weights[:, None] * np.where(valid, event_means, 0.0)
    denom = np.sum(weights[:, None] * valid, axis=0)
    averaged_trace = np.divide(np.sum(weighted, axis=0), denom, out=np.full(target_len, np.nan, dtype=float), where=denom > 0)

    diff = np.where(valid, event_means - averaged_trace[None, :], 0.0)
    var = np.divide(np.sum(weights[:, None] * diff**2, axis=0), denom, out=np.full(target_len, np.nan, dtype=float), where=denom > 0)
    std_trace = np.sqrt(var)

    if str(error_mode).lower() == "sem":
        sum_w2 = np.sum((weights[:, None] ** 2) * valid, axis=0)
        n_eff = np.divide(denom**2, sum_w2, out=np.full(target_len, np.nan, dtype=float), where=sum_w2 > 0)
        error_trace = np.divide(std_trace, np.sqrt(n_eff), out=np.full(target_len, np.nan, dtype=float), where=n_eff > 0)
    else:
        error_trace = std_trace

    return averaged_trace, error_trace, event_means


def event_average_5conditions_multi_fish_from_pack(
    pack: PackPaths,
    *,
    fish_names: Sequence[str],
    set_id: str,
    mode: str,
    trace_color: str,
    per_fish_component_specs: Dict[str, str],
    window_size: int = 30,
    plot_error: bool = True,
    error_mode: str = "sem",
    show_xaxis_elements: bool = True,
    show_mode_titles: bool = True,
    out_path: str | Path,
) -> str:
    fish_names = [str(x) for x in fish_names]
    fish_weights = np.full(len(fish_names), 1.0 / len(fish_names), dtype=float)

    def _single_fish_trace_groups(fish_name: str, spec: str) -> Dict[str, List[np.ndarray]]:
        component_number, component_index, intersection_mask_name = parse_component_spec(spec)
        fish_dir = pack.fish_dir(fish_name)
        time_stamps = np.load(str(fish_dir / "t.npy"))

        W = np.load(str(fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{component_number}.npy"))
        H = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{component_number}.npy"))
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        W_scaled = W * norms.T
        component_energies = np.abs(W_scaled)
        total_energies = np.maximum(np.sum(component_energies, axis=1, keepdims=True), 1e-12)
        FE_matrix = component_energies / total_energies
        selected_energy_ratio = FE_matrix[:, int(component_index)]

        intersection_mask = np.load(
            str(
                fish_dir
                / "intersection_masks_full_sets_Energy ratio_time_corrected"
                / str(component_number)
                / f"{intersection_mask_name}.npy"
            )
        ).astype(bool)

        p_mode = "locomotion" if str(mode).strip().lower() in {"omr", "locomotion"} else str(mode)
        intersection_mask = _load_filtered_intersection_mask(pack, fish_name, p_mode, intersection_mask, p_thr=0.05)

        masked_energy_ratio = selected_energy_ratio[intersection_mask]
        sort_indices = np.argsort(masked_energy_ratio)[::-1]

        bouts_info = _load_json(pack.behaviour / "bouts_info.json")
        passivity_json = _load_json(pack.behaviour / "passivity" / "passivity.json")
        convergence_periods = _load_json(pack.behaviour / "convergence periods" / "convergence_periods.json")
        sets = list(convergence_periods.get(fish_name, {}).keys())

        stimulus_periods, stimuli = get_stimulus_from_pack(pack, fish_name)
        prey_periods = []
        omr_periods = []
        for i in range(len(stimuli)):
            lab = str(stimuli[i]).strip().lower()
            if lab == "prey":
                prey_periods.append(stimulus_periods[i])
            elif lab == "omr":
                omr_periods.append(stimulus_periods[i])

        looming_periods = []
        for i in range(len(stimuli)):
            if str(stimuli[i]).strip().lower() == "loom":
                looming_periods.append(stimulus_periods[i])

        traces_hunting: List[np.ndarray] = []
        traces_passivity: List[np.ndarray] = []
        traces_omr: List[np.ndarray] = []
        traces_loom: List[np.ndarray] = []
        traces_struggle: List[np.ndarray] = []

        for set_index in sets:
            X = np.load(str(fish_dir / f"normalized_masked_dfof_{set_index}.npy"))
            X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)
            masked_X = X_norm[intersection_mask]
            sorted_masked_X = masked_X[sort_indices]

            bouts_info_temp = bouts_info.get(fish_name, {}).get(set_index, {})
            behavior_temp = np.asarray(bouts_info_temp.get("behaviour", []), dtype=object)
            bouts_temp = np.asarray(bouts_info_temp.get("bout", []), dtype=float)

            conv_list = convergence_periods.get(fish_name, {}).get(set_index, [])
            conv_sec = [[float(c[0]) / 300.0, float(c[1]) / 300.0] for c in conv_list]
            for bout in conv_sec:
                start = float(bout[0]) - float(window_size)
                end = float(bout[0]) + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                traces_hunting.append(slices)

            pass_list = passivity_json.get(fish_name, {}).get(set_index, [])
            for period in pass_list:
                start = float(period[0]) - float(window_size)
                end = float(period[0]) + float(window_size)
                traces_passivity.append(slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end]))

            for period in omr_periods:
                omr_start, omr_end = float(period[0]), float(period[1])
                omr_swim_bouts = []
                for behavior_item, bout_item in zip(behavior_temp, bouts_temp):
                    behavior_label = str(behavior_item).lower()
                    is_omr_swim = ("turn left" in behavior_label) or ("turn right" in behavior_label) or ("slow swim" in behavior_label)
                    if not is_omr_swim:
                        continue
                    bout_start = float(bout_item[0]) / 300.0
                    bout_end = float(bout_item[1]) / 300.0
                    if omr_start <= bout_start <= omr_end:
                        omr_swim_bouts.append([bout_start, bout_end])
                if not omr_swim_bouts:
                    continue
                omr_swim_bouts.sort(key=lambda x: x[0])
                cluster_start = None
                cluster_end = None
                omr_event_centers = []
                for bout_start, bout_end in omr_swim_bouts:
                    if cluster_start is None:
                        cluster_start = bout_start
                        cluster_end = bout_end
                    elif bout_start - cluster_end < float(window_size):
                        cluster_end = max(cluster_end, bout_end)
                    else:
                        omr_event_centers.append(cluster_start)
                        cluster_start = bout_start
                        cluster_end = bout_end
                if cluster_start is not None:
                    omr_event_centers.append(cluster_start)
                for center_time in omr_event_centers:
                    start = float(center_time) - float(window_size)
                    end = float(center_time) + float(window_size)
                    traces_omr.append(slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end]))

            escape_indices = list(np.where(np.asarray([str(x) for x in behavior_temp]) == "escape")[0])
            escape_periods = bouts_temp[escape_indices] / 300.0 if len(escape_indices) else np.empty((0, 2), dtype=float)
            struggle_indices = [idx for idx, b in enumerate(behavior_temp) if "struggle" in str(b).lower()]
            struggle_periods = bouts_temp[struggle_indices] / 300.0 if len(struggle_indices) else np.empty((0, 2), dtype=float)

            for period in looming_periods:
                loom_start, loom_end = float(period[0]), float(period[1])
                is_overlapping = False
                for esc in escape_periods:
                    if max(loom_start, float(esc[0])) < min(loom_end, float(esc[1])):
                        is_overlapping = True
                        break
                if not is_overlapping:
                    for st in struggle_periods:
                        if max(loom_start, float(st[0])) < min(loom_end, float(st[1])):
                            is_overlapping = True
                            break
                if is_overlapping:
                    continue
                start = loom_start - float(window_size)
                end = loom_start + float(window_size)
                traces_loom.append(slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end]))

            for st in struggle_periods:
                center_time = float(st[0])
                start = center_time - float(window_size)
                end = center_time + float(window_size)
                traces_struggle.append(slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end]))

        return {"hunting": traces_hunting, "passivity": traces_passivity, "omr": traces_omr, "loom": traces_loom, "struggle": traces_struggle}

    pooled: Dict[str, List[np.ndarray]] = {"hunting": [], "passivity": [], "omr": [], "loom": [], "struggle": []}
    pooled_w: Dict[str, List[float]] = {k: [] for k in pooled.keys()}
    for fish_idx, fish_name in enumerate(fish_names):
        spec = per_fish_component_specs[fish_name]
        groups = _single_fish_trace_groups(fish_name, spec)
        fish_weight = float(fish_weights[fish_idx])
        for key, traces in groups.items():
            if not traces:
                continue
            pooled[key].extend(traces)
            per_event_weight = fish_weight / float(len(traces))
            pooled_w[key].extend([per_event_weight] * len(traces))

    aggregated: Dict[str, dict] = {}
    per_cond_x: Dict[str, np.ndarray] = {}
    for key, trace_group in pooled.items():
        trace_weights = pooled_w[key]
        target_len = max((int(np.asarray(t).shape[1]) for t in trace_group), default=1)
        x = np.linspace(-float(window_size), float(window_size), target_len)
        per_cond_x[key] = x
        mean_trace, err_trace, _ = _summarize_trace_group(trace_group, trace_weights, target_len, error_mode=error_mode)
        aggregated[key] = {"mean": mean_trace, "err": err_trace}

    y_min = float(np.nanmin([np.nanmin(aggregated[k]["mean"] - aggregated[k]["err"]) for k in aggregated.keys()]))
    y_max = float(np.nanmax([np.nanmax(aggregated[k]["mean"] + aggregated[k]["err"]) for k in aggregated.keys()]))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0

    fig_h = 4.8 if show_xaxis_elements else 4.0
    fig = plt.figure(figsize=(12.8, fig_h))
    gs = gridspec.GridSpec(1, 5, wspace=0.22, hspace=0.1)
    axes = [plt.subplot(gs[0, i]) for i in range(5)]
    order = [("hunting", "Hunting", "Time from \nhunting onset (s)", "Hunting\nonset"),
             ("omr", "OMR", "Time from \nOMR onset (s)", "OMR\nonset"),
             ("passivity", "Passivity", "Time from \npassivity onset (s)", "Passivity\nonset"),
             ("struggle", "Struggle", "Time from \nstruggle onset (s)", "Struggle\nonset"),
             ("loom", "Loom", "Time from \nloom onset (s)", "Loom\nonset")]

    def _plot_panel(ax, x, data, title, xlabel, annotate_text, show_ylabel: bool):
        mean_trace = data["mean"]
        err_trace = data["err"]
        if plot_error:
            ax.fill_between(x, mean_trace - err_trace, mean_trace + err_trace, color="gray", alpha=0.35, edgecolor="none")
        ax.plot(x, mean_trace, color=trace_color, lw=5)
        if show_mode_titles:
            ax.set_title(title, fontsize=28)
        else:
            ax.set_title("")
        ax.axvline(x=0, color="black", linestyle="--")
        ax.set_xlim(-float(window_size), float(window_size))
        if show_xaxis_elements:
            ax.set_xlabel(xlabel, fontsize=28)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.set_xticks([])
        if show_ylabel:
            ax.set_ylabel("dF/F", fontsize=28)
        else:
            ax.set_yticks([])
        y_text = y_max * 0.6
        ax.annotate(
            annotate_text,
            xy=(0, y_text),
            xytext=(-3, y_text),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=28,
            fontstyle="italic",
        )
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="y", which="major", labelsize=22)
        if show_xaxis_elements:
            ax.tick_params(axis="x", which="major", labelsize=22)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v: .1f}"))

    for i, (key, title, xlabel, ann) in enumerate(order):
        _plot_panel(axes[i], per_cond_x[key], aggregated[key], title, xlabel, ann, show_ylabel=(i == 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bottom = 0.28 if show_xaxis_elements else 0.14
    plt.subplots_adjust(left=0.10, right=0.98, bottom=bottom, top=0.88)
    plt.savefig(str(out_path), dpi=300)
    plt.close(fig)
    return str(out_path)


def event_average_hunting_prey_sensory_multi_fish_from_pack(
    pack: PackPaths,
    *,
    fish_names: Sequence[str],
    set_id: str,
    average_ratio: Optional[Sequence[float]] = None,
    per_fish_component_specs: Optional[Dict[str, str]] = None,
    window_size: float = 30.0,
    trace_color: str = "deepskyblue",
    plot_error: bool = True,
    error_mode: str = "sem",
    set_minmax: bool = False,
    show_xaxis_elements: bool = True,
    show_mode_titles: bool = True,
    out_path: str | Path = "average.png",
) -> str:
    fish_names = [str(f) for f in fish_names]
    if not fish_names:
        raise ValueError("fish_names is empty.")
    if per_fish_component_specs is None:
        per_fish_component_specs = Hunting_dict

    if average_ratio is None:
        fish_weights = np.full(len(fish_names), 1.0 / len(fish_names), dtype=float)
    else:
        fish_weights = np.asarray(list(average_ratio), dtype=float).reshape(-1)
        if fish_weights.shape[0] != len(fish_names):
            raise ValueError("average_ratio length does not match fish_names length.")
        if np.any(~np.isfinite(fish_weights)) or np.any(fish_weights < 0):
            raise ValueError("average_ratio must be finite and non-negative.")
        s = float(np.sum(fish_weights))
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"average_ratio must sum to 1 (got {s}).")

    conv_all = _load_json(pack.behaviour / "convergence periods" / "convergence_periods.json")

    periods, stimuli = get_stimulus_from_pack(pack, fish_names[0])
    prey_periods: List[List[float]] = []
    for (start, end), lab in zip(periods, stimuli):
        if str(lab).strip().lower() == "prey":
            prey_periods.append([float(start), float(end)])

    pooled_trace_groups: Dict[str, List[np.ndarray]] = {"hunting": [], "prey_sensory": []}
    pooled_trace_weights: Dict[str, List[float]] = {"hunting": [], "prey_sensory": []}

    for fish_idx, fish_name in enumerate(fish_names):
        if fish_name not in per_fish_component_specs:
            raise KeyError(f"Missing component spec for fish_name={fish_name!r}")
        component_number, component_index, intersection_mask_name = parse_component_spec(per_fish_component_specs[fish_name])

        fish_dir = pack.fish_dir(fish_name)
        time_stamps = np.load(str(fish_dir / "t.npy"))
        W = np.load(str(fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{component_number}.npy"))
        H = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{component_number}.npy"))
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        W_scaled = W * norms.T
        component_energies = np.abs(W_scaled)
        total_energies = np.sum(component_energies, axis=1, keepdims=True)
        total_energies = np.maximum(total_energies, 1e-12)
        FE_matrix = component_energies / total_energies
        selected_energy_ratio = FE_matrix[:, int(component_index)]

        intersection_mask = np.load(
            str(fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(component_number) / f"{intersection_mask_name}.npy")
        )
        intersection_mask = _load_filtered_intersection_mask(pack, fish_name, "Hunting", intersection_mask, p_thr=0.05)

        masked_energy_ratio = selected_energy_ratio[intersection_mask]
        sort_indices = np.argsort(masked_energy_ratio)[::-1]

        conv_fish = conv_all.get(fish_name, {})
        set_ids = list(conv_fish.keys())

        traces_hunting: List[np.ndarray] = []
        traces_prey: List[np.ndarray] = []

        for set_index in set_ids:
            X = np.load(str(fish_dir / f"normalized_masked_dfof_{set_index}.npy"))
            X = np.asarray(X, dtype=float)
            X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)
            masked_X = X_norm[intersection_mask]
            sorted_masked_X = masked_X[sort_indices]

            current_convergence_periods = conv_fish.get(set_index, [])
            convergence_periods_seconds: List[List[float]] = []
            for c in current_convergence_periods:
                convergence_periods_seconds.append([float(c[0]) / 300.0, float(c[1]) / 300.0])

            for bout in convergence_periods_seconds:
                center_time = float(bout[0])
                start = center_time - float(window_size)
                end = center_time + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] >= 2:
                    traces_hunting.append(slices)

            for period in prey_periods:
                center_time = float(period[0])
                start = center_time - float(window_size)
                end = center_time + float(window_size)
                overlaps_hunting = False
                for hunt_start, hunt_end in convergence_periods_seconds:
                    if max(start, hunt_start) < min(end, hunt_end):
                        overlaps_hunting = True
                        break
                if overlaps_hunting:
                    continue
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] >= 2:
                    traces_prey.append(slices)

        fish_weight = float(fish_weights[fish_idx])
        if traces_hunting:
            per_event_weight = fish_weight / float(len(traces_hunting))
            pooled_trace_groups["hunting"].extend(traces_hunting)
            pooled_trace_weights["hunting"].extend([per_event_weight] * len(traces_hunting))
        if traces_prey:
            per_event_weight = fish_weight / float(len(traces_prey))
            pooled_trace_groups["prey_sensory"].extend(traces_prey)
            pooled_trace_weights["prey_sensory"].extend([per_event_weight] * len(traces_prey))

    aggregated: Dict[str, Dict[str, np.ndarray]] = {}
    per_cond_x: Dict[str, np.ndarray] = {}
    for key, trace_group in pooled_trace_groups.items():
        trace_weights = pooled_trace_weights[key]
        target_len = max(int(np.asarray(t).shape[1]) for t in trace_group) if trace_group else 1
        per_cond_x[key] = np.linspace(-float(window_size), float(window_size), target_len)
        mean_trace, err_trace, all_traces = _summarize_trace_group(trace_group, trace_weights, target_len=target_len, error_mode=error_mode)
        aggregated[key] = {"mean": mean_trace, "err": err_trace, "all": all_traces}

    y_min_top, y_max_top = _safe_minmax(
        [
            aggregated["hunting"]["mean"] - aggregated["hunting"]["err"],
            aggregated["hunting"]["mean"] + aggregated["hunting"]["err"],
            aggregated["prey_sensory"]["mean"] - aggregated["prey_sensory"]["err"],
            aggregated["prey_sensory"]["mean"] + aggregated["prey_sensory"]["err"],
        ]
    )

    fig_h = 4.8 if show_xaxis_elements else 4.0
    fig = plt.figure(figsize=(12.8, fig_h))
    gs = gridspec.GridSpec(1, 2, wspace=0.20, hspace=0.1)
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])

    def _plot_panel(ax, x, data, title, xlabel, annotate_text, show_ylabel: bool):
        mean_trace = data["mean"]
        err_trace = data["err"]
        if plot_error:
            ax.fill_between(x, mean_trace - err_trace, mean_trace + err_trace, color="gray", alpha=0.4, edgecolor="none")
        else:
            for t in data["all"]:
                ax.plot(x, t, color="grey", alpha=0.5, lw=1)
        ax.plot(x, mean_trace, color=trace_color, lw=5)
        ax.axvline(x=0, color="black", linestyle="--")
        ax.set_xlim(-float(window_size), float(window_size))
        if show_mode_titles:
            ax.set_title(title, fontsize=28)
        else:
            ax.set_title("")
        if show_xaxis_elements:
            ax.set_xlabel(xlabel, fontsize=28)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.set_xticks([])
        if show_ylabel:
            ax.set_ylabel("dF/F", fontsize=28)
        else:
            ax.set_yticks([])
        y_text = y_max_top * 0.6
        ax.annotate(
            annotate_text,
            xy=(0, y_text),
            xytext=(-3, y_text),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=28,
            fontstyle="italic",
        )
        if set_minmax:
            y_min_local, y_max_local = _safe_minmax([mean_trace])
            ax.set_ylim(y_min_local, y_max_local)
        else:
            ax.set_ylim(y_min_top, y_max_top)
        ax.tick_params(axis="y", which="major", labelsize=22)
        if show_xaxis_elements:
            ax.tick_params(axis="x", which="major", labelsize=22)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v: .1f}"))

    _plot_panel(ax0, per_cond_x["hunting"], aggregated["hunting"], "Hunting", "Time from \nhunting onset (s)", "Hunting\nonset", True)
    _plot_panel(ax1, per_cond_x["prey_sensory"], aggregated["prey_sensory"], "Prey sensory", "Time from \nprey onset (s)", "Prey\nonset", False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bottom = 0.28 if show_xaxis_elements else 0.14
    plt.subplots_adjust(left=0.10, right=0.98, bottom=bottom, top=0.88)
    plt.savefig(str(out_path), dpi=300)
    plt.close(fig)
    return str(out_path)


def event_average_loom_escape_multi_fish_from_pack(
    pack: PackPaths,
    *,
    fish_names: Sequence[str],
    set_id: str,
    average_ratio: Optional[Sequence[float]] = None,
    per_fish_component_specs: Optional[Dict[str, str]] = None,
    window_size: float = 30.0,
    trace_color: str = "deepskyblue",
    plot_error: bool = True,
    error_mode: str = "sem",
    set_minmax: bool = False,
    show_xaxis_elements: bool = True,
    show_mode_titles: bool = True,
    out_path: str | Path = "average.png",
) -> str:
    fish_names = [str(f) for f in fish_names]
    if not fish_names:
        raise ValueError("fish_names is empty.")
    if per_fish_component_specs is None:
        per_fish_component_specs = Loom_dict

    if average_ratio is None:
        fish_weights = np.full(len(fish_names), 1.0 / len(fish_names), dtype=float)
    else:
        fish_weights = np.asarray(list(average_ratio), dtype=float).reshape(-1)
        if fish_weights.shape[0] != len(fish_names):
            raise ValueError("average_ratio length does not match fish_names length.")
        if np.any(~np.isfinite(fish_weights)) or np.any(fish_weights < 0):
            raise ValueError("average_ratio must be finite and non-negative.")
        s = float(np.sum(fish_weights))
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"average_ratio must sum to 1 (got {s}).")

    bouts_info = _load_json(pack.behaviour / "bouts_info.json")
    conv_all = _load_json(pack.behaviour / "convergence periods" / "convergence_periods.json")

    periods, stimuli = get_stimulus_from_pack(pack, fish_names[0])
    looming_periods: List[List[float]] = []
    for (start, end), lab in zip(periods, stimuli):
        if str(lab).strip().lower() == "loom":
            looming_periods.append([float(start), float(end)])

    pooled_trace_groups: Dict[str, List[np.ndarray]] = {"loom_sensory": [], "escape": []}
    pooled_trace_weights: Dict[str, List[float]] = {"loom_sensory": [], "escape": []}

    for fish_idx, fish_name in enumerate(fish_names):
        if fish_name not in per_fish_component_specs:
            raise KeyError(f"Missing component spec for fish_name={fish_name!r}")
        component_number, component_index, intersection_mask_name = parse_component_spec(per_fish_component_specs[fish_name])

        fish_dir = pack.fish_dir(fish_name)
        time_stamps = np.load(str(fish_dir / "t.npy"))
        W = np.load(str(fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{component_number}.npy"))
        H = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{component_number}.npy"))
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        W_scaled = W * norms.T
        component_energies = np.abs(W_scaled)
        total_energies = np.sum(component_energies, axis=1, keepdims=True)
        total_energies = np.maximum(total_energies, 1e-12)
        FE_matrix = component_energies / total_energies
        selected_energy_ratio = FE_matrix[:, int(component_index)]

        intersection_mask = np.load(
            str(fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(component_number) / f"{intersection_mask_name}.npy")
        )
        intersection_mask = _load_filtered_intersection_mask(pack, fish_name, "Loom", intersection_mask, p_thr=0.05)
        masked_energy_ratio = selected_energy_ratio[intersection_mask]
        sort_indices = np.argsort(masked_energy_ratio)[::-1]

        set_ids = list(conv_all.get(fish_name, {}).keys())
        traces_loom: List[np.ndarray] = []
        traces_escape: List[np.ndarray] = []

        for set_index in set_ids:
            X = np.load(str(fish_dir / f"normalized_masked_dfof_{set_index}.npy"))
            X = np.asarray(X, dtype=float)
            X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)
            masked_X = X_norm[intersection_mask]
            sorted_masked_X = masked_X[sort_indices]

            bouts_info_temp = bouts_info.get(fish_name, {}).get(set_index, {})
            behavior_temp = np.asarray(bouts_info_temp.get("behaviour", []))
            bouts_temp = np.asarray(bouts_info_temp.get("bout", []), dtype=float)
            escape_indices = list(np.where(behavior_temp == "escape")[0])
            escape_periods = bouts_temp[escape_indices] / 300.0 if escape_indices else np.empty((0, 2), dtype=float)

            escape_bouts_in_loom: set[Tuple[float, float]] = set()
            for period in looming_periods:
                loom_start = float(period[0])
                loom_end = float(period[1])
                escape_overlapping = False
                for escape_period_item in escape_periods:
                    escape_start = float(escape_period_item[0])
                    escape_end = float(escape_period_item[1])
                    if max(loom_start, escape_start) < min(loom_end, escape_end):
                        escape_overlapping = True
                        escape_bouts_in_loom.add((escape_start, escape_end))
                        break
                start = loom_start - float(window_size)
                end = loom_start + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] < 2:
                    continue
                if not escape_overlapping:
                    traces_loom.append(slices)

            for escape_start, _escape_end in sorted(escape_bouts_in_loom):
                start = float(escape_start) - float(window_size)
                end = float(escape_start) + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] >= 2:
                    traces_escape.append(slices)

        fish_weight = float(fish_weights[fish_idx])
        if traces_loom:
            per_event_weight = fish_weight / float(len(traces_loom))
            pooled_trace_groups["loom_sensory"].extend(traces_loom)
            pooled_trace_weights["loom_sensory"].extend([per_event_weight] * len(traces_loom))
        if traces_escape:
            per_event_weight = fish_weight / float(len(traces_escape))
            pooled_trace_groups["escape"].extend(traces_escape)
            pooled_trace_weights["escape"].extend([per_event_weight] * len(traces_escape))

    aggregated: Dict[str, Dict[str, np.ndarray]] = {}
    per_cond_x: Dict[str, np.ndarray] = {}
    for key, trace_group in pooled_trace_groups.items():
        trace_weights = pooled_trace_weights[key]
        target_len = max(int(np.asarray(t).shape[1]) for t in trace_group) if trace_group else 1
        per_cond_x[key] = np.linspace(-float(window_size), float(window_size), target_len)
        mean_trace, err_trace, all_traces = _summarize_trace_group(trace_group, trace_weights, target_len=target_len, error_mode=error_mode)
        aggregated[key] = {"mean": mean_trace, "err": err_trace, "all": all_traces}

    y_min_top, y_max_top = _safe_minmax(
        [
            aggregated["loom_sensory"]["mean"] - aggregated["loom_sensory"]["err"],
            aggregated["loom_sensory"]["mean"] + aggregated["loom_sensory"]["err"],
            aggregated["escape"]["mean"] - aggregated["escape"]["err"],
            aggregated["escape"]["mean"] + aggregated["escape"]["err"],
        ]
    )

    fig_h = 4.8 if show_xaxis_elements else 4.0
    fig = plt.figure(figsize=(12.8, fig_h))
    gs = gridspec.GridSpec(1, 2, wspace=0.20, hspace=0.1)
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])

    def _plot_panel(ax, x, data, title, xlabel, annotate_text, show_ylabel: bool):
        mean_trace = data["mean"]
        err_trace = data["err"]
        if plot_error:
            ax.fill_between(x, mean_trace - err_trace, mean_trace + err_trace, color="gray", alpha=0.4, edgecolor="none")
        else:
            for t in data["all"]:
                ax.plot(x, t, color="grey", alpha=0.3, lw=1)
        ax.plot(x, mean_trace, color=trace_color, lw=5)
        ax.axvline(x=0, color="black", linestyle="--")
        ax.set_xlim(-float(window_size), float(window_size))
        if show_mode_titles:
            ax.set_title(title, fontsize=28)
        else:
            ax.set_title("")
        if show_xaxis_elements:
            ax.set_xlabel(xlabel, fontsize=28)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.set_xticks([])
        if show_ylabel:
            ax.set_ylabel("dF/F", fontsize=28)
        else:
            ax.set_yticks([])
        y_text = y_max_top * 0.6
        ax.annotate(
            annotate_text,
            xy=(0, y_text),
            xytext=(-3, y_text),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=28,
            fontstyle="italic",
        )
        if set_minmax:
            y_min_local, y_max_local = _safe_minmax([mean_trace])
            ax.set_ylim(y_min_local, y_max_local)
        else:
            ax.set_ylim(y_min_top, y_max_top)
        ax.tick_params(axis="y", which="major", labelsize=22)
        if show_xaxis_elements:
            ax.tick_params(axis="x", which="major", labelsize=22)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v: .1f}"))

    _plot_panel(ax0, per_cond_x["escape"], aggregated["escape"], "Escape", "Time from \nescape onset (s)", "Escape\nonset", True)
    _plot_panel(ax1, per_cond_x["loom_sensory"], aggregated["loom_sensory"], "Loom Sensory", "Time from \nloom onset (s)", "Loom\nonset", False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bottom = 0.28 if show_xaxis_elements else 0.14
    plt.subplots_adjust(left=0.10, right=0.98, bottom=bottom, top=0.88)
    plt.savefig(str(out_path), dpi=300)
    plt.close(fig)
    return str(out_path)


def event_average_locomotion_rois_across_events_multi_fish_from_pack(
    pack: PackPaths,
    *,
    fish_names: Sequence[str],
    set_id: str,
    average_ratio: Optional[Sequence[float]] = None,
    per_fish_locomotion_specs: Optional[Dict[str, str]] = None,
    window_size: float = 30.0,
    trace_color: str = "#0e4f8f",
    plot_error: bool = True,
    error_mode: str = "sem",
    p_threshold: float = 0.05,
    set_minmax: bool = False,
    show_xaxis_elements: bool = True,
    show_mode_titles: bool = True,
    out_path: str | Path = "average.png",
) -> str:
    fish_names = [str(f) for f in fish_names]
    if not fish_names:
        raise ValueError("fish_names is empty.")
    if per_fish_locomotion_specs is None:
        per_fish_locomotion_specs = locomtion_1_dict

    if average_ratio is None:
        fish_weights = np.full(len(fish_names), 1.0 / len(fish_names), dtype=float)
    else:
        fish_weights = np.asarray(list(average_ratio), dtype=float).reshape(-1)
        if fish_weights.shape[0] != len(fish_names):
            raise ValueError("average_ratio length does not match fish_names length.")
        if np.any(~np.isfinite(fish_weights)) or np.any(fish_weights < 0):
            raise ValueError("average_ratio must be finite and non-negative.")
        s = float(np.sum(fish_weights))
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"average_ratio must sum to 1 (got {s}).")

    bouts_info = _load_json(pack.behaviour / "bouts_info.json")
    conv_all = _load_json(pack.behaviour / "convergence periods" / "convergence_periods.json")

    periods, stimuli = get_stimulus_from_pack(pack, fish_names[0])
    prey_periods: List[List[float]] = []
    looming_periods: List[List[float]] = []
    omr_periods: List[List[float]] = []
    for (start, end), lab in zip(periods, stimuli):
        lab_s = str(lab).strip().lower()
        if lab_s == "prey":
            prey_periods.append([float(start), float(end)])
        if lab_s == "loom":
            looming_periods.append([float(start), float(end)])
        if "omr" in lab_s:
            omr_periods.append([float(start), float(end)])

    def _filter_intersection_by_p(fish_name: str, intersection_mask: np.ndarray) -> np.ndarray:
        if p_threshold is None:
            return np.asarray(intersection_mask, dtype=bool)
        return _load_filtered_intersection_mask(pack, fish_name, "locomotion", intersection_mask, p_thr=float(p_threshold))

    pooled_trace_groups: Dict[str, List[np.ndarray]] = {
        "hunting": [],
        "prey_sensory": [],
        "escape_swim": [],
        "loom_sensory": [],
        "omr_swim": [],
        "omr_sensory_no_bout": [],
    }
    pooled_trace_weights: Dict[str, List[float]] = {k: [] for k in pooled_trace_groups.keys()}

    for fish_idx, fish_name in enumerate(fish_names):
        if fish_name not in per_fish_locomotion_specs:
            raise KeyError(f"Missing locomotion spec for fish_name={fish_name!r}")
        component_number, component_index, intersection_mask_name = parse_component_spec(per_fish_locomotion_specs[fish_name])

        fish_dir = pack.fish_dir(fish_name)
        time_stamps = np.load(str(fish_dir / "t.npy"))
        W = np.load(str(fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{component_number}.npy"))
        H = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{component_number}.npy"))
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        W_scaled = W * norms.T
        component_energies = np.abs(W_scaled)
        total_energies = np.sum(component_energies, axis=1, keepdims=True)
        total_energies = np.maximum(total_energies, 1e-12)
        FE_matrix = component_energies / total_energies
        selected_energy_ratio = FE_matrix[:, int(component_index)]

        intersection_mask = np.load(
            str(fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(component_number) / f"{intersection_mask_name}.npy")
        )
        intersection_mask = _filter_intersection_by_p(fish_name, intersection_mask)
        masked_energy_ratio = selected_energy_ratio[intersection_mask]
        sort_indices = np.argsort(masked_energy_ratio)[::-1]

        set_ids = list(conv_all.get(fish_name, {}).keys())
        fish_traces: Dict[str, List[np.ndarray]] = {k: [] for k in pooled_trace_groups.keys()}

        for set_index in set_ids:
            X = np.load(str(fish_dir / f"normalized_masked_dfof_{set_index}.npy"))
            X = np.asarray(X, dtype=float)
            X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)
            masked_X = X_norm[intersection_mask]
            sorted_masked_X = masked_X[sort_indices]

            current_convergence_periods = conv_all.get(fish_name, {}).get(set_index, [])
            convergence_periods_seconds: List[List[float]] = []
            for c in current_convergence_periods:
                convergence_periods_seconds.append([float(c[0]) / 300.0, float(c[1]) / 300.0])

            for bout in convergence_periods_seconds:
                center_time = float(bout[0])
                start = center_time - float(window_size)
                end = center_time + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] >= 2:
                    fish_traces["hunting"].append(slices)

            for period in prey_periods:
                center_time = float(period[0])
                start = center_time - float(window_size)
                end = center_time + float(window_size)
                overlaps_hunting = False
                for hunt_start, hunt_end in convergence_periods_seconds:
                    if max(start, hunt_start) < min(end, hunt_end):
                        overlaps_hunting = True
                        break
                if overlaps_hunting:
                    continue
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] >= 2:
                    fish_traces["prey_sensory"].append(slices)

            bouts_info_temp = bouts_info.get(fish_name, {}).get(set_index, {})
            behavior_temp = np.asarray(bouts_info_temp.get("behaviour", []))
            bouts_temp = np.asarray(bouts_info_temp.get("bout", []), dtype=float)
            all_bouts_sec = bouts_temp / 300.0 if bouts_temp.size else np.empty((0, 2), dtype=float)
            escape_indices = list(np.where(behavior_temp == "escape")[0])
            escape_periods = bouts_temp[escape_indices] / 300.0 if escape_indices else np.empty((0, 2), dtype=float)

            escape_bouts_in_loom: set[Tuple[float, float]] = set()
            for period in looming_periods:
                loom_start = float(period[0])
                loom_end = float(period[1])
                escape_overlapping = False
                for escape_period_item in escape_periods:
                    escape_start = float(escape_period_item[0])
                    escape_end = float(escape_period_item[1])
                    if max(loom_start, escape_start) < min(loom_end, escape_end):
                        escape_overlapping = True
                        escape_bouts_in_loom.add((escape_start, escape_end))
                        break
                start = loom_start - float(window_size)
                end = loom_start + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] < 2:
                    continue
                if not escape_overlapping:
                    fish_traces["loom_sensory"].append(slices)

            for escape_start, _escape_end in sorted(escape_bouts_in_loom):
                start = float(escape_start) - float(window_size)
                end = float(escape_start) + float(window_size)
                slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                if slices.shape[1] >= 2:
                    fish_traces["escape_swim"].append(slices)

            try:
                omr_swim_bouts: List[Tuple[float, float]] = []
                for period in omr_periods:
                    omr_start = float(period[0])
                    omr_end = float(period[1])
                    for behavior_item, bout_item in zip(behavior_temp, bouts_temp):
                        behavior_label = str(behavior_item).lower()
                        is_omr_swim = ("turn left" in behavior_label) or ("turn right" in behavior_label) or ("slow swim" in behavior_label)
                        if not is_omr_swim:
                            continue
                        bout_start = float(bout_item[0]) / 300.0
                        bout_end = float(bout_item[1]) / 300.0
                        if omr_start <= bout_start <= omr_end:
                            omr_swim_bouts.append((bout_start, bout_end))
                if omr_swim_bouts:
                    omr_swim_bouts.sort(key=lambda x: x[0])
                    cluster_start = None
                    cluster_end = None
                    omr_event_centers: List[float] = []
                    for bout_start, bout_end in omr_swim_bouts:
                        if cluster_start is None:
                            cluster_start = bout_start
                            cluster_end = bout_end
                        elif bout_start - float(cluster_end) < float(window_size):
                            cluster_end = max(float(cluster_end), float(bout_end))
                        else:
                            omr_event_centers.append(float(cluster_start))
                            cluster_start = bout_start
                            cluster_end = bout_end
                    if cluster_start is not None:
                        omr_event_centers.append(float(cluster_start))
                    for center_time in omr_event_centers:
                        start = float(center_time) - float(window_size)
                        end = float(center_time) + float(window_size)
                        slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [start, end])
                        if slices.shape[1] >= 2:
                            fish_traces["omr_swim"].append(slices)
            except Exception:
                pass

            try:
                win_len = 2.0 * float(window_size)

                def _overlaps_any_bout(seg_start: float, seg_end: float) -> bool:
                    for bs, be in all_bouts_sec:
                        if max(seg_start, float(bs)) < min(seg_end, float(be)):
                            return True
                    return False

                chosen_segments: List[Tuple[float, float]] = []
                for period in omr_periods:
                    omr_start = float(period[0])
                    omr_end = float(period[1])
                    if omr_end - omr_start < win_len:
                        continue
                    s0 = omr_start
                    while s0 + win_len <= omr_end + 1e-9:
                        seg_start = float(s0)
                        seg_end = float(s0 + win_len)
                        if _overlaps_any_bout(seg_start, seg_end):
                            s0 += win_len
                            continue
                        overlaps_prev = any(max(seg_start, ps) < min(seg_end, pe) for ps, pe in chosen_segments)
                        if overlaps_prev:
                            s0 += win_len
                            continue
                        chosen_segments.append((seg_start, seg_end))
                        slices = slice_matrix_by_time(sorted_masked_X, time_stamps, [seg_start, seg_end])
                        if slices.shape[1] >= 2:
                            fish_traces["omr_sensory_no_bout"].append(slices)
                        s0 += win_len
            except Exception:
                pass

        fish_weight = float(fish_weights[fish_idx])
        for key, traces in fish_traces.items():
            if not traces:
                continue
            per_event_weight = fish_weight / float(len(traces))
            pooled_trace_groups[key].extend(traces)
            pooled_trace_weights[key].extend([per_event_weight] * len(traces))

    aggregated: Dict[str, Dict[str, np.ndarray]] = {}
    per_cond_x: Dict[str, np.ndarray] = {}
    for key, trace_group in pooled_trace_groups.items():
        trace_weights = pooled_trace_weights[key]
        target_len = max(int(np.asarray(t).shape[1]) for t in trace_group) if trace_group else 1
        if key == "omr_sensory_no_bout":
            per_cond_x[key] = np.linspace(0.0, 2.0 * float(window_size), target_len)
        else:
            per_cond_x[key] = np.linspace(-float(window_size), float(window_size), target_len)
        mean_trace, err_trace, all_traces = _summarize_trace_group(trace_group, trace_weights, target_len=target_len, error_mode=error_mode)
        aggregated[key] = {"mean": mean_trace, "err": err_trace, "all": all_traces}

    y_min_top, y_max_top = _safe_minmax(
        [
            aggregated["escape_swim"]["mean"] - aggregated["escape_swim"]["err"],
            aggregated["escape_swim"]["mean"] + aggregated["escape_swim"]["err"],
            aggregated["loom_sensory"]["mean"] - aggregated["loom_sensory"]["err"],
            aggregated["loom_sensory"]["mean"] + aggregated["loom_sensory"]["err"],
            aggregated["hunting"]["mean"] - aggregated["hunting"]["err"],
            aggregated["hunting"]["mean"] + aggregated["hunting"]["err"],
            aggregated["prey_sensory"]["mean"] - aggregated["prey_sensory"]["err"],
            aggregated["prey_sensory"]["mean"] + aggregated["prey_sensory"]["err"],
            aggregated["omr_swim"]["mean"] - aggregated["omr_swim"]["err"],
            aggregated["omr_swim"]["mean"] + aggregated["omr_swim"]["err"],
            aggregated["omr_sensory_no_bout"]["mean"] - aggregated["omr_sensory_no_bout"]["err"],
            aggregated["omr_sensory_no_bout"]["mean"] + aggregated["omr_sensory_no_bout"]["err"],
        ]
    )

    fig_h = 6.2 if show_xaxis_elements else 5.4
    fig = plt.figure(figsize=(40, fig_h))
    gs = gridspec.GridSpec(1, 6, wspace=0.16, hspace=0.0)
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[0, 3])
    ax4 = plt.subplot(gs[0, 4])
    ax5 = plt.subplot(gs[0, 5])

    def _plot_panel(ax, x, data, title, xlabel, annotate_text, centered_window: bool, show_onset: bool, show_ylabel: bool):
        mean_trace = data["mean"]
        err_trace = data["err"]
        if plot_error:
            ax.fill_between(x, mean_trace - err_trace, mean_trace + err_trace, color="gray", alpha=0.4, edgecolor="none")
        else:
            for t in data["all"]:
                ax.plot(x, t, color="grey", alpha=0.3, lw=1)
        ax.plot(x, mean_trace, color=trace_color, lw=5)
        if show_mode_titles:
            ax.set_title(title, fontsize=35)
        else:
            ax.set_title("")
        if centered_window:
            ax.axvline(x=0, color="black", linestyle="--")
            ax.set_xlim(-float(window_size), float(window_size))
        else:
            ax.set_xlim(0.0, 2.0 * float(window_size))
        if show_xaxis_elements:
            ax.set_xlabel(xlabel, fontsize=35)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.set_xticks([])
        if show_ylabel:
            ax.set_ylabel("dF/F", fontsize=35)
        else:
            ax.set_yticks([])
        if centered_window and show_onset:
            y_text = y_max_top * 0.6
            ax.annotate(
                annotate_text,
                xy=(0, y_text),
                xytext=(-3, y_text),
                arrowprops=dict(facecolor="black", shrink=0.05),
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=35,
                fontstyle="italic",
            )
        if set_minmax:
            y_min_local, y_max_local = _safe_minmax([mean_trace])
            ax.set_ylim(y_min_local, y_max_local)
        else:
            ax.set_ylim(y_min_top, y_max_top)
        ax.tick_params(axis="y", which="major", labelsize=29)
        if show_xaxis_elements:
            ax.tick_params(axis="x", which="major", labelsize=29)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v: .1f}"))

    _plot_panel(ax0, per_cond_x["hunting"], aggregated["hunting"], "Hunting", "Time from \nhunting onset (s)", "Hunting\nonset", True, True, True)
    _plot_panel(ax1, per_cond_x["prey_sensory"], aggregated["prey_sensory"], "Prey sensory", "Time from \nprey onset (s)", "Prey\nonset", True, True, False)
    _plot_panel(ax2, per_cond_x["escape_swim"], aggregated["escape_swim"], "Escape swim", "Time from \nescape onset (s)", "Escape\nonset", True, True, False)
    _plot_panel(ax3, per_cond_x["loom_sensory"], aggregated["loom_sensory"], "Loom sensory", "Time from \nloom onset (s)", "Loom\nonset", True, True, False)
    _plot_panel(ax4, per_cond_x["omr_swim"], aggregated["omr_swim"], "OMR", "Time from \nOMR onset (s)", "OMR\nonset", True, True, False)
    _plot_panel(ax5, per_cond_x["omr_sensory_no_bout"], aggregated["omr_sensory_no_bout"], "Optic flow", "Time in optic flow\nsensory window (s)", "", False, False, False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bottom = 0.30 if show_xaxis_elements else 0.24
    plt.subplots_adjust(left=0.06, right=0.99, bottom=bottom, top=0.90)
    plt.savefig(str(out_path), dpi=300)
    plt.close(fig)
    return str(out_path)


def Fig1_C_from_pack(
    pack: PackPaths,
    *,
    fish_name: str,
    set_id: Sequence[str] | str,
    time_range: Optional[Sequence[Sequence[float]] | Sequence[float]] = None,
    component_number: int,
    component_index: int,
    concat_sets: Optional[bool] = None,
    show_convergence_hatch: bool = True,
    convergence_fill_color: str = "magenta",
    convergence_fill_alpha: float = 0.3,
    convergence_shift_sec: float = -1.0,
    out_path: str | Path,
) -> str:
    bouts_info = _load_json(pack.behaviour / "bouts_info.json")
    passivity_json = _load_json(pack.behaviour / "passivity" / "passivity.json")
    convergence_periods_all = _load_json(pack.behaviour / "convergence periods" / "convergence_periods.json")

    if isinstance(set_id, str):
        set_ids = [set_id]
    else:
        set_ids = list(set_id)

    parsed_ranges: List[Tuple[float, float]] = []
    if time_range is None:
        pass
    elif isinstance(time_range, (list, tuple)):
        if len(time_range) > 0 and isinstance(time_range[0], (list, tuple)):
            for tr in time_range:
                parsed_ranges.append((float(tr[0]), float(tr[1])))
        elif len(time_range) == 2:
            parsed_ranges.append((float(time_range[0]), float(time_range[1])))

    plot_segments_specs: List[dict] = []
    if len(set_ids) == 1:
        curr_set = set_ids[0]
        if not parsed_ranges:
            plot_segments_specs.append({"set_id": curr_set, "range": None})
        else:
            for rng in parsed_ranges:
                plot_segments_specs.append({"set_id": curr_set, "range": rng})
    else:
        if not parsed_ranges:
            for s in set_ids:
                plot_segments_specs.append({"set_id": s, "range": None})
        else:
            for i, s in enumerate(set_ids):
                rng = parsed_ranges[i] if i < len(parsed_ranges) else None
                plot_segments_specs.append({"set_id": s, "range": rng})

    fish_dir = pack.fish_dir(fish_name)
    H_set1 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set001_{component_number}.npy"))
    H_set2 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set002_{component_number}.npy"))
    H_set3 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set003_{component_number}.npy"))
    H_set4 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set004_{component_number}.npy"))
    H = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    H_norm = H / norms
    H_norm_set1, H_norm_set2, H_norm_set3, H_norm_set4 = np.split(H_norm, 4, axis=1)

    periods, stimuli = get_stimulus_from_pack(pack, fish_name)
    omr_periods: List[Tuple[float, float]] = []
    loom_periods: List[Tuple[float, float]] = []
    for (start, end), label in zip(periods, stimuli):
        label_str = str(label).strip().lower()
        s = float(start)
        e = float(end)
        if "omr" in label_str:
            omr_periods.append((s, e))
        if "loom" in label_str:
            loom_periods.append((s, e))

    processed_segments: List[dict] = []
    for spec in plot_segments_specs:
        curr_set_id = spec["set_id"]

        tail_npz = pack.tail_npz(fish_name, curr_set_id)
        if not tail_npz.exists():
            raise FileNotFoundError(f"Missing tail_angles npz in package: {tail_npz}")
        tail_payload = np.load(str(tail_npz))
        t_tail = np.asarray(tail_payload["t_tail_sec"], dtype=float).reshape(-1)
        averaged_tail_angles = np.asarray(tail_payload["averaged_tail_angle_deg"], dtype=float).reshape(-1)

        bouts = np.asarray(bouts_info[fish_name][curr_set_id]["bout"])
        behavior = np.asarray(bouts_info[fish_name][curr_set_id]["behaviour"])
        passivity = np.asarray(passivity_json[fish_name][curr_set_id])
        conv_periods = convergence_periods_all.get(fish_name, {}).get(curr_set_id, [])
        convergence_periods_seconds: List[List[float]] = []
        for c in conv_periods:
            cs = float(c[0]) / 300.0 + float(convergence_shift_sec)
            ce = float(c[1]) / 300.0 + float(convergence_shift_sec)
            convergence_periods_seconds.append([cs, ce])

        if "001" in curr_set_id:
            selected_temporal_component = H_norm_set1[component_index]
        elif "002" in curr_set_id:
            selected_temporal_component = H_norm_set2[component_index]
        elif "003" in curr_set_id:
            selected_temporal_component = H_norm_set3[component_index]
        elif "004" in curr_set_id:
            selected_temporal_component = H_norm_set4[component_index]
        else:
            selected_temporal_component = H_norm_set1[component_index]

        t_img = np.arange(len(selected_temporal_component)) / 3.0
        max_t = max(float(t_tail[-1]) if len(t_tail) else 0.0, float(t_img[-1]) if len(t_img) else 0.0)
        if spec["range"] is None:
            x_min, x_max = 0.0, max_t
        else:
            x_min, x_max = spec["range"]

        processed_segments.append(
            {
                "set_id": curr_set_id,
                "range": (float(x_min), float(x_max)),
                "t_tail": t_tail,
                "averaged_tail_angles": averaged_tail_angles,
                "t_img": t_img,
                "selected_temporal_component": np.asarray(selected_temporal_component, dtype=float),
                "bouts": bouts,
                "behavior": behavior,
                "passivity": passivity,
                "convergence_periods_seconds": convergence_periods_seconds,
            }
        )

    base_font = 43
    axis_linewidth = 2.5
    tick_width = 2.5
    tick_length = 10
    if concat_sets is None:
        concat_sets = len(set_ids) > 1

    if concat_sets and len(processed_segments) > 1:
        from matplotlib.transforms import blended_transform_factory

        total_dur = 0.0
        for seg_data in processed_segments:
            s, e = seg_data["range"]
            total_dur += float(max(0.0, e - s))
        fig_w = max(24.0, min(60.0, 18.0 + total_dur / 30.0))
        fig_w = fig_w * 1.125

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(fig_w, 7), gridspec_kw={"height_ratios": [2, 3]})
        ax_tail, ax_tp = axs
        ax_tail.spines["top"].set_visible(False)
        ax_tp.spines["top"].set_visible(False)
        for _ax in (ax_tail, ax_tp):
            _ax.tick_params(axis="both", which="major", labelsize=base_font, width=tick_width, length=tick_length)
            for _sp in _ax.spines.values():
                _sp.set_linewidth(axis_linewidth)
        ax_tail.set_ylabel("Tail angle (°)", fontsize=base_font)
        ax_tp.set_ylabel("AU", fontsize=base_font)
        ax_tp.set_xlabel("Time (s)", fontsize=base_font, labelpad=25)

        trans_tail = blended_transform_factory(ax_tail.transData, ax_tail.transAxes)
        tail_tri_y = 1.08
        tri_size = 26

        offset = 0.0
        for seg_data in processed_segments:
            x_min, x_max = seg_data["range"]
            t_tail = seg_data["t_tail"]
            averaged_tail_angles = seg_data["averaged_tail_angles"]
            t_img = seg_data["t_img"]
            selected_temporal_component = seg_data["selected_temporal_component"]
            bouts = seg_data["bouts"]
            behavior = seg_data["behavior"]
            passivity = seg_data["passivity"]
            convergence_periods_seconds = seg_data.get("convergence_periods_seconds", [])

            tail_mask = (t_tail >= x_min) & (t_tail <= x_max)
            img_mask = (t_img >= x_min) & (t_img <= x_max)

            if np.any(tail_mask):
                ax_tail.plot(t_tail[tail_mask] - x_min + offset, averaged_tail_angles[tail_mask], color="darkslategrey", lw=2.5, zorder=3)
            if np.any(img_mask):
                ax_tp.plot(t_img[img_mask] - x_min + offset, selected_temporal_component[img_mask], color="black", lw=3, zorder=3)

            loom_onsets_plotted: set[float] = set()
            for ls, le in loom_periods:
                ls0 = float(ls)
                if x_min <= ls0 <= x_max:
                    key = round(ls0, 3)
                    if key in loom_onsets_plotted:
                        continue
                    loom_onsets_plotted.add(key)
                    x0 = ls0 - x_min + offset
                    ax_tail.plot(
                        [x0],
                        [tail_tri_y],
                        marker="v",
                        markersize=tri_size,
                        color="orange",
                        transform=trans_tail,
                        clip_on=False,
                        linestyle="None",
                        zorder=12,
                        markeredgecolor="none",
                        markeredgewidth=0,
                    )

            for cp in passivity:
                cs = float(cp[0])
                ce = float(cp[1])
                v_start = max(cs, x_min)
                v_end = min(ce, x_max)
                if v_end > v_start:
                    ax_tail.axvspan(v_start - x_min + offset, v_end - x_min + offset, color="#05830D", alpha=0.45, linewidth=0, zorder=0.5)
                if x_min <= cs <= x_max:
                    x0 = cs - x_min + offset
                    ax_tail.plot(
                        [x0],
                        [tail_tri_y],
                        marker="v",
                        markersize=tri_size,
                        color="#05830D",
                        transform=trans_tail,
                        clip_on=False,
                        linestyle="None",
                        zorder=10,
                        markeredgecolor="none",
                        markeredgewidth=0,
                    )

            if show_convergence_hatch and convergence_periods_seconds:
                for cs, ce in convergence_periods_seconds:
                    v_start = max(float(cs), x_min)
                    v_end = min(float(ce), x_max)
                    if v_end <= v_start:
                        continue
                    for _ax in (ax_tail, ax_tp):
                        _ax.axvspan(
                            v_start - x_min + offset,
                            v_end - x_min + offset,
                            facecolor=convergence_fill_color,
                            alpha=float(convergence_fill_alpha),
                            edgecolor="none",
                            linewidth=0.0,
                            zorder=1.0,
                        )

            for bout_item, behavior_item in zip(bouts, behavior):
                label = str(behavior_item).lower()
                is_struggle = "struggle" in label
                is_jturn = "j-turn" in label
                bs = float(bout_item[0]) / 300.0
                be = float(bout_item[1]) / 300.0 + 0.2
                v_start = max(bs, x_min)
                v_end = min(be, x_max)
                if v_end <= v_start:
                    continue
                in_omr_stimulus = any(os <= bs <= oe for os, oe in omr_periods)
                if is_struggle and in_omr_stimulus:
                    continue
                if is_struggle:
                    color_b = "purple"
                elif in_omr_stimulus:
                    color_b = "#0e4f8f"
                elif is_jturn:
                    color_b = "magenta"
                else:
                    continue
                x0 = bs - x_min + offset
                ax_tail.plot(
                    [x0],
                    [tail_tri_y],
                    marker="v",
                    markersize=tri_size,
                    color=color_b,
                    transform=trans_tail,
                    clip_on=False,
                    linestyle="None",
                    zorder=4,
                    markeredgecolor="none",
                    markeredgewidth=0,
                )

            offset += float(max(0.0, x_max - x_min))

        ax_tail.set_xlim(0.0, offset if offset > 0 else 1.0)
        ax_tp.xaxis.set_major_locator(mticker.MaxNLocator(6))
        fig.align_ylabels([ax_tail, ax_tp])
        plt.tight_layout(rect=[0, 0, 0.99, 1])
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    durations = []
    for seg in processed_segments:
        s, e = seg["range"]
        dur = abs(float(e) - float(s))
        durations.append(dur if dur > 1e-6 else 1e-6)

    fig = plt.figure(figsize=(18 * 1.125, 7))
    gs = fig.add_gridspec(2, len(processed_segments), height_ratios=[2, 3], width_ratios=durations, wspace=0.18, hspace=0.25)

    axs_grid: List[List[plt.Axes]] = []
    for r in range(2):
        row_axs: List[plt.Axes] = []
        for c in range(len(processed_segments)):
            sharey = row_axs[0] if c > 0 else None
            ax = fig.add_subplot(gs[r, c], sharey=sharey)
            row_axs.append(ax)
        axs_grid.append(row_axs)

    for i, seg_data in enumerate(processed_segments):
        x_min, x_max = seg_data["range"]
        t_tail = seg_data["t_tail"]
        averaged_tail_angles = seg_data["averaged_tail_angles"]
        t_img = seg_data["t_img"]
        selected_temporal_component = seg_data["selected_temporal_component"]
        bouts = seg_data["bouts"]
        behavior = seg_data["behavior"]
        passivity = seg_data["passivity"]
        convergence_periods_seconds = seg_data.get("convergence_periods_seconds", [])

        ax_tail = axs_grid[0][i]
        ax_tp = axs_grid[1][i]

        ax_tail.plot(t_tail, averaged_tail_angles, color="darkslategrey", lw=2.5)
        ax_tail.set_xlim(x_min, x_max)
        ax_tail.spines["top"].set_visible(False)
        ax_tail.tick_params(axis="both", which="major", labelsize=base_font, width=tick_width, length=tick_length)
        ax_tail.tick_params(axis="x", labelbottom=False)
        for _sp in ax_tail.spines.values():
            _sp.set_linewidth(axis_linewidth)
        if i == 0:
            ax_tail.set_ylabel("Tail angle (°)", fontsize=base_font)
        else:
            ax_tail.tick_params(axis="y", labelleft=False, left=False)
            ax_tail.spines["left"].set_visible(False)

        ax_tp.plot(t_img, selected_temporal_component, color="black", lw=3)
        ax_tp.set_xlim(x_min, x_max)
        ax_tp.spines["top"].set_visible(False)
        ax_tp.tick_params(axis="both", which="major", labelsize=base_font, width=tick_width, length=tick_length)
        ax_tp.xaxis.set_major_locator(mticker.MaxNLocator(4))
        for _sp in ax_tp.spines.values():
            _sp.set_linewidth(axis_linewidth)
        if i == 0:
            ax_tp.set_ylabel("AU", fontsize=base_font)
        else:
            ax_tp.tick_params(axis="y", labelleft=False, left=False)
            ax_tp.spines["left"].set_visible(False)

        target_axes = (ax_tail, ax_tp)

        loom_onsets_plotted: set[float] = set()
        for ls, le in loom_periods:
            cs = float(ls)
            if x_min <= cs <= x_max:
                key = round(cs, 3)
                if key in loom_onsets_plotted:
                    continue
                loom_onsets_plotted.add(key)
                ax_tail.plot(
                    [cs],
                    [1.03],
                    marker="v",
                    markersize=26,
                    color="orange",
                    transform=ax_tail.get_xaxis_transform(),
                    clip_on=False,
                    linestyle="None",
                    zorder=9,
                    markeredgecolor="none",
                    markeredgewidth=0,
                )

        for cp in passivity:
            cs = float(cp[0])
            ce = float(cp[1])
            v_start = max(cs, x_min)
            v_end = min(ce, x_max)
            if v_end > v_start:
                ax_tail.axvspan(v_start, v_end, color="#05830D", alpha=0.45, linewidth=0, zorder=0.6)
            if x_min <= cs <= x_max:
                ax_tail.plot(
                    [cs],
                    [1.03],
                    marker="v",
                    markersize=26,
                    color="#05830D",
                    transform=ax_tail.get_xaxis_transform(),
                    clip_on=False,
                    linestyle="None",
                    zorder=9,
                    markeredgecolor="none",
                    markeredgewidth=0,
                )

        if show_convergence_hatch and convergence_periods_seconds:
            for cs, ce in convergence_periods_seconds:
                v_start = max(float(cs), x_min)
                v_end = min(float(ce), x_max)
                if v_end <= v_start:
                    continue
                for _ax in target_axes:
                    _ax.axvspan(v_start, v_end, facecolor=convergence_fill_color, alpha=float(convergence_fill_alpha), edgecolor="none", linewidth=0.0, zorder=1.0)

        for bout_item, behavior_item in zip(bouts, behavior):
            label = str(behavior_item).lower()
            is_struggle = "struggle" in label
            is_jturn = "j-turn" in label
            bs = float(bout_item[0]) / 300.0
            be = float(bout_item[1]) / 300.0 + 0.2
            v_start = max(bs, x_min)
            v_end = min(be, x_max)
            if v_end <= v_start:
                continue
            in_omr_stimulus = any(os <= bs <= oe for os, oe in omr_periods)
            if is_struggle and in_omr_stimulus:
                continue
            if is_struggle:
                color_b = "purple"
            elif in_omr_stimulus:
                color_b = "#0e4f8f"
            elif is_jturn:
                color_b = "magenta"
            else:
                continue
            ax_tail.plot(
                [bs],
                [1.03],
                marker="v",
                markersize=26,
                color=color_b,
                transform=ax_tail.get_xaxis_transform(),
                clip_on=False,
                linestyle="None",
                zorder=9,
                markeredgecolor="none",
                markeredgewidth=0,
            )

        if len(processed_segments) > 1 and i < len(processed_segments) - 1:
            ax_tail.spines["right"].set_visible(False)
            ax_tp.spines["right"].set_visible(False)

    if len(processed_segments) > 0:
        ax_row0 = fig.add_subplot(gs[0, :], frameon=False)
        ax_row0.set_title("")
        ax_row0.set_xticks([])
        ax_row0.set_yticks([])
        ax_row1 = fig.add_subplot(gs[1, :], frameon=False)
        ax_row1.set_title("")
        ax_row1.set_xlabel("Time (s)", fontsize=base_font, labelpad=25)
        ax_row1.set_xticks([])
        ax_row1.set_yticks([])

    fig.align_ylabels([row[0] for row in axs_grid if len(row) > 0])
    plt.tight_layout(rect=[0, 0, 0.99, 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def Fig2_energy_ratio_hiearchy_from_pack(
    pack: PackPaths,
    *,
    fish_name: str,
    set_id: str,
    time_range: Optional[Sequence[Sequence[float]] | Sequence[float]] = None,
    component_number: int,
    component_index: int,
    mask_name: str,
    er_values: Optional[Sequence[float]] = None,
    out_path: str | Path,
) -> str:
    bouts_info = _load_json(pack.behaviour / "bouts_info.json")
    periods, stimuli = get_stimulus_from_pack(pack, fish_name)

    parsed_ranges: List[Tuple[float, float]] = []
    if time_range is None:
        pass
    elif isinstance(time_range, (list, tuple)):
        if len(time_range) > 0 and isinstance(time_range[0], (list, tuple)):
            for tr in time_range:
                parsed_ranges.append((float(tr[0]), float(tr[1])))
        elif len(time_range) == 2:
            parsed_ranges.append((float(time_range[0]), float(time_range[1])))

    fish_dir = pack.fish_dir(fish_name)
    W = np.load(str(fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{component_number}.npy"))
    H_set1 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set001_{component_number}.npy"))
    H_set2 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set002_{component_number}.npy"))
    H_set3 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set003_{component_number}.npy"))
    H_set4 = np.load(str(fish_dir / "Temporal_components_full_sets" / f"Temporal_components_prey-loom-omr-set004_{component_number}.npy"))
    H = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    H_norm = H / norms
    H_norm_set1, H_norm_set2, H_norm_set3, H_norm_set4 = np.split(H_norm, 4, axis=1)

    if "001" in set_id:
        H_current = H_norm_set1
    elif "002" in set_id:
        H_current = H_norm_set2
    elif "003" in set_id:
        H_current = H_norm_set3
    elif "004" in set_id:
        H_current = H_norm_set4
    else:
        H_current = H_norm_set1

    TP_index_1 = int(Hunting_dict[fish_name].split("component_")[1].split("_intersection")[0])
    TP_index_2 = int(Loom_dict[fish_name].split("component_")[1].split("_intersection")[0])
    TP1 = np.asarray(H_current[TP_index_1], dtype=float)
    TP2 = np.asarray(H_current[TP_index_2], dtype=float)

    intersection_mask = np.load(
        str(fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(component_number) / f"{mask_name}.npy")
    )

    W_scaled = W * norms.T
    component_energies = np.abs(W_scaled)
    total_energies = np.sum(component_energies, axis=1, keepdims=True)
    total_energies = np.maximum(total_energies, 1e-12)
    FE_matrix = component_energies / total_energies
    selected_energy_ratio = np.asarray(FE_matrix[:, int(component_index)], dtype=float)

    er_threshold = None
    _m = np.asarray(intersection_mask).astype(bool)
    if _m.shape[0] == selected_energy_ratio.shape[0]:
        _vals = selected_energy_ratio[_m]
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size:
            er_threshold = float(np.min(_vals))

    def _sample_indices_by_er_values(values: np.ndarray, targets: Sequence[float]) -> Tuple[List[int], List[float]]:
        values = np.asarray(values, dtype=float)
        valid_mask = np.isfinite(values)
        if not np.any(valid_mask):
            return [], []
        valid_indices = np.flatnonzero(valid_mask)
        valid_values = values[valid_mask]
        targets = [float(v) for v in targets]
        chosen_local: List[int] = []
        remaining = np.arange(valid_values.shape[0])
        for target in targets:
            if remaining.size == 0:
                break
            diffs = np.abs(valid_values[remaining] - float(target))
            local_choice = int(remaining[int(np.argmin(diffs))])
            chosen_local.append(local_choice)
            remaining = remaining[remaining != local_choice]
        chosen_global = valid_indices[np.array(chosen_local, dtype=int)]
        order = np.argsort(values[chosen_global])[::-1]
        chosen_global = chosen_global[order]
        chosen_values = values[chosen_global]
        return chosen_global.tolist(), chosen_values.tolist()

    if er_values is None:
        finite_vals = selected_energy_ratio[np.isfinite(selected_energy_ratio)]
        quantiles = (0.99, 0.95, 0.85, 0.7, 0.5, 0.25)
        er_values = np.quantile(finite_vals, quantiles).tolist() if finite_vals.size else [0.0]
    sampled_indices, y_positions = _sample_indices_by_er_values(selected_energy_ratio, targets=er_values)

    X = np.load(str(fish_dir / f"normalized_masked_dfof_{set_id}.npy"))
    X = np.asarray(X, dtype=float)
    X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X_norm = np.nan_to_num(X_norm)
    sampled_X = X_norm[np.asarray(sampled_indices, dtype=int)] if sampled_indices else np.empty((0, X_norm.shape[1]), dtype=float)
    t_img = np.arange(sampled_X.shape[1]) / 3.0 if sampled_X.size else np.arange(X_norm.shape[1]) / 3.0

    if parsed_ranges:
        x_min, x_max = parsed_ranges[0]
        time_mask = (t_img >= float(x_min)) & (t_img <= float(x_max))
        if np.any(time_mask):
            t_plot = t_img[time_mask]
            sampled_X_plot = sampled_X[:, time_mask] if sampled_X.size else np.empty((0, t_plot.shape[0]), dtype=float)
        else:
            t_plot = t_img
            sampled_X_plot = sampled_X
    else:
        t_plot = t_img
        sampled_X_plot = sampled_X

    valid_values = selected_energy_ratio[np.isfinite(selected_energy_ratio)]
    fig = plt.figure(figsize=(15, 10))

    from mpl_toolkits.axisartist.axislines import Subplot as AxisSubplot

    outer_gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 5.0], width_ratios=[3.0, 5.0], wspace=0.12, hspace=0.30)
    ax_blank = fig.add_subplot(outer_gs[0, 0], frameon=False)
    ax_blank.set_xticks([])
    ax_blank.set_yticks([])
    ax_blank.set_visible(False)

    ax_tp = fig.add_subplot(outer_gs[0, 1])
    ax_dist = AxisSubplot(fig, outer_gs[1, 0])
    fig.add_subplot(ax_dist)
    ax_trace = fig.add_subplot(outer_gs[1, 1], sharex=ax_tp)

    bins = 400
    y_min = 0.0
    y_max = float(np.max(valid_values)) if valid_values.size else 1.0
    y_grid = np.linspace(y_min, y_max, 4096)

    pdfs = None
    mixture_pdf = None
    if valid_values.size >= 3:
        try:
            gmm = GaussianMixture(n_components=4, random_state=0)
            gmm.fit(valid_values.reshape(-1, 1))
            weights = gmm.weights_
            means = gmm.means_.reshape(-1)
            variances = gmm.covariances_.reshape(-1)
            stds = np.sqrt(np.maximum(variances, 1e-12))
            order = np.argsort(means)
            weights, means, stds = weights[order], means[order], stds[order]
            pdfs = [float(w) * norm.pdf(y_grid, loc=float(m), scale=float(s)) for w, m, s in zip(weights, means, stds)]
            mixture_pdf = np.sum(pdfs, axis=0)
        except Exception:
            pdfs = None
            mixture_pdf = None

    _n_hist, _bins_hist, _patches_hist = ax_dist.hist(
        valid_values,
        bins=bins,
        range=(y_min, y_max),
        density=True,
        orientation="horizontal",
        alpha=1.0,
        color="grey",
        edgecolor="none",
    )

    if pdfs is not None:
        for _pdf in pdfs:
            ax_dist.plot(_pdf, y_grid, color="black", lw=1.5, linestyle="--", zorder=4)

    if er_threshold is not None:
        for _p in _patches_hist:
            _yc = float(_p.get_y()) + 0.5 * float(_p.get_height())
            if _yc > float(er_threshold):
                _p.set_facecolor("#4AC9FF")
            else:
                _p.set_facecolor("#808080")
        ax_dist.axhline(float(er_threshold), color="red", lw=2, zorder=10)

    ax_dist.invert_xaxis()

    y_positions = [float(v) for v in y_positions[: sampled_X_plot.shape[0]]] if sampled_X_plot.size else []
    t_tp = np.arange(TP1.shape[0]) / 3.0
    if parsed_ranges:
        x_min, x_max = parsed_ranges[0]
        tp_mask = (t_tp >= float(x_min)) & (t_tp <= float(x_max))
        if np.any(tp_mask):
            t_tp_plot = t_tp[tp_mask]
            tp1_plot = TP1[tp_mask]
            tp2_plot = TP2[tp_mask]
        else:
            t_tp_plot = t_tp
            tp1_plot = TP1
            tp2_plot = TP2
    else:
        t_tp_plot = t_tp
        tp1_plot = TP1
        tp2_plot = TP2

    tp1_plot = np.asarray(tp1_plot, dtype=float)
    tp2_plot = np.asarray(tp2_plot, dtype=float)
    tp1_plot = tp1_plot - float(np.nanmean(tp1_plot)) if tp1_plot.size else tp1_plot
    tp2_plot = tp2_plot - float(np.nanmean(tp2_plot)) if tp2_plot.size else tp2_plot
    tp1_std = float(np.nanstd(tp1_plot)) if tp1_plot.size else 1.0
    tp2_std = float(np.nanstd(tp2_plot)) if tp2_plot.size else 1.0
    tp1_std = tp1_std if np.isfinite(tp1_std) and tp1_std > 0 else 1.0
    tp2_std = tp2_std if np.isfinite(tp2_std) and tp2_std > 0 else 1.0
    tp1_plot = tp1_plot / tp1_std
    tp2_plot = tp2_plot / tp2_std
    tp_sep = 9.0
    ax_tp.plot(t_tp_plot, tp1_plot + tp_sep, color="black", lw=2)
    ax_tp.plot(t_tp_plot, tp2_plot, color="black", lw=2)
    ax_tp.set_yticks([])
    for side in ("top", "right", "left"):
        ax_tp.spines[side].set_visible(False)
    ax_tp.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_tp.set_xticks([])

    def _shade_hunting_and_loom(_ax, _x_min, _x_max):
        for (start, end), label in zip(periods, stimuli):
            if "loom" not in str(label).lower():
                continue
            cs = float(start)
            ce = float(end)
            v_start = max(cs, float(_x_min))
            v_end = min(ce, float(_x_max))
            if v_end > v_start:
                _ax.axvspan(v_start, v_end, color="orange", alpha=0.35, linewidth=0, zorder=0.5)
        try:
            bouts = bouts_info[fish_name][set_id]["bout"]
            behavior = bouts_info[fish_name][set_id]["behaviour"]
            targeted_behaviors = {"J-turn right"}
            for interval, btype in zip(bouts, behavior):
                if str(btype) not in targeted_behaviors:
                    continue
                s_sec = float(interval[0]) / 300.0 - 1.0
                e_sec = float(interval[1]) / 300.0 + 0.2
                v_start = max(s_sec, float(_x_min))
                v_end = min(e_sec, float(_x_max))
                if v_end > v_start:
                    _ax.axvspan(v_start, v_end, color="magenta", alpha=0.5, linewidth=1, zorder=0.6)
        except Exception:
            pass

    x_min_plot = float(t_plot[0]) if len(t_plot) else 0.0
    x_max_plot = float(t_plot[-1]) if len(t_plot) else 0.0
    _shade_hunting_and_loom(ax_tp, x_min_plot, x_max_plot)

    ref_trace = sampled_X_plot[0] if sampled_X_plot.shape[0] > 0 else np.array([0.0])
    ref_trace = ref_trace[np.isfinite(ref_trace)]
    if ref_trace.size == 0:
        ref_vmin, ref_vmax = 0.0, 1.0
    else:
        ref_vmin = float(np.min(ref_trace))
        ref_vmax = float(np.max(ref_trace))
    ref_center = 0.5 * (ref_vmin + ref_vmax)
    ref_half_range = 0.5 * (ref_vmax - ref_vmin)
    if not np.isfinite(ref_half_range) or ref_half_range <= 0:
        ref_half_range = 1.0

    diffs = np.diff(np.sort(np.asarray(y_positions, dtype=float))) if len(y_positions) > 1 else np.array([])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    gap = float(np.min(diffs)) if diffs.size else (float(np.max(valid_values)) if valid_values.size else 1.0)
    desired_half_height_y = max(1e-6, 0.45 * gap)

    ax_dist.set_ylim(0.0, 0.11)
    ax_trace.set_ylim(0.0, 0.11)

    def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
        s = str(hex_color).lstrip("#")
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)

    c_top = np.array(_hex_to_rgb01("#4AC9FF"), dtype=float)
    c_bot = np.array(_hex_to_rgb01("#808080"), dtype=float)
    first_trace_color = tuple(c_top)
    for i in range(int(sampled_X_plot.shape[0])):
        if er_threshold is not None:
            color_i = tuple(c_top) if float(y_positions[i]) > float(er_threshold) else tuple(c_bot)
        else:
            er_min = float(np.min(y_positions)) if y_positions else 0.0
            er_max = float(np.max(y_positions)) if y_positions else 1.0
            denom = (er_max - er_min) if er_max > er_min else 1.0
            tt = ((er_max - float(y_positions[i])) / denom) ** 1.0 if i < len(y_positions) else 0.0
            tt = float(np.clip(tt, 0.0, 1.0))
            color_i = tuple((1.0 - tt) * c_top + tt * c_bot)
        if i == 0:
            first_trace_color = color_i
        y0 = float(y_positions[i])
        ax_trace.plot(t_plot, ((sampled_X_plot[i] - ref_center) / ref_half_range) * desired_half_height_y + y0, color=color_i, lw=2)

    _shade_hunting_and_loom(ax_trace, x_min_plot, x_max_plot)

    x_peak = float(np.max(mixture_pdf)) if mixture_pdf is not None else None
    if x_peak is not None and np.isfinite(x_peak) and x_peak > 0:
        ax_dist.set_xlim(x_peak * 1.15, 0)

    selected_count = None
    unselected_count = None
    try:
        _m_count = np.asarray(intersection_mask).astype(bool).reshape(-1)
        _n_total = int(selected_energy_ratio.shape[0])
        if _m_count.shape[0] == _n_total:
            selected_count = int(np.count_nonzero(_m_count))
            unselected_count = int(_n_total - selected_count)
    except Exception:
        selected_count = None
        unselected_count = None

    if selected_count is not None and unselected_count is not None:
        try:
            pos = ax_dist.get_position()
            w = float(0.66 * pos.width)
            x0 = float(pos.x0 + 0.5 * (pos.width - w))
            h_total = float(0.64 * pos.height)
            y0 = float(pos.y1 - h_total + 0.03 * pos.height)
            ax_cnt = fig.add_axes([x0, y0, w, h_total])
            ax_cnt.set_facecolor("none")
            for _s in ("top", "right", "left", "bottom"):
                ax_cnt.spines[_s].set_visible(True)
            ax_cnt.tick_params(top=False, labeltop=False, right=False, labelright=False)
            fig.text(x0 - 0.06 * pos.width, y0 + 0.5 * h_total, "Count", rotation=90, va="center", ha="right", fontsize=20)
            x = np.array([0.0, 1.0], dtype=float)
            heights = np.array([float(unselected_count), float(selected_count)], dtype=float)
            colors = [tuple(c_bot), first_trace_color]
            ax_cnt.bar(x, heights, color=colors, width=0.55, edgecolor="none")
            ax_cnt.set_xlim(-0.5, 1.5)
            ax_cnt.set_xticks([0.0, 1.0])
            ax_cnt.set_xticklabels(["Other\nROIs", "Hunting\nROIs"], fontsize=20)
            ax_cnt.tick_params(axis="y", labelsize=20)
            ax_cnt.set_ylim(0.0, float(np.max(heights)) * 1.08 if np.max(heights) > 0 else 1.0)
        except Exception:
            pass

    ax_dist.axis["top"].set_visible(False)
    ax_dist.axis["left"].set_visible(False)
    ax_dist.axis["right"].set_visible(True)
    ax_dist.axis["right"].major_ticks.set_visible(False)
    ax_dist.axis["right"].major_ticklabels.set_visible(False)
    ax_dist.axis["right"].label.set_visible(False)
    ax_dist.axis["bottom"].major_ticks.set_visible(False)
    ax_dist.axis["bottom"].major_ticklabels.set_visible(False)
    ax_dist.axis["bottom"].label.set_visible(False)
    ax_dist.set_xticks([])
    ax_dist.set_yticks([])
    ax_dist.axis["right"].set_axisline_style("-|>", size=3)
    ax_dist.text(0.92, 0.88, "Energy ratio", transform=ax_dist.transAxes, rotation=90, va="center", ha="left", fontsize=20)
    ax_dist.text(0.5, -0.02, "Density", transform=ax_dist.transAxes, va="top", ha="center", fontsize=20)

    ax_trace.spines["top"].set_visible(True)
    ax_trace.spines["right"].set_visible(True)
    ax_trace.set_yticks([])
    ax_trace.set_xlabel("Time (s)", fontsize=20)
    try:
        if parsed_ranges:
            _xmin, _xmax = parsed_ranges[0]
            ax_tp.set_xlim(float(_xmin), float(_xmax))
            ax_trace.set_xlim(float(_xmin), float(_xmax))
        else:
            ax_tp.set_xlim(left=0.0)
            ax_trace.set_xlim(left=0.0)
    except Exception:
        pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def package_data(
    *,
    out_dir: str | Path,
    data_root: str | Path,
    fish_names: Optional[Sequence[str]] = None,
) -> str:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    pack = PackPaths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bouts_info_src = data_root / "behaviour" / "bouts_info.json"
    bouts_info = _filter_bouts_info(_load_json(bouts_info_src))
    default_fish = sorted(
        {
            *Hunting_dict.keys(),
            *Passivity_dict.keys(),
            *Struggle_dict.keys(),
            *Loom_dict.keys(),
            *locomtion_1_dict.keys(),
            *locomtion_2_dict.keys(),
        }
    )
    default_fish = [f for f in default_fish if _is_allowed_fish_id(f)]
    if fish_names is None:
        fish_names_use = [f for f in default_fish if f in bouts_info]
    else:
        fish_names_use = [str(f) for f in fish_names if _is_allowed_fish_id(str(f))]
        bouts_info = {k: v for k, v in bouts_info.items() if k in fish_names_use}

    _save_json(pack.behaviour / "bouts_info.json", bouts_info)

    passivity_src = data_root / "behaviour" / "passivity" / "passivity.json"
    convergence_src = data_root / "behaviour" / "convergence periods" / "convergence_periods.json"
    passivity_all = _load_json(passivity_src)
    conv_all = _load_json(convergence_src)
    _save_json(pack.behaviour / "passivity" / "passivity.json", {k: passivity_all.get(k, {}) for k in fish_names_use})
    _save_json(pack.behaviour / "convergence periods" / "convergence_periods.json", {k: conv_all.get(k, {}) for k in fish_names_use})

    stim_src = data_root / "behaviour" / "10 fish data" / "stimulus_info.xlsx"
    stim_dst = pack.behaviour / "10 fish data" / "stimulus_info.xlsx"
    _ensure_parent(stim_dst)
    shutil.copy2(str(stim_src), str(stim_dst))

    pvals_src_dir = data_root / "imaging" / "p_vals_final"
    pvals_dst_dir = pack.pvals
    pvals_dst_dir.mkdir(parents=True, exist_ok=True)

    modes_needed = {"Hunting", "Loom", "Passivity", "Struggle", "locomotion"}
    for fish_name in fish_names_use:
        for mode in modes_needed:
            src_inbrain = pvals_src_dir / f"{fish_name}_{mode}_inbrain.npy"
            src_pvals = pvals_src_dir / f"{fish_name}_{mode}_p_vals.npy"
            if src_inbrain.exists():
                shutil.copy2(str(src_inbrain), str(pvals_dst_dir / src_inbrain.name))
            if src_pvals.exists():
                shutil.copy2(str(src_pvals), str(pvals_dst_dir / src_pvals.name))

    specs_by_fish: Dict[str, List[str]] = {str(f): [] for f in fish_names_use}
    for mapping in (Hunting_dict, Passivity_dict, Struggle_dict, Loom_dict, locomtion_1_dict, locomtion_2_dict):
        for fish, spec in mapping.items():
            if fish in specs_by_fish:
                specs_by_fish[fish].append(str(spec))
    if "20250807-F3" in specs_by_fish:
        specs_by_fish["20250807-F3"].extend(["90_component_16_intersection_3_4", "90_component_26_intersection_2_3"])
    for fish in list(specs_by_fish.keys()):
        uniq = sorted(set([s for s in specs_by_fish[fish] if s]))
        specs_by_fish[fish] = uniq

    missing_files: List[str] = []

    for fish_name in fish_names_use:
        src_fish_dir = data_root / "imaging" / "Registered 10-fish" / fish_name
        dst_fish_dir = pack.fish_dir(fish_name)
        dst_fish_dir.mkdir(parents=True, exist_ok=True)

        t_src = src_fish_dir / "t.npy"
        if t_src.exists():
            shutil.copy2(str(t_src), str(dst_fish_dir / "t.npy"))

        spec_list = list(specs_by_fish.get(fish_name, []))

        component_numbers = set()
        mask_names: List[Tuple[int, str]] = []
        for s in spec_list:
            cn, _, mask = parse_component_spec(s)
            component_numbers.add(int(cn))
            mask_names.append((int(cn), str(mask)))

        for cn in sorted(component_numbers):
            src_W = src_fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{cn}.npy"
            dst_W = dst_fish_dir / "Spatial_components_full_sets" / f"Spatial_components_{cn}.npy"
            if src_W.exists():
                _ensure_parent(dst_W)
                shutil.copy2(str(src_W), str(dst_W))
            else:
                missing_files.append(str(src_W))

            for set_id in ("prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set003", "prey-loom-omr-set004"):
                src_H = src_fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{cn}.npy"
                dst_H = dst_fish_dir / "Temporal_components_full_sets" / f"Temporal_components_{set_id}_{cn}.npy"
                if src_H.exists():
                    _ensure_parent(dst_H)
                    shutil.copy2(str(src_H), str(dst_H))
                else:
                    missing_files.append(str(src_H))

            for set_id in ("prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set003", "prey-loom-omr-set004"):
                src_X = src_fish_dir / f"normalized_masked_dfof_{set_id}.npy"
                dst_X = dst_fish_dir / f"normalized_masked_dfof_{set_id}.npy"
                if src_X.exists():
                    _ensure_parent(dst_X)
                    shutil.copy2(str(src_X), str(dst_X))
                else:
                    missing_files.append(str(src_X))

        for cn, mask in mask_names:
            src_mask = src_fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(cn) / f"{mask}.npy"
            dst_mask = dst_fish_dir / "intersection_masks_full_sets_Energy ratio_time_corrected" / str(cn) / f"{mask}.npy"
            if src_mask.exists():
                _ensure_parent(dst_mask)
                shutil.copy2(str(src_mask), str(dst_mask))
            else:
                missing_files.append(str(src_mask))

    for fish_name in fish_names_use:
        for set_id in ("prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set003", "prey-loom-omr-set004"):
            h5_path = data_root / "behaviour" / fish_name / "TOPCAMERA" / set_id / f"{fish_name.replace('-', '_')}_Trial1.h5"
            if not h5_path.exists():
                missing_files.append(str(h5_path))
                continue
            df_tail = pd.read_hdf(str(h5_path), "tail")
            df_eye = pd.read_hdf(str(h5_path), "eye")
            heading = df_eye["heading"].values
            xy = df_tail.values[:, ::2] + df_tail.values[:, 1::2] * 1j
            midline = -np.exp(1j * np.deg2rad(np.asarray(heading)))
            tail_angles = -np.angle(np.diff(xy, axis=1) / midline[:, None])
            tail_angles = tail_angles[:, 5:]
            averaged_tail_angles = np.degrees(np.average(tail_angles, axis=1))
            t_tail = np.arange(averaged_tail_angles.shape[0]) / 300.0
            out_npz = pack.tail_npz(fish_name, set_id)
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_npz), t_tail_sec=t_tail.astype(float), averaged_tail_angle_deg=averaged_tail_angles.astype(float))

    for fish_name in fish_names_use:
        for mode in modes_needed:
            req_inbrain = pvals_dst_dir / f"{fish_name}_{mode}_inbrain.npy"
            req_pvals = pvals_dst_dir / f"{fish_name}_{mode}_p_vals.npy"
            if not req_inbrain.exists():
                missing_files.append(str(pvals_src_dir / req_inbrain.name))
            if not req_pvals.exists():
                missing_files.append(str(pvals_src_dir / req_pvals.name))

    missing_files = sorted(set([p for p in missing_files if p]))
    if missing_files:
        raise FileNotFoundError("Missing required files for packaging:\n" + "\n".join(missing_files))

    manifest = {
        "fish_names": fish_names_use,
        "specs": {
            "Hunting_dict": {k: v for k, v in Hunting_dict.items() if k in fish_names_use},
            "Passivity_dict": {k: v for k, v in Passivity_dict.items() if k in fish_names_use},
            "Struggle_dict": {k: v for k, v in Struggle_dict.items() if k in fish_names_use},
            "Loom_dict": {k: v for k, v in Loom_dict.items() if k in fish_names_use},
            "locomtion_1_dict": {k: v for k, v in locomtion_1_dict.items() if k in fish_names_use},
            "locomtion_2_dict": {k: v for k, v in locomtion_2_dict.items() if k in fish_names_use},
        },
    }
    _save_json(out_dir / "manifest.json", manifest)
    return str(out_dir)


def make_figures(*, data_dir: str | Path, out_dir: str | Path) -> List[str]:
    pack = PackPaths(Path(data_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []

    example_dir = out_dir / "example_tps"
    example_dir.mkdir(parents=True, exist_ok=True)

    misha_fish = "20250807-F3"
    outputs.append(
        Example_TPs_from_pack(
            pack,
            fish_name=misha_fish,
            set_id=["prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002"],
            time_range=[[0, 50], [400, 450], [230, 280], [65, 115], [470, 520]],
            component_number=90,
            component_index=76,
            mode="Hunting",
            notable_label=["None", "passivity", "Struggle", "Loom", "locomotion"],
            show_scale_bar=False,
            out_path=example_dir / f"{misha_fish}_Hunting_Example_TPs.png",
        )
    )
    outputs.append(
        Example_TPs_from_pack(
            pack,
            fish_name=misha_fish,
            set_id=["prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002"],
            time_range=[[0, 50], [400, 450], [230, 280], [65, 115], [470, 520]],
            component_number=90,
            component_index=7,
            mode="Passivity",
            notable_label=["None", "passivity", "Struggle", "Loom", "locomotion"],
            show_scale_bar=False,
            out_path=example_dir / f"{misha_fish}_Passivity_Example_TPs.png",
        )
    )
    outputs.append(
        Example_TPs_from_pack(
            pack,
            fish_name=misha_fish,
            set_id=["prey-loom-omr-set001", "prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002"],
            time_range=[[120, 170], [400, 450], [230, 280], [65, 115], [470, 520]],
            component_number=90,
            component_index=18,
            mode="Struggle",
            notable_label=["None", "passivity", "Struggle", "Loom", "locomotion"],
            show_scale_bar=False,
            out_path=example_dir / f"{misha_fish}_Struggle_Example_TPs.png",
        )
    )
    outputs.append(
        Example_TPs_from_pack(
            pack,
            fish_name=misha_fish,
            set_id=["prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002"],
            time_range=[[0, 50], [400, 450], [230, 280], [65, 115], [470, 520]],
            component_number=90,
            component_index=42,
            mode="Loom",
            notable_label=["None", "passivity", "Struggle", "Loom", "locomotion"],
            show_scale_bar=True,
            out_path=example_dir / f"{misha_fish}_Loom_Example_TPs.png",
        )
    )
    outputs.append(
        Example_TPs_from_pack(
            pack,
            fish_name=misha_fish,
            set_id=["prey-loom-omr-set001", "prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set002"],
            time_range=[[0, 50], [400, 450], [230, 280], [65, 115], [470, 520]],
            component_number=90,
            component_index=16,
            mode="Locomotion",
            notable_label=["None", "passivity", "Struggle", "Loom", "locomotion"],
            show_scale_bar=False,
            out_path=example_dir / f"{misha_fish}_Locomotion_Example_TPs.png",
        )
    )
    outputs.append(
        concat_example_tps_vertical(
            [
                example_dir / f"{misha_fish}_Hunting_Example_TPs.png",
                example_dir / f"{misha_fish}_Locomotion_Example_TPs.png",
                example_dir / f"{misha_fish}_Passivity_Example_TPs.png",
                example_dir / f"{misha_fish}_Struggle_Example_TPs.png",
                example_dir / f"{misha_fish}_Loom_Example_TPs.png",
            ],
            example_dir / "Example_hunting_TPs_vertical_concat_misha.png",
            gap=20,
        )
        or ""
    )

    hunting_tp_dir = out_dir / "example_tps_hunting"
    hunting_tp_dir.mkdir(parents=True, exist_ok=True)
    hunting_specs = [
        ("20250807-F1", ["prey-loom-omr-set001", "prey-loom-omr-set003", "prey-loom-omr-set001", "prey-loom-omr-set004", "prey-loom-omr-set001"],
         [[10, 60], [407.5, 457.5], [305, 355], [185, 235], [562, 612]], 90, 29, True),
        ("20250807-F2", ["prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set001", "prey-loom-omr-set001", "prey-loom-omr-set002"],
         [[10, 60], [517, 567], [663, 713], [65, 115], [511, 561]], 90, 50, True),
        ("20250807-F3", ["prey-loom-omr-set002", "prey-loom-omr-set002", "prey-loom-omr-set002", "prey-loom-omr-set002", "prey-loom-omr-set002"],
         [[10, 60], [443, 493], [225, 275], [185, 235], [470, 520]], 90, 76, True),
        ("20250807-F4", ["prey-loom-omr-set001", "prey-loom-omr-set002", "prey-loom-omr-set003", "prey-loom-omr-set001", "prey-loom-omr-set001"],
         [[10, 60], [414, 464], [25, 75], [185, 235], [293, 343]], 90, 61, True),
        ("20250808-F1", ["prey-loom-omr-set001", "prey-loom-omr-set003", "prey-loom-omr-set001", "prey-loom-omr-set001", "prey-loom-omr-set003"],
         [[10, 60], [405, 455], [579, 629], [185, 235], [401, 451]], 70, 30, False),
    ]
    hunting_tp_paths = []
    for fish_name, set_ids, ranges, cn, ci, show_scale in hunting_specs:
        outp = hunting_tp_dir / f"{fish_name}_Hunting_Example_TPs.png"
        hunting_tp_paths.append(outp)
        outputs.append(
            Example_TPs_from_pack(
                pack,
                fish_name=fish_name,
                set_id=set_ids,
                time_range=ranges,
                component_number=cn,
                component_index=ci,
                mode="Hunting",
                notable_label=["None", "passivity", "Struggle", "Loom", "locomotion"],
                show_scale_bar=bool(show_scale),
                out_path=outp,
            )
        )
    outputs.append(
        concat_example_tps_vertical(
            hunting_tp_paths,
            hunting_tp_dir / "Example_hunting_TPs_vertical_concat.png",
            gap=0,
        )
        or ""
    )

    avg_dir = out_dir / "event_average"
    avg_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    out_paths.append(
        event_average_5conditions_multi_fish_from_pack(
            pack,
            fish_names=["20250807-F1", "20250807-F2", "20250807-F3", "20250807-F4", "20250808-F1"],
            mode="Hunting",
            trace_color="#4AC9FF",
            per_fish_component_specs=Hunting_dict,
            set_id="prey-loom-omr-set001",
            show_xaxis_elements=False,
            show_mode_titles=True,
            out_path=avg_dir / "multi_fish_Hunting_5cond.png",
        )
    )
    out_paths.append(
        event_average_5conditions_multi_fish_from_pack(
            pack,
            fish_names=["20250807-F1", "20250807-F2", "20250807-F3", "20250807-F4", "20250808-F1"],
            mode="Locomotion",
            trace_color="#0e5fab",
            per_fish_component_specs=locomtion_1_dict,
            set_id="prey-loom-omr-set001",
            show_xaxis_elements=False,
            show_mode_titles=False,
            out_path=avg_dir / "multi_fish_Locomotion_5cond.png",
        )
    )
    out_paths.append(
        event_average_5conditions_multi_fish_from_pack(
            pack,
            fish_names=["20250807-F1", "20250807-F3", "20250807-F4", "20250808-F1"],
            mode="Passivity",
            trace_color="#05830D",
            per_fish_component_specs=Passivity_dict,
            set_id="prey-loom-omr-set001",
            show_xaxis_elements=False,
            show_mode_titles=False,
            out_path=avg_dir / "multi_fish_Passivity_5cond.png",
        )
    )
    out_paths.append(
        event_average_5conditions_multi_fish_from_pack(
            pack,
            fish_names=["20250807-F2", "20250807-F3", "20250808-F1"],
            mode="Struggle",
            trace_color="purple",
            per_fish_component_specs=Struggle_dict,
            set_id="prey-loom-omr-set001",
            show_xaxis_elements=False,
            show_mode_titles=False,
            out_path=avg_dir / "multi_fish_Struggle_5cond.png",
        )
    )
    out_paths.append(
        event_average_5conditions_multi_fish_from_pack(
            pack,
            fish_names=["20250807-F1", "20250807-F2", "20250807-F3", "20250807-F4", "20250808-F1"],
            mode="Loom",
            trace_color="orange",
            per_fish_component_specs=Loom_dict,
            set_id="prey-loom-omr-set001",
            show_xaxis_elements=True,
            show_mode_titles=False,
            out_path=avg_dir / "multi_fish_Loom_5cond.png",
        )
    )
    outputs.extend(out_paths)
    outputs.append(concat_event_average_vertical(out_paths, avg_dir / "multi_fish_vertical_concat.png") or "")

    overlap_dir = out_dir / "overlap_scatter"
    overlap_dir.mkdir(parents=True, exist_ok=True)
    res = hunting_overlap_and_energy_ratio_distributions_single_fish_from_pack(
        pack,
        fish_name="20250807-F3",
        output_dir=overlap_dir,
        default_xlim=(0, 0.06),
        default_ylim=(0, 0.06),
        axis_limits={
            "Loom": {"xlim": (0, 0.1), "ylim": (0, 0.2)},
            "Passivity": {"xlim": (0, 0.15), "ylim": (0, 0.25)},
            "Struggle": {"xlim": (0, 0.15), "ylim": (0, 0.2)},
            "Locomotion": {"xlim": (0, 0.1), "ylim": (0, 0.25)},
        },
    )
    outputs.extend([v.get("figure_path", "") for v in res.values() if isinstance(v, dict)])

    avg2_dir = out_dir / "event_average_2condition"
    avg2_dir.mkdir(parents=True, exist_ok=True)
    outputs.append(
        event_average_hunting_prey_sensory_multi_fish_from_pack(
            pack,
            fish_names=list(Hunting_dict.keys()),
            set_id="prey-loom-omr-set001",
            per_fish_component_specs=Hunting_dict,
            window_size=20,
            trace_color="#4AC9FF",
            plot_error=True,
            error_mode="sem",
            set_minmax=False,
            show_xaxis_elements=True,
            show_mode_titles=True,
            out_path=avg2_dir / "Hunting_PreySensory_multi_fish.png",
        )
    )
    outputs.append(
        event_average_loom_escape_multi_fish_from_pack(
            pack,
            fish_names=list(Loom_dict.keys()),
            set_id="prey-loom-omr-set001",
            per_fish_component_specs=Loom_dict,
            window_size=20,
            trace_color="orange",
            plot_error=True,
            error_mode="sem",
            set_minmax=False,
            show_xaxis_elements=True,
            show_mode_titles=True,
            out_path=avg2_dir / "LoomSensory_Escape_multi_fish.png",
        )
    )

    loco_dir = out_dir / "event_average_locomotion_6events"
    loco_dir.mkdir(parents=True, exist_ok=True)
    outputs.append(
        event_average_locomotion_rois_across_events_multi_fish_from_pack(
            pack,
            fish_names=["20250807-F1", "20250807-F2", "20250807-F3", "20250807-F4", "20250808-F1"],
            set_id="prey-loom-omr-set001",
            per_fish_locomotion_specs=locomtion_1_dict,
            window_size=20,
            trace_color="#0e4f8f",
            plot_error=True,
            error_mode="sem",
            p_threshold=0.05,
            set_minmax=False,
            show_xaxis_elements=True,
            show_mode_titles=True,
            out_path=loco_dir / "LocomotionROIs_6events_multi_fish.png",
        )
    )

    fig1_dir = out_dir / "Fig1"
    fig1_dir.mkdir(parents=True, exist_ok=True)
    outputs.append(
        Fig1_C_from_pack(
            pack,
            fish_name="20250807-F2",
            set_id=["prey-loom-omr-set001", "prey-loom-omr-set002"],
            time_range=[(0, 450), (450, 650)],
            component_number=90,
            component_index=50,
            concat_sets=True,
            show_convergence_hatch=True,
            out_path=fig1_dir / "Fig1C.png",
        )
    )

    fig2_dir = out_dir / "Fig2A"
    fig2_dir.mkdir(parents=True, exist_ok=True)
    outputs.append(
        Fig2_energy_ratio_hiearchy_from_pack(
            pack,
            fish_name="20250807-F2",
            set_id="prey-loom-omr-set001",
            time_range=[[0, 240]],
            component_number=90,
            component_index=50,
            mask_name="component_50_intersection_1_2",
            er_values=[0.1, 0.075, 0.050, 0.025],
            out_path=fig2_dir / "Fig2A.png",
        )
    )

    outputs = [p for p in outputs if p]
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=Path(__file__).name)
    sub = p.add_subparsers(dest="command", required=True)

    p_pack = sub.add_parser("package-data")
    p_pack.add_argument("--out", required=True, help="Output package directory.")
    p_pack.add_argument("--data-root", default=None, help="Original dataset root (default: env ASTROCYTE_DATA_ROOT).")
    p_pack.add_argument("--fish", nargs="*", default=None, help="Optional fish IDs to include (must start with 20250807/20250808).")

    p_fig = sub.add_parser("make-figures")
    p_fig.add_argument("--data", required=True, help="Package directory produced by package-data.")
    p_fig.add_argument("--out", required=True, help="Output figure directory.")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "package-data":
        data_root = _data_root_from_env(args.data_root)
        package_data(out_dir=args.out, data_root=data_root, fish_names=args.fish)
    elif args.command == "make-figures":
        make_figures(data_dir=args.data, out_dir=args.out)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
