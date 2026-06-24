# Astroglia-paper
This repository contains the code for the paper "Astrocytes Rapidly Initiate Goal-Directed Visual Behavior"
## Installation
Clone the repository by running the following command:
```bash
git clone https://github.com/SemmelhackLab/Astroglia-paper.git
```

Navigate to the repository directory:
```bash
cd Astroglia-paper
```

# Demo: Astroglia NMF analysis pipeline

This repository contains a runnable demo (run_demo.py) that reproduces the full analysis pipeline for a single fish (default: `20250807-F1`), starting from the astroglia activity matrix (cells × time) and ending with:

- Rastermap visualization of the original activity matrix
- Hunting-related temporal pattern (TP) identification (shuffle-significance pipeline)
- Astroglia selectivity quantification via energy ratio
- GMM-based selection of the most selective astroglia
- 5-condition event-aligned average traces (trace color uses the hunting color)

You must download the required demo data `demo_data_F1_20250807-F2.zip` in `https://www.dropbox.com/scl/fi/7wkruxep2qujr1va386va/demo_data_F1_20250807-F2.zip?rlkey=z9gln3p8cose6qepwx39zh5kr&st=bwl8h4r8&dl=0`, and put it in the same directory with `run_demo.py` before you run the code.

## What the demo does

1. Load 4 activity matrices (`normalized_masked_dfof_*`) and concatenate them (cells × concatenated time).
2. Visualize the original activity matrix using **rastermap** sorting.
3. Identify the **hunting TP** by fully replicating the shuffle-significance logic:
   - shuffle-based per-timepoint significance
   - event-window mask built from hunting (convergence) onsets
   - low-variability ratio `std(background) / mean(|event|)`
   - objective thresholding for the ratio to select the hunting TP
4. Compute **energy ratio** selectivity of every astroglia to the selected hunting TP.
5. Use **GMM** to select the most selective astroglia.
6. Plot **5-condition event-averaged traces** (Hunting / Passivity / Locomotion / Loom / Struggle).
7. Plot the hunting TP trace in **black**, and mark each hunting onset with a **magenta upward triangle** below the x-axis.

## Files

- Demo script: `run_demo.py`
- Demo data: `demo_data_F1_20250807-F2/`
- Demo outputs: `demo_outputs_20250807-F2/`

## How to run

Run from the project root:

```bash
python run_demo.py
```

The default fish is `20250807-F2`.

## Dependencies (auto-checked / auto-installed)

The script checks required Python packages at startup and installs missing ones using:

```bash
python -m pip install --user --disable-pip-version-check <package>
```

Required packages:

- `numpy`
- `matplotlib`
- `scikit-learn` (imported as `sklearn`)
- `rastermap`

If your environment cannot install packages automatically (e.g., restricted permissions or no internet), the demo raises a clear error message including the pip stdout/stderr.

## Outputs

After running, a folder is created:

- `demo_outputs_<fish_name>/`

Key output files:

- `activity_matrix_rastermap.png` — rastermap view of the concatenated activity matrix
- `hunting_tp_trace.png` — hunting TP trace (black) + magenta onset markers + legend
- `tp_low_variability_ratio_hist.png` — ratio distribution + objective GMM intersection threshold
- `energy_ratio_hist.png` — energy ratio distribution for the selected TP
- `event_average_5_conditions.png` — 5-condition event-aligned average traces
- `energy_ratio.npy`, `gmm_selected_mask.npy`, `selected_tp_index.npy` — saved intermediate results

## Useful environment variables

- `FISH_NAME` (default: `20250807-F2`)
- `RECOMPUTE_NMF` (`0/1`, default: `0`)  
  If `1`, recomputes NMF from the concatenated activity matrix instead of loading precomputed W/H from the demo data folder.
- `TP_WINDOW_S` (default: `10`)  
  Event window half-width (seconds) used in the hunting TP shuffle-significance pipeline.
- `SHUFFLE_ALPHA` (default: `0.05`)  
  Significance level for shuffle-based testing.
- `SHUFFLE_APPLY_FDR` (`0/1`, default: `0`)  
  If `1`, applies Benjamini–Hochberg FDR across timepoints for each TP.


