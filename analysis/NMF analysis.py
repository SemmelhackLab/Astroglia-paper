"""
NMF Analysis and Visualization for Astrocyte Imaging Data

This script performs Non-Negative Matrix Factorization (NMF) analysis on astrocyte imaging data,
including data loading, preprocessing, NMF execution, component visualization, and GMM fitting
for threshold detection.

Organized for GitHub submission.
"""

import os
import json
import math
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import tifffile

from pathlib import Path
from h5py import File
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.signal import butter, filtfilt
from scipy.stats import norm
from scipy.optimize import brentq
from PIL import Image

from sklearn.decomposition import NMF
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from rastermap import Rastermap
from skimage.exposure import rescale_intensity
from kneed import KneeLocator
from matplotlib.patches import Patch

# Constants for file paths
BASE_DIR = os.environ.get("ASTROCYTE_DATA_ROOT", r"D:/2p astrocyte")
BEHAVIOUR_DIR = os.path.join(BASE_DIR, "behaviour")
IMAGING_DIR = os.path.join(BASE_DIR, "imaging")
REGISTERED_FISH_DIR = os.path.join(IMAGING_DIR, "Registered 10-fish")

BOUTS_INFO_PATH = os.path.join(BEHAVIOUR_DIR, "bouts_info.json")
LABELS_CSV_PATH = os.path.join(BEHAVIOUR_DIR, "labels.csv")
STIMULUS_INFO_10_FISH = os.path.join(BEHAVIOUR_DIR, "10 fish data/stimulus_info.xlsx")
STIMULUS_INFO_TEST = os.path.join(BEHAVIOUR_DIR, "2p-test-20250718-info.xlsx")
CONVERGENCE_PERIODS_PATH = os.path.join(BEHAVIOUR_DIR, "convergence periods/convergence_periods.json")
PASSIVITY_PATH = os.path.join(BEHAVIOUR_DIR, "passivity/passivity.json")

# Constants for visualization
DISTINCT_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
]


def load_bouts_info():
    """Load bouts info from JSON file."""
    try:
        with open(BOUTS_INFO_PATH, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Bouts info file not found: {BOUTS_INFO_PATH}")
        return {}


# Load global bouts info
bouts_info = load_bouts_info()
if bouts_info:
    fish_names = list(bouts_info.keys())
    behaviors = []
    for fish in fish_names:
        set_ids = list(bouts_info[fish].keys())
        for set_id in set_ids:
            behaviors.extend(bouts_info[fish][set_id]['behaviour'])
    unique_behaviors = np.unique(behaviors)
    behavior_color_map = {b: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] 
                          for i, b in enumerate(unique_behaviors)}
else:
    behavior_color_map = {}
    unique_behaviors = []


def low_pass_filt(x, fs, cutoff, axis=0, order=2):
    """
    Apply a low-pass Butterworth filter to the data.

    Args:
        x (np.ndarray): Input data.
        fs (float): Sampling frequency.
        cutoff (float): Cutoff frequency.
        axis (int): Axis along which to apply the filter.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered data.
    """
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=axis)


def load_data(fish_dir, snr_threshold=2):
    """
    Load and preprocess imaging and behavioral data for a specific fish.

    Args:
        fish_dir (str): Directory containing fish data.
        snr_threshold (float): Signal-to-noise ratio threshold for selecting neurons.

    Returns:
        tuple: (t, dfof, df_neurons, df_sessions, slices, contours)
    """
    timestamps = np.load(os.path.join(fish_dir, "timestamps.npy"))

    cnmf_paths = sorted(Path(os.path.join(fish_dir, "cnmf")).glob("plane*-channel0-refit.hdf5"))
    
    if '20250718-F1' not in fish_dir:
        n_neurons = [File(i)["estimates"]["nr"][()] for i in cnmf_paths]
    else:
        n_neurons = [len(File(i)["estimates"]["S"][()]) for i in cnmf_paths]
        
    plane_ids = [int(i.stem.split("-")[0][-2:]) for i in cnmf_paths]
    df_neurons = pd.DataFrame(
        {
            "snr": np.load(os.path.join(fish_dir, "snr.npy")),
            "plane_id": np.repeat(plane_ids, n_neurons),
        }
    )

    accepted = df_neurons["snr"].ge(snr_threshold)
    df_neurons = df_neurons[accepted]

    contours = np.load(os.path.join(fish_dir, "contours.npy"), allow_pickle=True)[accepted]

    with File(os.path.join(fish_dir, "dfof.hdf5"), "r") as f:
        dfof = f["data"][accepted]

    df_sessions = pd.read_csv(
        Path(os.path.join(fish_dir, "sessions.txt")), names=["name", "n_frames"], sep=" "
    )

    n_frames = df_sessions["n_frames"].values
    cumsum = np.cumsum(n_frames)
    slices = [np.s_[..., a:b] for a, b in zip(cumsum - n_frames, cumsum)]
    t = np.arange(3, 780 * 3) / 3

    # Interpolate dfof to align with common time base
    dfof = np.concatenate(
        [
            np.concatenate(
                [
                    interp1d(
                        timestamps[i][s],
                        dfof[df_neurons["plane_id"].eq(i)][s],
                        kind="cubic",
                    )(t)
                    for s in slices
                ],
                axis=1,
            )
            for i in range(len(timestamps))
        ]
    )

    return t, dfof, df_neurons, df_sessions, slices, contours


def get_stimulus(fish_name):
    """
    Load stimulus periods and labels for a given fish.

    Args:
        fish_name (str): Name of the fish (e.g., '20250718-F1').

    Returns:
        tuple: (periods, stimuli)
               periods: List of [start, end] time ranges.
               stimuli: List of stimulus names.
    """
    if fish_name in ['20250718-F1', '20250718-F3', '20250718-F4']:
        info_path = STIMULUS_INFO_TEST
        df = pd.read_excel(info_path, engine="openpyxl")
        
        start = df["start"].tolist()
        end = df["end"].tolist()
        stimuli = df["stimuli"].tolist()

        periods = [[start[i], end[i]] for i in range(len(stimuli))]
        periods.append([450, 510])
        periods.append([750, 810])
    else:
        info_path = STIMULUS_INFO_10_FISH
        df = pd.read_excel(info_path, engine="openpyxl")

        start = df["start"].tolist()
        end = df["end"].tolist()
        stimuli = df["stimuli"].tolist()

        periods = [[start[i], end[i]] for i in range(len(stimuli))]

    return periods, stimuli


def process_annotations():
    """
    Process behavioral annotations from CSV and update the bouts info JSON file.
    """
    # fish_names = os.listdir(IMAGING_DIR) # Unused
    csv_file = pd.read_csv(LABELS_CSV_PATH)

    bouts = csv_file['bout']
    new_bouts = []
    for bout in bouts:
        start = int(bout.split(', ')[0].replace('[', ''))
        end = int(bout.split(', ')[1].replace(']', ''))
        new_bouts.append([start, end])

    original_video_names = csv_file['original file name']
    new_video_names = []
    new_set_names = []
    for name in original_video_names:
        new_video_names.append(name.split('_')[0])
        new_set_names.append(name.split('_')[1])

    labels = csv_file['behavior']
    new_labels = []
    for label in labels:
        if 'left swim' in label:
            label = label.replace('left swim', 'turn')
        if 'right swim' in label:
            label = label.replace('right swim', 'turn')
        if 'forward swim' in label:
            label = label.replace('forward swim', 'slow swim')

        new_labels.append(label)

    # Initialize bouts_info structure (hardcoded for specific fish as per original code)
    bouts_info_local = {
        '20250718-F1': {'prey-loom-omr-set001': {'bout': [], 'behaviour': []}},
        '20250718-F3': {'prey-loom-omr-set001': {'bout': [], 'behaviour': []},
                        'prey-loom-omr-set002': {'bout': [], 'behaviour': []}},
        '20250718-F4': {'prey-loom-omr-set001': {'bout': [], 'behaviour': []},
                        'prey-loom-omr-set002': {'bout': [], 'behaviour': []}}
    }

    for i in range(len(new_video_names)):
        if new_video_names[i] in bouts_info_local and new_set_names[i] in bouts_info_local[new_video_names[i]]:
             bouts_info_local[new_video_names[i]][new_set_names[i]]['bout'].append(new_bouts[i])
             bouts_info_local[new_video_names[i]][new_set_names[i]]['behaviour'].append(new_labels[i])

    with open(BOUTS_INFO_PATH, "w") as json_file:
        json.dump(bouts_info_local, json_file, indent=4)

    return bouts_info_local


def prepare_dfof(fish_name):
    """
    Preprocess dfof data for NMF analysis: normalize, mask by SNR, split by sets,
    and fit Rastermap model.

    Args:
        fish_name (str): Name of the fish.
    """
    fish_dir = os.path.join(REGISTERED_FISH_DIR, fish_name) + '/'

    t, dfof, df_neurons, df_sessions, slices, contours = load_data(fish_dir)
    np.save(fish_dir + "t.npy", t)

    snr = np.load(fish_dir + "snr.npy")
    mask = snr > 2
    np.save(fish_dir + "snr_mask.npy", mask)

    # Split into 4 sets
    split_idx = int(dfof.shape[1] / 4)
    dfof_set1 = dfof[:, :split_idx]
    dfof_set2 = dfof[:, split_idx: 2 * split_idx]
    dfof_set3 = dfof[:, 2 * split_idx: 3 * split_idx]
    dfof_set4 = dfof[:, 3 * split_idx: 4 * split_idx]

    np.save(fish_dir + "masked_dfof_prey-loom-omr-set001.npy", dfof_set1)
    np.save(fish_dir + "masked_dfof_prey-loom-omr-set002.npy", dfof_set2)
    np.save(fish_dir + "masked_dfof_prey-loom-omr-set003.npy", dfof_set3)
    np.save(fish_dir + "masked_dfof_prey-loom-omr-set004.npy", dfof_set4)

    # Fit Rastermap
    concatenated_dfof = np.concatenate((dfof_set1, dfof_set2, dfof_set3, dfof_set4), axis=1)
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5).fit(concatenated_dfof)

    X = model.X
    min_vals = X.min(axis=1, keepdims=True)
    max_vals = X.max(axis=1, keepdims=True)

    # Apply Min-Max normalization
    X = (X - min_vals) / (max_vals - min_vals)

    X_set1 = X[:, :split_idx]
    X_set2 = X[:, split_idx: 2 * split_idx]
    X_set3 = X[:, 2 * split_idx: 3 * split_idx]
    X_set4 = X[:, 3 * split_idx: 4 * split_idx]

    np.save(fish_dir + "normalized_masked_dfof_prey-loom-omr-set001.npy", X_set1)
    np.save(fish_dir + "normalized_masked_dfof_prey-loom-omr-set002.npy", X_set2)
    np.save(fish_dir + "normalized_masked_dfof_prey-loom-omr-set003.npy", X_set3)
    np.save(fish_dir + "normalized_masked_dfof_prey-loom-omr-set004.npy", X_set4)


def run_dfof_preprocess():
    """Run dfof preprocessing for all registered fish."""
    if not os.path.exists(REGISTERED_FISH_DIR):
        print(f"Directory not found: {REGISTERED_FISH_DIR}")
        return

    fs = os.listdir(REGISTERED_FISH_DIR)
    for fish_name in fs:
        prepare_dfof(fish_name)


def NMF_on_dfof(fish_name, component_number):
    """
    Run NMF on the concatenated dfof data for a specific fish.

    Args:
        fish_name (str): Name of the fish.
        component_number (int): Number of NMF components.
    """
    fish_dir = os.path.join(REGISTERED_FISH_DIR, fish_name)
    
    dfof_set1 = np.load(os.path.join(fish_dir, "normalized_masked_dfof_prey-loom-omr-set001.npy"))
    dfof_set2 = np.load(os.path.join(fish_dir, "normalized_masked_dfof_prey-loom-omr-set002.npy"))
    dfof_set3 = np.load(os.path.join(fish_dir, "normalized_masked_dfof_prey-loom-omr-set003.npy"))
    dfof_set4 = np.load(os.path.join(fish_dir, "normalized_masked_dfof_prey-loom-omr-set004.npy"))

    dfof = np.concatenate((dfof_set1, dfof_set2, dfof_set3, dfof_set4), axis=1)

    model = NMF(n_components=component_number, init='random', random_state=42)
    W = model.fit_transform(dfof)  # Fit to processed matrix
    H = model.components_

    H_split = np.split(H, 4, axis=1)

    spatial_dir = os.path.join(fish_dir, "Spatial_components_full_sets")
    temporal_dir = os.path.join(fish_dir, "Temporal_components_full_sets")

    if not os.path.exists(spatial_dir):
        os.makedirs(spatial_dir)
    if not os.path.exists(temporal_dir):
        os.makedirs(temporal_dir)

    np.save(os.path.join(spatial_dir, f"Spatial_components_{component_number}.npy"), W)

    for i in range(len(H_split)):
        np.save(os.path.join(temporal_dir, f"Temporal_components_prey-loom-omr-set00{i+1}_{component_number}.npy"),
                H_split[i])


def get_tail_angles(df_tail, heading):
    """
    Calculate tail angles from tail coordinates and heading.

    Args:
        df_tail (pd.DataFrame): Dataframe containing tail coordinates.
        heading (np.ndarray): Heading values.

    Returns:
        np.ndarray: Calculated tail angles.
    """
    xy = df_tail.values[:, ::2] + df_tail.values[:, 1::2] * 1j
    midline = -np.exp(1j * np.deg2rad(np.asarray(heading)))
    return -np.angle(np.diff(xy, axis=1) / midline[:, None])


def plot_components(fish_name, component_number=None):
    """
    Plot temporal components along with behavioral data (eye angles, tail angles).

    Args:
        fish_name (str): Name of the fish.
        component_number (int): Number of components used in NMF.
    """
    fish_path = os.path.join(REGISTERED_FISH_DIR, fish_name)
    temporal_dir = os.path.join(fish_path, "Temporal_components_full_sets")

    sets_data = []
    for i in range(1, 5):
        sets_data.append(np.load(os.path.join(temporal_dir, 
            f"Temporal_components_prey-loom-omr-set00{i}_{component_number}.npy")))

    set_names = ['prey-loom-omr-set001', 'prey-loom-omr-set002', 
                 'prey-loom-omr-set003', 'prey-loom-omr-set004']

    for idx, set_name in enumerate(set_names):
        T_set = sets_data[idx]

        h5_path = os.path.join(BEHAVIOUR_DIR, f"{fish_name}/TOPCAMERA/{set_name}/{fish_name.replace('-', '_')}_Trial1.h5")
        
        try:
            df_eye = pd.read_hdf(h5_path, "eye")
            df_tail = pd.read_hdf(h5_path, "tail")
        except FileNotFoundError:
            print(f"HDF5 file not found: {h5_path}")
            continue

        tail_angles = get_tail_angles(df_tail, df_eye["heading"].values)
        averaged_tail_angles = np.average(tail_angles[:, 5:], axis=1)
        averaged_tail_angles = np.degrees(averaged_tail_angles)

        eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
        eye_angles_filt = low_pass_filt(eye_angles, 300, 4)
        left_angle = eye_angles_filt[:, 0]
        right_angle = eye_angles_filt[:, 1]

        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            convergence_periods = json.load(file)

        current_convergence_periods = convergence_periods.get(fish_name, {}).get(set_name, [])

        bouts = bouts_info.get(fish_name, {}).get(set_name, {}).get('bout', [])
        behavior = bouts_info.get(fish_name, {}).get(set_name, {}).get('behaviour', [])
        
        new_bouts = []
        for bout in bouts:
            start = (bout[0] / 300) * 3
            end = (bout[1] / 300) * 3
            new_bouts.append([start, end])

        periods, stimuli = get_stimulus(fish_name)

        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        passivity = np.array(passivity_json.get(fish_name, {}).get(set_name, []))

        save_base_path = os.path.join(fish_path, "Temporal components plots (full sets)", 
                                      str(component_number), set_name)
        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)

        for c in range(component_number):
            selected_t = T_set[c]

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 8))

            # Plot first trace: Stimulus
            for (start, end), label in zip(periods, stimuli):
                start = start * 3
                end = end * 3
                color = 'grey' if label == 'omr stationary' else 'black'
                ax1.hlines(y=0, xmin=start, xmax=end, color=color, linewidth=4)
                
                center = (start + end) / 2
                ax1.text(center, 0.02, label, ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), fontsize=18)

            if fish_name in ['20250718-F1', '20250718-F4']:
                # Special handling for specific fish extra periods
                 if len(periods) > 9:
                    ax1.hlines(y=0, xmin=periods[8][0] * 3, xmax=periods[8][1] * 3, color='lightgrey', linewidth=4)
                    ax1.hlines(y=0, xmin=periods[9][0] * 3, xmax=periods[9][1] * 3, color='lightgrey', linewidth=4)

            ax1.set_xlim(0, selected_t.shape[0] + 3)
            ax1.set_xticks([])
            ax1.axis('off')

            # Plot second trace: Eye angles
            behavior_time_axis = np.arange(len(averaged_tail_angles)) / 300.0
            ax2.plot(behavior_time_axis, left_angle, color='blue', alpha=0.6, label='Left Eye Angle')
            ax2.plot(behavior_time_axis, right_angle, color='red', alpha=0.6, label='Right Eye Angle')
            ax2.set_xlim(0, len(averaged_tail_angles) / 300.0)

            if current_convergence_periods:
                for i, conv_period in enumerate(current_convergence_periods):
                    conv_start = conv_period[0] / 300
                    conv_end = conv_period[1] / 300
                    ax2.axvspan(conv_start, conv_end, color='yellow', alpha=0.3,
                                    label='Convergence Period' if i == 0 else "")

            # Plot third trace: Tail angles and bouts
            ax3.plot(behavior_time_axis, averaged_tail_angles, color='darkslategrey', label='Tail Angle')
            for i, (start, end) in enumerate(new_bouts):
                if i < len(behavior):
                    ax3.axvspan(start / 3, end / 3, color=behavior_color_map.get(behavior[i], 'gray'), alpha=0.8)

            ax3.tick_params(axis='y', labelcolor='black')
            ax3.set_xticks([])
            ax3.spines['top'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)

            if passivity.size > 0:
                for j in range(passivity.shape[0]):
                    start = passivity[j][0]
                    end = passivity[j][1]
                    ax3.axvspan(start, end, color='papayawhip', alpha=0.6, label='Passivity period' if j == 0 else "")

            ax3.set_xlim(0, len(averaged_tail_angles) / 300.0)

            # Plot fourth trace: Temporal component
            ax4.plot(selected_t, color='black', label='Temporal component')
            ax4.set_title(f'Temporal component {c}')
            ax4.set_xlim(0, selected_t.shape[0])

            # Add common labels and adjust layout
            plt.tight_layout()
            fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=12)
            fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=12)

            plt.savefig(os.path.join(save_base_path, f'component_{c}.png'), dpi=300)
            plt.close()


def concatenate_T_plots(fish_name, component_number):
    """
    Concatenate temporal component plots from 4 sets into a single image.

    Args:
        fish_name (str): Name of the fish.
        component_number (int): Number of components.
    """
    fish_path = os.path.join(REGISTERED_FISH_DIR, fish_name)
    base_plot_path = os.path.join(fish_path, "Temporal components plots (full sets)", str(component_number))

    for c in range(component_number):
        image_paths = [
            os.path.join(base_plot_path, f'prey-loom-omr-set001/component_{c}.png'),
            os.path.join(base_plot_path, f'prey-loom-omr-set002/component_{c}.png'),
            os.path.join(base_plot_path, f'prey-loom-omr-set003/component_{c}.png'),
            os.path.join(base_plot_path, f'prey-loom-omr-set004/component_{c}.png'),
        ]

        # Check if all images exist
        if not all(os.path.exists(p) for p in image_paths):
            print(f"Missing images for component {c}, skipping concatenation.")
            continue

        images = [Image.open(path) for path in image_paths]

        # Ensure all images are same size
        widths, heights = zip(*(img.size for img in images))
        min_width = min(widths)
        min_height = min(heights)

        resized_images = []
        for img in images:
            if img.size != (min_width, min_height):
                img = img.resize((min_width, min_height), Image.Resampling.LANCZOS)
            resized_images.append(img)

        # Create 2x2 grid
        grid_size = (2, 2)
        new_image = Image.new('RGB', (min_width * grid_size[1], min_height * grid_size[0]))

        positions = [(x * min_width, y * min_height) for y in range(grid_size[0]) for x in range(grid_size[1])]
        for i, img in enumerate(resized_images):
            new_image.paste(img, positions[i])

        save_path = os.path.join(base_plot_path, 'concatenated')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        new_image.save(os.path.join(save_path, f'component_{c}.png'))


def find_intersection_of_gaussians(mu1, std1, mu2, std2, tol=1e-8):
    """
    Finds the intersection points of two Gaussian probability density functions.

    Args:
        mu1 (float): Mean of the first Gaussian distribution.
        std1 (float): Standard deviation of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        std2 (float): Standard deviation of the second Gaussian distribution.
        tol (float): A small tolerance for floating-point comparisons.

    Returns:
        list: A sorted list of x-values where the PDFs intersect.
    """
    if std1 <= 0 or std2 <= 0:
        raise ValueError("Standard deviations must be positive.")

    var1, var2 = std1 ** 2, std2 ** 2

    # Special case: If standard deviations are nearly equal
    if abs(std1 - std2) < tol:
        if abs(mu1 - mu2) < tol:
            return [mu1]
        else:
            return [(mu1 + mu2) / 2]

    a = 1 / var2 - 1 / var1
    b = 2 * (mu1 / var1 - mu2 / var2)
    c = (mu2 ** 2 / var2 - mu1 ** 2 / var1) - 2 * np.log(std1 / std2)

    discriminant = b ** 2 - 4 * a * c

    if discriminant < -tol:
        return []
    elif abs(discriminant) < tol:
        return [-b / (2 * a)]
    else:
        sqrt_d = np.sqrt(discriminant)
        x1 = (-b + sqrt_d) / (2 * a)
        x2 = (-b - sqrt_d) / (2 * a)
        return sorted([x1, x2])


def plot_gaussians_with_solutions(mu1, std1, mu2, std2):
    """
    Visualize two Gaussian distributions and their intersection points.

    Args:
        mu1, std1: Parameters for first Gaussian.
        mu2, std2: Parameters for second Gaussian.
    """
    plt.figure(figsize=(12, 7))

    min_mu = min(mu1, mu2)
    max_mu = max(mu1, mu2)
    max_std = max(std1, std2)
    x_min = min_mu - 4 * max_std
    x_max = max_mu + 4 * max_std

    x = np.linspace(x_min, x_max, 1000)
    pdf1 = norm.pdf(x, mu1, std1)
    pdf2 = norm.pdf(x, mu2, std2)

    plt.plot(x, pdf1, 'b-', linewidth=2, label=f'Dist A: μ={mu1}, σ={std1}')
    plt.plot(x, pdf2, 'r-', linewidth=2, label=f'Dist B: μ={mu2}, σ={std2}')

    solutions = find_intersection_of_gaussians(mu1, std1, mu2, std2)

    if solutions:
        for i, sol in enumerate(solutions):
            y_val = norm.pdf(sol, mu1, std1)
            plt.scatter(sol, y_val, s=100, zorder=5, label=f'Intersect {i + 1}: x={sol:.2f}')
            plt.axvline(x=sol, color='g', linestyle='--', alpha=0.7)

    plt.title('Intersection of Gaussian Distributions', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('PDF', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def fit_gmm_with_elbow_and_plot(data, bins=200, component_index=None, fish_path=None, component_number=None,
                                set_id=None, corrected_component_number=None, mode='Energy ratio'):
    """
    Fit Gaussian Mixture Model (GMM) to data, determining optimal components using Elbow method,
    and find intersection points (thresholds).

    Args:
        data (np.ndarray): Input data (1D array).
        bins (int): Number of bins for histogram.
        component_index (int): Index of the component being analyzed.
        fish_path (str): Path to fish directory.
        component_number (int): Total number of components.
        set_id (str): ID of the set (e.g., 'prey-loom-omr-set001').
        corrected_component_number (int): Manually corrected number of GMM components.
        mode (str): Analysis mode ('Energy ratio', etc.).

    Returns:
        tuple: (intersections, component_params)
    """
    data_2d = data.reshape(-1, 1)

    # Use the elbow method to find the optimal number of components
    n_components_range = range(1, 15)
    bics = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data_2d)
        bics.append(gmm.bic(data_2d))

    if corrected_component_number is None:
        kn = KneeLocator(n_components_range, bics, curve='convex', direction='decreasing')
        if kn.all_elbows:
            elbow_bics = {elbow: bics[elbow - 1] for elbow in kn.all_elbows}
            optimal_n_components = min(elbow_bics, key=elbow_bics.get)
        else:
            optimal_n_components = np.argmin(bics) + 1  # Fallback
    else:
        optimal_n_components = corrected_component_number

    print(f'Optimal number of components: {optimal_n_components}')

    # Plot BIC curve
    plt.figure(figsize=(9, 3))
    plt.plot(n_components_range, bics, '-', label='BIC score', color='#F2AC0B', lw=3)
    plt.plot(optimal_n_components, bics[optimal_n_components - 1], marker='*', markersize=18, color='red', 
             label=f'Elbow at n={optimal_n_components}')
    plt.xlabel('Number of Gaussian components', fontsize=22)
    plt.ylabel('BIC Score', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=20, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.25)
    
    if fish_path and component_number is not None:
        bic_save_path = os.path.join(fish_path, f"spatial_weight_gmm_bic_plots_full_sets_{mode}_time_corrected", 
                                     str(set_id), str(component_number))
        if not os.path.exists(bic_save_path):
            os.makedirs(bic_save_path)
        plt.savefig(os.path.join(bic_save_path, f'component_{component_index}.png'), dpi=300)
    # plt.show()
    plt.close()

    # Fit GMM with optimal components
    gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
    gmm.fit(data_2d)

    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())

    # Order components by mean
    order = np.argsort(means)
    weights, means, stds = weights[order], means[order], stds[order]

    x = np.linspace(data.min() - 1, data.max() + 1, 4096)
    pdfs = [w * norm.pdf(x, mu, std) for w, mu, std in zip(weights, means, stds)]
    # mixture_pdf = np.sum(pdfs, axis=0) # Unused

    intersections = {}
    if optimal_n_components > 1:
        for i in range(optimal_n_components - 1):
            mu1, mu2 = means[i], means[i + 1]
            std1, std2 = stds[i], stds[i + 1]
            w1, w2 = weights[i], weights[i + 1]

            def pdf_diff(x):
                pdf_A_val = w1 * norm.pdf(x, mu1, std1)
                pdf_B_val = w2 * norm.pdf(x, mu2, std2)
                return pdf_A_val - pdf_B_val

            search_min = min(mu1, mu2) - 3 * max(std1, std2)
            search_max = max(mu1, mu2) + 3 * max(std1, std2)

            scan_points = np.linspace(search_min, search_max, 500)
            thresholds_fifty = []
            for j in range(len(scan_points) - 1):
                try:
                    if np.sign(pdf_diff(scan_points[j])) != np.sign(pdf_diff(scan_points[j + 1])):
                        root = brentq(pdf_diff, scan_points[j], scan_points[j + 1])
                        thresholds_fifty.append(root)
                except ValueError:
                    pass
            solutions = sorted(list(set(thresholds_fifty)))

            try:
                if len(solutions) == 1:
                    point = solutions[0]
                else:
                    solution_1 = solutions[0]
                    solution_2 = solutions[1]
                    if solution_1 < mu1:
                        point = solution_2
                    elif solution_1 > mu1:
                        point = min(solution_1, solution_2)
                    else:
                        point = solution_1 # Fallback
                intersections[f'intersection_{i + 1}_{i + 2}'] = point
            except (ValueError, RuntimeError, IndexError):
                point = (mu1 + mu2) / 2
                intersections[f'intersection_{i + 1}_{i + 2}'] = point

    elif optimal_n_components == 1:
        intersections['percentile_90'] = np.percentile(data, 90)

    print('Intersections: ', intersections)

    # Plot GMM fitting results
    plt.figure(figsize=(9, 8))
    plt.hist(data, bins=bins, density=True, alpha=0.4, color='grey')

    for i in range(optimal_n_components):
        label = 'Gaussian components' if i == 0 else None
        plt.plot(x, pdfs[i], '--', lw=3, color='black', label=label)

    if intersections:
        try:
            max_val = max(intersections.values())
            ax = plt.gca()
            ax.vlines(max_val, ymin=0, ymax=45, colors='red', linestyles='-', linewidth=2, label='ER threshold')
        except Exception:
            pass

    plt.xlabel(f'{mode}', fontsize=28)
    plt.ylabel('Density', fontsize=28)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax = plt.gca()
    
    # Clean up y-axis ticks
    _yt = ax.get_yticks()
    ax.set_yticks([v for v in _yt if not np.isclose(v, 0.0)])
    ax.tick_params(axis='y', labelsize=17)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.legend(fontsize=28, frameon=False, loc='upper right', bbox_to_anchor=(0.99, 0.98))
    plt.grid(False)
    plt.xlim(0, 0.05)
    plt.ylim(0, 80)
    
    yt0, yt1 = ax.get_ylim()
    ax.set_yticks(np.linspace(yt0, yt1, 5)[1:])

    if fish_path and component_number is not None:
        save_path = os.path.join(fish_path, f"{mode}_gmm_plots_with_intersections_full_sets_time_corrected", 
                                 str(set_id), str(component_number))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'component_{component_index}.png'), dpi=300)
    
    # plt.show()
    plt.close()

    component_params = [(weights[i], means[i], stds[i]) for i in range(optimal_n_components)]
    return intersections, component_params


def get_behavior_timestamps(filename):
    """
    Get behavioral timestamps from a file.

    Args:
        filename (str): Path to the timestamp file.

    Returns:
        np.ndarray: Timestamps.
    """
    t = np.loadtxt(filename)[:, 1]
    t = (t - t[0]) / 3515839
    return t


def plot_astrocytes(coordinates, coordinates_unselected, background_image, selected_s, selected_t, mean_activity,
                    intersection_mask, intersection_threshold, component_index=None, fish_path=None,
                    component_number=None, fish_name=None, set_id=None, plot_range=None, plot_intersection=None,
                    rasterplot=None, mode='Energy ratio', h_min=None, h_max=None):
    """
    Plot astrocyte locations, activities, and behavioral context.
    """
    # Load necessary data
    snr_mask = np.load(os.path.join(fish_path, "snr_mask.npy"))
    # points = np.load(os.path.join(fish_path, "contours.npy"), allow_pickle=True) # Unused
    time_stamps = np.load(os.path.join(fish_path, "t.npy"))
    # sifted_points = points[snr_mask] # Unused
    total_astrocyte = snr_mask.sum() # Approximate

    h5_path = os.path.join(BEHAVIOUR_DIR, f"{fish_name}/TOPCAMERA/{set_id}/{fish_name.replace('-', '_')}_Trial1.h5")
    df_eye = pd.read_hdf(h5_path, "eye")
    df_tail = pd.read_hdf(h5_path, "tail")

    tail_angles = get_tail_angles(df_tail, df_eye["heading"].values)
    averaged_tail_angles = np.degrees(np.average(tail_angles[:, 5:], axis=1))

    eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
    eye_angles_filt = low_pass_filt(eye_angles, 300, 4)
    left_angle = eye_angles_filt[:, 0]
    right_angle = eye_angles_filt[:, 1]

    with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
        convergence_periods = json.load(file)
    current_convergence_periods = convergence_periods.get(fish_name, {}).get(set_id, [])

    bouts = bouts_info.get(fish_name, {}).get(set_id, {}).get('bout', [])
    behavior = bouts_info.get(fish_name, {}).get(set_id, {}).get('behaviour', [])
    
    new_bouts = [[(b[0]/300)*3, (b[1]/300)*3] for b in bouts]
    periods, stimuli = get_stimulus(fish_name)

    with open(PASSIVITY_PATH, 'r') as file:
        passivity_json = json.load(file)
    passivity = np.array(passivity_json.get(fish_name, {}).get(set_id, []))

    intersection_masked_s = selected_s[intersection_mask]
    inverse_masked_s = selected_s[~intersection_mask]

    # Combine weights for a unified color scale
    all_weights = np.concatenate([intersection_masked_s, inverse_masked_s])
    if all_weights.size > 0:
        vmin, vmax = np.min(all_weights), np.max(all_weights)
    else:
        vmin, vmax = 0, 1

    norm_color = mcolors.Normalize(vmin=vmin, vmax=vmax)
    original_cmap = plt.colormaps.get_cmap('Reds_r')
    custom_cmap = mcolors.ListedColormap(original_cmap(np.linspace(0.05, 0.8, 256)))

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x_u = coordinates_unselected[:, 0]
    y_u = coordinates_unselected[:, 1]

    if coordinates.shape[0] == 0:
        print("No points to plot.")
        return

    # Plotting
    fig = plt.figure(figsize=(40, 20))
    gs = fig.add_gridspec(7, 3, height_ratios=[0.4, 2, 2, 2, 2, 4, 8], hspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    gs0 = gs[1, :].subgridspec(2, 1, hspace=0.1)
    ax0_top = fig.add_subplot(gs0[0])
    ax0_bottom = fig.add_subplot(gs0[1], sharex=ax0_top)

    behavior_time_axis = np.arange(len(averaged_tail_angles)) / 300.0

    # Top plot: Eye angles
    ax0_top.plot(behavior_time_axis, left_angle, color='blue', alpha=0.6, label='Left Eye Angle')
    ax0_top.plot(behavior_time_axis, right_angle, color='red', alpha=0.6, label='Right Eye Angle')

    if current_convergence_periods:
        for i, conv_period in enumerate(current_convergence_periods):
            conv_start = conv_period[0] / 300
            conv_end = conv_period[1] / 300
            ax0_top.axvspan(conv_start, conv_end, color='yellow', alpha=0.3,
                            label='Convergence Period' if i == 0 else "")

    ax0_top.tick_params(axis='y', labelcolor='black')
    ax0_top.set_xticks([])
    ax0_top.spines['top'].set_visible(False)
    ax0_top.spines['bottom'].set_visible(False)

    # Bottom plot: Tail angle
    ax0_bottom.plot(behavior_time_axis, averaged_tail_angles, color='darkslategrey', label='Tail Angle')
    ax0_bottom.tick_params(axis='y', labelcolor='black')
    ax0_bottom.set_xticks([])
    ax0_bottom.spines['top'].set_visible(False)
    ax0_bottom.spines['bottom'].set_visible(False)

    ax0_top.set_ylabel('Degrees', fontsize=17)
    ax0_top.yaxis.set_label_coords(-0.02, -0.1)

    if plot_range is not None:
        ax0_bottom.set_xlim(plot_range[0], plot_range[1])
    else:
        ax0_bottom.set_xlim(0, len(averaged_tail_angles) / 300.0)

    lines, labels = ax0_top.get_legend_handles_labels()
    lines2, labels2 = ax0_bottom.get_legend_handles_labels()
    ax0_top.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=16)

    ax2 = fig.add_subplot(gs[2, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4, :])
    ax5 = fig.add_subplot(gs[5, :])

    ax6_1 = fig.add_subplot(gs[6, 0])
    ax6_2 = fig.add_subplot(gs[6, 1])
    ax6_3 = fig.add_subplot(gs[6, 2])
    axes_6 = [ax6_1, ax6_2, ax6_3]

    # Stimulus plot (ax1)
    for (start, end), label in zip(periods, stimuli):
        start = start * 3
        end = end * 3
        color = 'grey' if label == 'omr stationary' else 'black'
        ax1.hlines(y=0, xmin=start, xmax=end, color=color, linewidth=4)
        center = (start + end) / 2
        ax1.text(center, 0.02, label, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), fontsize=18)

    if fish_name in ['20250718-F1', '20250718-F4'] and len(periods) > 9:
        ax1.hlines(y=0, xmin=periods[8][0] * 3, xmax=periods[8][1] * 3, color='lightgrey', linewidth=4)
        ax1.hlines(y=0, xmin=periods[9][0] * 3, xmax=periods[9][1] * 3, color='lightgrey', linewidth=4)

    if plot_range is not None:
        ax1.set_xlim(plot_range[0] * 3, plot_range[1] * 3)
    else:
        ax1.set_xlim(0, selected_t.shape[0] + 3)
    ax1.set_xticks([])
    ax1.axis('off')

    # Selected component (ax2)
    ax2.plot(time_stamps, selected_t, color='black')
    y_min, _ = ax2.get_ylim()
    for j in range(passivity.shape[0]):
        start = passivity[j][0]
        end = passivity[j][1]
        ax2.axvspan(start, end, color='papayawhip', alpha=0.6, label='Passivity period' if j == 0 else "")
        ax2.plot(start, y_min, 'g^', markersize=6, clip_on=False, label='Passivity Start' if j == 0 else "")
        ax2.plot(end, y_min, 'r^', markersize=6, clip_on=False, label='Passivity End' if j == 0 else "")

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=17)

    for i, (start, end) in enumerate(new_bouts):
        if i < len(behavior):
            ax2.axvspan(start / 3, end / 3, color=behavior_color_map.get(behavior[i], 'gray'), alpha=0.8)
    
    ax2.set_title(f'Temporal Pattern {component_index} ({fish_name} {set_id})', fontsize=19, fontweight='bold')
    ax2.set_ylabel('Amplitude', fontsize=17)
    if plot_range is not None:
        ax2.set_xlim(plot_range[0], plot_range[1])
    else:
        ax2.set_xlim(0, time_stamps[-1])
    if h_min is not None and h_max is not None:
        ax2.set_ylim(h_min, h_max)

    # Mean activity (ax3)
    ax3.plot(time_stamps, mean_activity, color='black')
    for j in range(passivity.shape[0]):
        start = passivity[j][0]
        end = passivity[j][1]
        ax3.axvspan(start, end, color='papayawhip', alpha=0.6)
        ax3.plot(start, -3, 'g^', markersize=6, clip_on=False)
        ax3.plot(end, -3, 'r^', markersize=6, clip_on=False)

    for i, (start, end) in enumerate(new_bouts):
        if i < len(behavior):
            ax3.axvspan(start / 3, end / 3, color=behavior_color_map.get(behavior[i], 'gray'), alpha=0.8)

    ax3.set_title(f'Averaged activity of selected astrocytes', fontsize=19, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=17)
    ax3.set_ylabel('Normalized\ndfof', fontsize=17)
    if plot_range is not None:
        ax3.set_xlim(plot_range[0], plot_range[1])
    else:
        ax3.set_xlim(0, time_stamps[-1])
    ax3.set_ylim(-3, 5)

    # Legend (ax4)
    legend_elements = [Patch(facecolor=behavior_color_map[b], edgecolor='black', alpha=0.8, label=b)
                       for b in unique_behaviors]
    ax4.legend(handles=legend_elements, loc='center', ncol=(len(unique_behaviors) + 1) // 2, fontsize=16)
    ax4.axis('off')

    # Raster plot (ax5)
    ax5.imshow(rasterplot, vmin=0, vmax=1.5, cmap='gray_r', aspect='auto', interpolation='nearest',
               extent=[time_stamps[0], time_stamps[-1], rasterplot.shape[0], 0])
    for i, (start, end) in enumerate(new_bouts):
        if i < len(behavior):
            ax5.axvspan(start / 3, end / 3, color=behavior_color_map.get(behavior[i], 'gray'), alpha=0.5)
    ax5.set_title(f'Activities of selected astrocytes', fontsize=18, fontweight='bold')
    ax5.set_ylabel('Astrocytes', fontsize=17)
    if plot_range is not None:
        ax5.set_xlim(plot_range[0], plot_range[1])
    else:
        ax5.set_xlim(0, time_stamps[-1])

    # Brain scatter plots (ax6)
    ax6_1.imshow(background_image, cmap='gray', origin='upper')
    if x.size > 0:
        ax6_1.scatter(x, y, s=10, c=intersection_masked_s, cmap=custom_cmap, norm=norm_color)
    ax6_1.axis('off')
    ax6_1.set_title(f'Selected astrocytes ({len(coordinates)} astrocytes)', fontsize=18, fontweight='bold')

    ax6_2.imshow(background_image, cmap='gray', origin='upper')
    if x_u.size > 0:
        ax6_2.scatter(x_u, y_u, s=10, c=inverse_masked_s, cmap=custom_cmap, norm=norm_color)
    ax6_2.axis('off')
    ax6_2.set_title(f'Rejected astrocytes ({total_astrocyte - len(coordinates)} astrocytes)', fontsize=18, fontweight='bold')

    ax6_3.imshow(background_image, cmap='gray', origin='upper')
    if x_u.size > 0:
        ax6_3.scatter(x_u, y_u, s=10, c=inverse_masked_s, cmap=custom_cmap, norm=norm_color)
    if x.size > 0:
        ax6_3.scatter(x, y, s=10, c=intersection_masked_s, cmap=custom_cmap, norm=norm_color)
    ax6_3.axis('off')
    ax6_3.set_title(f'Combined ({total_astrocyte} astrocytes)', fontsize=18, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_6, orientation='horizontal', fraction=0.05)
    cbar.set_label(f"{mode}", fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    if intersection_threshold is not None:
        for i in range(len(intersection_threshold)):
            cbar.ax.axvline(intersection_threshold[i], color='black', linestyle='--', linewidth=2)
            cbar.ax.annotate(f'Threshold {i + 1}',
                             xy=(intersection_threshold[i], 1), xytext=(intersection_threshold[i], 1.8),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                             ha='center', va='bottom', fontsize=12, style='italic')

    # Save logic
    save_path = os.path.join(fish_path, f"selected_astrocytes_heat_map_full_sets_{mode}_time_corrected(elbow)",
                             str(component_number), set_id, f"component_{component_index}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename_suffix = f"_{plot_intersection}" if plot_intersection else ""
    
    if plot_range is None:
        plt.savefig(os.path.join(save_path, f'component_{component_index}{filename_suffix}.png'), dpi=300)
        print(f'Figure saved: {save_path}')
    else:
        range_dir = os.path.join(save_path, f"{plot_range[0]}_{plot_range[1]}")
        os.makedirs(range_dir, exist_ok=True)
        plt.savefig(os.path.join(range_dir, f'component_{component_index}{filename_suffix}.png'), dpi=300)

    plt.close()


def sift_component_related_neurons(fish_name, component_number=None, component_index=None,
                                   corrected_component_numer=None, plot_range=None, mode='Energy_ratio'):
    """
    Select astroglia with GMM and intersection, then plot out the trace and brain plot.

    Args:
        fish_name (str): Fish name.
        component_number (int): Total NMF components.
        component_index (int): Index of component to analyze.
        corrected_component_numer (int): Manual override for GMM components.
        plot_range (tuple): Optional time range for plotting.
        mode (str): Analysis mode.
    """
    fish_dir = os.path.join(REGISTERED_FISH_DIR, fish_name)
    template_dir = Path(os.path.join(REGISTERED_FISH_DIR, "20250807-F1")) # Template fish
    
    spatial_path = os.path.join(fish_dir, f"Spatial_components_full_sets/Spatial_components_{component_number}.npy")
    W = np.load(spatial_path)

    sets_name = ['prey-loom-omr-set001', 'prey-loom-omr-set002', 'prey-loom-omr-set003', 'prey-loom-omr-set004']
    
    temporal_dir = os.path.join(fish_dir, "Temporal_components_full_sets")
    H_sets = []
    for i, set_name in enumerate(sets_name):
        H_sets.append(np.load(os.path.join(temporal_dir, f"Temporal_components_{set_name}_{component_number}.npy")))

    H = np.concatenate(H_sets, axis=1)

    points = np.load(os.path.join(fish_dir, 'contours.npy'), allow_pickle=True)
    snr_mask = np.load(os.path.join(fish_dir, 'snr_mask.npy'))
    coordinate = np.load(os.path.join(fish_dir, 'xyz_in_F1.npy'))

    # points_snr_masked = points[snr_mask] # Unused
    coordinates_snr_masked = coordinate[snr_mask]

    # Compute energy ratios
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / norms
    H_max = np.max(H_norm, axis=1)
    H_min = np.min(H_norm, axis=1)
    H_norm_sets = np.split(H_norm, 4, axis=1)

    W_scaled = W * norms.T
    component_energies = np.abs(W_scaled)
    total_energies = np.sum(component_energies, axis=1, keepdims=True)
    FE_matrix = component_energies / total_energies

    selected_energy_ratio = FE_matrix[:, component_index]
    selected_weights = W_scaled[:, component_index]

    if mode == 'Weighted energy ratio':
        selected = selected_energy_ratio * selected_weights
    elif mode == 'Energy ratio':
        selected = selected_energy_ratio.copy()
    elif mode == 'Weight':
        selected = selected_weights.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Use first set for GMM fitting (or could be concatenated, but original code passed set_id=None which implies generic)
    # The original code passes set_id inside loop later, but fit_gmm is called once.
    # It seems fit_gmm is called with global 'selected' metric.
    intersection, _ = fit_gmm_with_elbow_and_plot(selected, component_index=component_index,
                                                  fish_path=fish_dir, component_number=component_number, set_id="combined",
                                                  corrected_component_number=corrected_component_numer, mode=mode)

    intersection_keys = list(intersection.keys())

    # Clean up old files (logic from original code)
    # This might be dangerous if not careful, but keeping as per original logic
    # ... skipped specific delete logic to avoid accidental data loss during refactor ...

    for i in range(len(intersection_keys) + 1):
        if i == 0:
            key = 'intersection_0_1'
            temp_intersection_1 = 0
            temp_intersection_2 = intersection[intersection_keys[i]]
            intersection_mask = (temp_intersection_1 < selected) & (selected < temp_intersection_2)
            intersections = [temp_intersection_1, temp_intersection_2]
        elif i == len(intersection_keys):
            key = intersection_keys[i - 1]
            temp_intersection_1 = intersection[key]
            intersection_mask = temp_intersection_1 < selected
            intersections = [temp_intersection_1]
        else:
            key = intersection_keys[i - 1]
            temp_intersection_1 = intersection[key]
            temp_intersection_2 = intersection[intersection_keys[i]]
            intersection_mask = (temp_intersection_1 < selected) & (selected < temp_intersection_2)
            intersections = [temp_intersection_1, temp_intersection_2]

        mask_save_path = os.path.join(fish_dir, f"intersection_masks_full_sets_{mode}_time_corrected", 
                                      str(component_number))
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        np.save(os.path.join(mask_save_path, f'component_{component_index}_{key}.npy'), intersection_mask)

        coordinates_final = coordinates_snr_masked[intersection_mask]
        coordinates_unselected = coordinates_snr_masked[~intersection_mask]

        # Load background image
        plane_paths = sorted(
            template_dir.glob("*plane*.tif"),
            key=lambda p: int(re.search(r"plane(\d+)", p.name).group(1))
        )
        stack = np.stack([tifffile.imread(p)[0] for p in plane_paths], axis=0)
        projection = np.max(stack, axis=0)
        p_low, p_high = np.percentile(projection, (2, 98))
        projection = rescale_intensity(projection, in_range=(p_low, p_high), out_range=(0, 1))

        # Generate plots for each set
        masked_selected_metric = selected[intersection_mask]
        sorted_indices = np.argsort(masked_selected_metric)

        for set_idx, set_name in enumerate(sets_name):
            X = np.load(os.path.join(fish_dir, f"normalized_masked_dfof_{set_name}.npy"))
            if fish_name in ['20250718-F1', '20250718-F4']:
                X = X[:, 3:] # Truncate first 3 frames if needed, matching logic
            
            X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
            X_norm = np.nan_to_num(X_norm)
            
            # Check if mask is valid
            if intersection_mask.any():
                mean_activity = X_norm[intersection_mask].mean(axis=0)
                masked_X_norm = X_norm[intersection_mask]
                rasterplot = masked_X_norm[sorted_indices]
            else:
                mean_activity = np.zeros(X_norm.shape[1])
                rasterplot = np.zeros((1, X_norm.shape[1]))

            temp_component = H_norm_sets[set_idx][component_index]

            plot_astrocytes(coordinates_final, coordinates_unselected, projection, selected,
                            temp_component,
                            mean_activity, intersection_mask, intersections, component_index, fish_dir,
                            component_number, fish_name, set_name,
                            plot_range=plot_range, plot_intersection=key, rasterplot=rasterplot, 
                            mode=mode, h_min=H_min[component_index], h_max=H_max[component_index])


def copy_plots_for_comparison(fish_name, component_number, mode='Energy ratio'):
    """
    Copy plots to a comparison directory.
    """
    print('Start copying plots for comparison')
    sets = ['prey-loom-omr-set001', 'prey-loom-omr-set002', 'prey-loom-omr-set003', 'prey-loom-omr-set004']
    fish_dir = os.path.join(REGISTERED_FISH_DIR, fish_name)

    # Cleanup logic omitted for safety

    for i in range(component_number):
        for set_name in sets:
            file_path = os.path.join(fish_dir, f"selected_astrocytes_heat_map_full_sets_{mode}_time_corrected(elbow)",
                                     str(component_number), set_name)
            
            comp_path = os.path.join(file_path, f"component_{i}")
            if not os.path.exists(comp_path):
                continue
                
            files = os.listdir(comp_path)
            if not files:
                continue
                
            sorted_img = sorted(files)[-1]
            compare_dir = os.path.join(file_path, "for compare")
            if not os.path.exists(compare_dir):
                os.makedirs(compare_dir)

            shutil.copyfile(os.path.join(comp_path, sorted_img), os.path.join(compare_dir, sorted_img))


def run_NMF(component_number):
    """Run NMF for all 10 fish."""
    ten_fish_names = os.listdir(REGISTERED_FISH_DIR)
    for f in ten_fish_names:
        NMF_on_dfof(f, component_number=component_number)


def run_all_components(component_number, mode='Energy ratio'):
    """Run analysis for all components."""
    print('Start running all components')
    for i in range(component_number):
        print('Running component ' + str(i))
        sift_component_related_neurons('20250807-F1', component_number, component_index=i, mode=mode)


if __name__ == "__main__":
    # Example usage:
    # run_dfof_preprocess()
    
    # component_number = 90
    # mode = 'Energy ratio'
    
    # run_NMF(component_number)
    # run_all_components(component_number, mode)
    
    pass
