"""
Detect convergence periods from binocular eye angles (10-fish astroglia dataset).

This script estimates thresholds for binocular eye convergence by fitting Gaussian mixture
models (GMMs) to eye-angle distributions and then detects convergence episodes across time.

To make the code portable for GitHub, file paths are derived from the environment variable
ASTROCYTE_DATA_ROOT (default: D:/2p astrocyte). Set this variable to your local data root.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from side_top_cam_analysis_utils import low_pass_filt

# Constants for file paths
ASTROCYTE_DATA_ROOT = os.environ.get("ASTROCYTE_DATA_ROOT", r"D:/2p astrocyte")
BEHAVIOUR_BASE_DIR = os.path.join(ASTROCYTE_DATA_ROOT, "behaviour")
STIMULUS_INFO_10_FISH = os.path.join(BEHAVIOUR_BASE_DIR, "10 fish data", "stimulus_info.xlsx")
STIMULUS_INFO_TEST = os.path.join(BEHAVIOUR_BASE_DIR, "2p-test-20250718-info.xlsx")
BOUTS_INFO_PATH = os.path.join(BEHAVIOUR_BASE_DIR, "bouts_info_appended.json")
CONVERGENCE_DICT_DIR = os.path.join(BEHAVIOUR_BASE_DIR, "convergence periods", "convergence threshold dictionary")
CONVERGENCE_PERIODS_OUTPUT = os.path.join(BEHAVIOUR_BASE_DIR, "convergence periods", "convergence_periods.json")
BIMODAL_FIT_PLOTS_DIR = os.path.join(BEHAVIOUR_BASE_DIR, "convergence periods", "bimodal_fit_plots")

def convert_numpy_to_python(obj):
    """
    Recursively convert NumPy data types to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    return obj

def get_stimulus(stim_path):
    """
    Load stimulus information from Excel files.
    
    Args:
        stim_path (int or str): Indicator for which stimulus file to load. 
                                10 for '10 fish data', otherwise '2p-test-20250718-info'.
    
    Returns:
        tuple: (periods, stimuli) where periods is a list of [start, end] lists, 
               and stimuli is a list of stimulus names.
    """
    if stim_path == 10:
        df = pd.read_excel(STIMULUS_INFO_10_FISH, engine="openpyxl")
    else:
        df = pd.read_excel(STIMULUS_INFO_TEST, engine="openpyxl")

    start = df["start"].tolist()
    end = df["end"].tolist()
    stimuli = df["stimuli"].tolist()

    periods = []
    for i in range(len(stimuli)):
        periods.append([start[i], end[i]])

    if stim_path != 10:
        periods.append([450, 510])
        periods.append([750, 810])

    return periods, stimuli

def detect_hunting_episodes_elbow(eye_angles, fish_name, savefig=True, criterion='bic'):
    """
    Detect hunting episodes using GMM and elbow method to find optimal components.
    
    Args:
        eye_angles (np.array): Array of eye angles.
        fish_name (str): Name of the fish for labeling and saving.
        savefig (bool): Whether to save the plot.
        criterion (str): 'bic' or 'aic' for model selection.
        
    Returns:
        dict: Dictionary containing threshold info and GMM parameters, or None if failed.
    """
    data = eye_angles.reshape(-1, 1)

    # Step 1: Determine the optimal number of components using BIC/AIC and elbow method
    n_components_range = range(1, 21)
    bics = []
    aics = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, n_init=10)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        aics.append(gmm.aic(data))

    print(f'BIC: {bics}')
    print(f'AIC: {aics}')

    # Find the elbow point
    scores = bics if criterion == 'bic' else aics
    elbow_point = None
    if len(n_components_range) >= 3:
        try:
            kl = KneeLocator(n_components_range, scores, curve='convex', direction='decreasing', S=1.0)
            elbow_point = kl.elbow
            if elbow_point is None:
                print(f"KneeLocator did not find an elbow for {fish_name} using {criterion.upper()}. Defaulting to minimum.")
                elbow_point = n_components_range[np.argmin(scores)]
        except Exception as e:
            print(f"Error finding elbow point for {fish_name} using {criterion.upper()}: {e}. Defaulting to minimum.")
            elbow_point = n_components_range[np.argmin(scores)]
    else:
        elbow_point = n_components_range[np.argmin(scores)]

    print(f"Selected number of components for {fish_name}: {elbow_point}")

    # Step 2: Fit GMM with the selected number of components
    final_gmm = GaussianMixture(n_components=elbow_point, covariance_type='full', random_state=42, n_init=10)
    final_gmm.fit(data)

    weights = final_gmm.weights_
    means = final_gmm.means_.flatten()
    covariances = final_gmm.covariances_.flatten()
    stds = np.sqrt(covariances)

    # Sort components by mean in descending order
    sorted_indices = np.argsort(means)[::-1]
    weights = weights[sorted_indices]
    means = means[sorted_indices]
    stds = stds[sorted_indices]

    if elbow_point < 1:
        print(f"No components found for {fish_name}. Skipping threshold calculation.")
        return

    # Step 3: Identify Component A (largest mean) and Component B (second largest mean)
    component_A_mean = means[0]
    component_A_std = stds[0]
    component_A_weight = weights[0]

    component_B_mean = None
    component_B_std = None
    component_B_weight = None

    if elbow_point > 1:
        component_B_mean = means[1]
        component_B_std = stds[1]
        component_B_weight = weights[1]
    else:
        print(f"Only one component found for {fish_name}. Cannot define Component B or calculate thresholds.")
        # Visualization for single component (omitted for brevity in cleanup, but can be added back if needed)
        # For now, returning None as per logic
        return {
            'fish_name': fish_name,
            'selected_components': elbow_point,
            'threshold_fifty': [],
            'threshold_eighty': [],
            'gmm_weights': final_gmm.weights_,
            'gmm_means': final_gmm.means_,
            'gmm_covariances': final_gmm.covariances_
        }

    # Step 4: Calculate thresholds_fifty
    thresholds_fifty = []

    def pdf_diff(x):
        pdf_A_val = component_A_weight * norm.pdf(x, component_A_mean, component_A_std)
        pdf_B_val = component_B_weight * norm.pdf(x, component_B_mean, component_B_std)
        return pdf_A_val - pdf_B_val

    search_min = min(component_A_mean, component_B_mean) - 3 * max(component_A_std, component_B_std)
    search_max = max(component_A_mean, component_B_mean) + 3 * max(component_A_std, component_B_std)

    scan_points = np.linspace(search_min, search_max, 500)
    for i in range(len(scan_points) - 1):
        try:
            if np.sign(pdf_diff(scan_points[i])) != np.sign(pdf_diff(scan_points[i + 1])):
                root = brentq(pdf_diff, scan_points[i], scan_points[i + 1])
                thresholds_fifty.append(root)
        except ValueError:
            pass
    thresholds_fifty = sorted(list(set(thresholds_fifty)))

    # Step 5: Calculate thresholds_eighty
    thresholds_eighty = []

    def prob_A_given_AB_minus_08(x):
        pdf_A_val = component_A_weight * norm.pdf(x, component_A_mean, component_A_std)
        pdf_B_val = component_B_weight * norm.pdf(x, component_B_mean, component_B_std)
        if (pdf_A_val + pdf_B_val) == 0:
            return 0 - 0.8
        return (pdf_A_val / (pdf_A_val + pdf_B_val)) - 0.8

    scan_points_80 = np.linspace(search_min, search_max, 500)
    for i in range(len(scan_points_80) - 1):
        try:
            if np.sign(prob_A_given_AB_minus_08(scan_points_80[i])) != np.sign(prob_A_given_AB_minus_08(scan_points_80[i + 1])):
                root = brentq(prob_A_given_AB_minus_08, scan_points_80[i], scan_points_80[i + 1])
                thresholds_eighty.append(root)
        except ValueError:
            pass
    thresholds_eighty = sorted(list(set(thresholds_eighty)))

    threshold_eighty = None
    if thresholds_eighty:
        pdf_A_values_at_thresholds = component_A_weight * norm.pdf(np.array(thresholds_eighty), component_A_mean, component_A_std)
        threshold_eighty = thresholds_eighty[np.argmax(pdf_A_values_at_thresholds)]

    # Step 6: Determine final threshold_fifty
    threshold_fifty = None
    if thresholds_fifty and threshold_eighty is not None:
        for i in range(len(thresholds_fifty)):
            if thresholds_fifty[i] <= threshold_eighty:
                threshold_fifty = thresholds_fifty[i]

    print(f"For {fish_name}:")
    print(f"  Thresholds 50%: {thresholds_fifty}")
    print(f"  Selected threshold_fifty: {threshold_fifty}")
    print(f"  Thresholds 80%: {thresholds_eighty}")
    print(f"  Selected threshold_eighty: {threshold_eighty}")

    # Step 7: Visualization
    if savefig:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot 1: Scores
        scores_to_plot = bics if criterion == 'bic' else aics
        axs[0].plot(n_components_range, scores_to_plot, marker='o', linestyle='-', label=f'{criterion.upper()} Score')
        axs[0].axvline(elbow_point, color='r', linestyle='--', label=f'Elbow Point: {elbow_point} components')
        axs[0].set_xlabel('Number of GMM Components')
        axs[0].set_ylabel(f'{criterion.upper()} Score')
        axs[0].set_title(f'{criterion.upper()} Score vs. Number of Components for {fish_name}')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_xlim([0, 20])
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot 2: Data and GMM
        axs[1].hist(data.flatten(), bins=100, density=True, alpha=0.5, label='Data Distribution')
        x_plot = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
        log_prob = final_gmm.score_samples(x_plot)
        axs[1].plot(x_plot.flatten(), np.exp(log_prob), color='black', lw=2.5, label='GMM Fitted Distribution')

        colors = plt.cm.viridis(np.linspace(0, 1, elbow_point))
        for i in range(elbow_point):
            comp_mean = final_gmm.means_[i, 0]
            comp_std = np.sqrt(final_gmm.covariances_[i, 0, 0])
            comp_weight = final_gmm.weights_[i]
            
            label_text = f'Comp {i + 1} (μ={comp_mean:.2f})'
            if comp_mean == component_A_mean:
                label_text = f'Comp A (μ={comp_mean:.2f})'
            elif component_B_mean is not None and comp_mean == component_B_mean:
                label_text = f'Comp B (μ={comp_mean:.2f})'

            axs[1].plot(x_plot.flatten(), comp_weight * norm.pdf(x_plot.flatten(), comp_mean, comp_std),
                        linestyle='--', color=colors[i % len(colors)], label=label_text)

        if threshold_fifty is not None:
            axs[1].axvline(threshold_fifty, color='purple', linestyle='-', lw=3, label=f'Threshold 50%: {threshold_fifty:.2f}')
        if threshold_eighty is not None:
            axs[1].axvline(threshold_eighty, color='orange', linestyle='-', lw=3, label=f'Threshold 80%: {threshold_eighty:.2f}')

        axs[1].set_xlabel('Eye Angle')
        axs[1].set_ylabel('Density')
        axs[1].set_title(f'GMM Fit and Thresholds for {fish_name} ({elbow_point} components)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'{BIMODAL_FIT_PLOTS_DIR}{fish_name}.png', dpi=300)
        plt.close(fig)

    return {
        'fish_name': fish_name,
        'selected_components': elbow_point,
        'threshold_fifty': threshold_fifty,
        'threshold_eighty': threshold_eighty,
        'gmm_weights': final_gmm.weights_,
        'gmm_means': final_gmm.means_,
        'gmm_covariances': final_gmm.covariances_
    }

def calculate_threshold(bouts_info, periods_1, periods_2):
    """
    Calculate thresholds for all fish in bouts_info.
    """
    fish_names = list(bouts_info.keys())
    for fish_name in fish_names:
        if fish_name in ['20250718-F1', '20250718-F3', '20250718-F4']:
            periods = periods_1.copy()
        else:
            periods = periods_2.copy()
            
        set_ids = list(bouts_info[fish_name].keys())
        eye_angles_all = []
        for set_id in set_ids:
            h5_path = f"{BEHAVIOUR_BASE_DIR}{fish_name}/TOPCAMERA/{set_id}/{fish_name.replace('-','_')}_Trial1.h5"
            try:
                df_eye = pd.read_hdf(h5_path, "eye")
                eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
                eye_angles_filt = low_pass_filt(eye_angles, 300, 4)
                left_angle = eye_angles_filt[:, 0]
                right_angle = eye_angles_filt[:, 1]
                bino_eye_angles = left_angle + right_angle

                for i in range(len(periods)):
                    start = periods[i][0] * 300
                    end = periods[i][1] * 300
                    eye_angles_all.append(bino_eye_angles[start: end])
            except FileNotFoundError:
                print(f"File not found: {h5_path}")
                continue

        if eye_angles_all:
            eye_angles_fish = np.concatenate(eye_angles_all)
            dict_fish = detect_hunting_episodes_elbow(eye_angles_fish, fish_name)
            if dict_fish:
                dict_fish = convert_numpy_to_python(dict_fish)
                with open(f'{CONVERGENCE_DICT_DIR}{fish_name}.json', "w") as json_file:
                    json.dump(dict_fish, json_file, indent=4)

def find_segments(arr, x, y):
    """
    Find segments in array where values satisfy thresholds x and y.
    Merges segments closer than 300 units.
    """
    if x is None:
        return []

    # Step 1: Find all continuous segments that satisfy >= x
    mask_x = arr >= x
    padded_mask = np.concatenate(([False], mask_x, [False]))
    diff = np.diff(padded_mask.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    if len(starts) == 0:
        return []

    # Step 2: Filter based on y value
    segments = []
    if y is None or y <= x:
        segments = [[s, e] for s, e in zip(starts, ends)]
    else:
        mask_y = arr >= y
        cumsum = np.concatenate(([0], np.cumsum(mask_y).reshape(-1)))
        for s, e in zip(starts, ends):
            segment_len = e - s + 1
            count_good = cumsum[e + 1] - cumsum[s]
            if 2 * count_good >= segment_len:
                segments.append([s, e])

    if not segments:
        return []

    # Step 3: Merge segments
    segments = [[int(s), int(e)] for s, e in segments]
    merged_segments = [segments[0]]
    for current_start, current_end in segments[1:]:
        last_merged_end = merged_segments[-1][1]
        if current_start - last_merged_end < 300:
            merged_segments[-1][1] = max(last_merged_end, current_end)
        else:
            merged_segments.append([current_start, current_end])

    return merged_segments

def find_convergence_periods(bouts_info, hunting_periods):
    """
    Find convergence periods based on calculated thresholds.
    """
    fish_names = list(bouts_info.keys())
    print('Fish names:', fish_names)
    fish_conv_dict = {}
    
    for fish_name in fish_names:
        try:
            with open(f'{CONVERGENCE_DICT_DIR}{fish_name}.json', 'r') as file:
                threshold_info = json.load(file)
        except FileNotFoundError:
            print(f"Threshold info not found for {fish_name}")
            continue
            
        threshold_fifty = threshold_info.get('threshold_fifty')
        threshold_eighty = threshold_info.get('threshold_eighty')

        convergence_periods_dict = {}
        set_ids = list(bouts_info[fish_name].keys())
        
        for set_id in set_ids:
            if threshold_fifty:
                h5_path = f"{BEHAVIOUR_BASE_DIR}{fish_name}/TOPCAMERA/{set_id}/{fish_name.replace('-', '_')}_Trial1.h5"
                try:
                    df_eye = pd.read_hdf(h5_path, "eye")
                    eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
                    eye_angles_filt = low_pass_filt(eye_angles, 300, 4)
                    left_angle = eye_angles_filt[:, 0]
                    right_angle = eye_angles_filt[:, 1]
                    bino_eye_angles = left_angle + right_angle

                    segments = find_segments(bino_eye_angles, threshold_fifty, threshold_eighty)

                    new_segments = []
                    for segment in segments:
                        start, end = segment
                        is_hunting = False
                        for h_period in hunting_periods:
                            if h_period[0] <= segment[0]/300 <= h_period[1]:
                                is_hunting = True
                                break
                        if is_hunting:
                            new_segments.append([start, end])

                    convergence_periods_dict[set_id] = new_segments
                except FileNotFoundError:
                    print(f"H5 file not found: {h5_path}")
                    convergence_periods_dict[set_id] = []
            else:
                convergence_periods_dict[set_id] = []

        fish_conv_dict[fish_name] = convergence_periods_dict

    fish_conv_dict = convert_numpy_to_python(fish_conv_dict)
    with open(CONVERGENCE_PERIODS_OUTPUT, "w") as json_file:
        json.dump(fish_conv_dict, json_file, indent=4)

def plot_eye_angles(bouts_info, periods_1, stimuli_1, periods_2, stimuli_2):
    """
    Plot eye angles and convergence periods.
    """
    fish_names = list(bouts_info.keys())
    try:
        with open(CONVERGENCE_PERIODS_OUTPUT, 'r') as file:
            segments = json.load(file)
    except FileNotFoundError:
        print("Convergence periods file not found.")
        return

    for fish_name in fish_names:
        if fish_name in ['20250718-F1', '20250718-F3', '20250718-F4']:
            periods = periods_1.copy()
            stimuli = stimuli_1.copy()
        else:
            periods = periods_2.copy()
            stimuli = stimuli_2.copy()

        set_ids = list(bouts_info[fish_name].keys())
        for set_id in set_ids:
            if fish_name in segments and set_id in segments[fish_name]:
                convergence_segments = segments[fish_name][set_id]
            else:
                convergence_segments = []

            h5_path = f"{BEHAVIOUR_BASE_DIR}{fish_name}/TOPCAMERA/{set_id}/{fish_name.replace('-', '_')}_Trial1.h5"
            try:
                df_eye = pd.read_hdf(h5_path, "eye")
            except FileNotFoundError:
                continue

            eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
            eye_angles_filt = low_pass_filt(eye_angles, 300, 4)
            left_angle = eye_angles_filt[:, 0]
            right_angle = eye_angles_filt[:, 1]

            fig, axs = plt.subplots(3, 1, figsize=(110, 15), sharex=False)
            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]

            plt.title(f'{fish_name} {set_id}', fontsize=18)

            for (start, end), label in zip(periods, stimuli):
                start = start * 3
                end = end * 3
                ax1.hlines(y=0, xmin=start, xmax=end, color='black', linewidth=4)
                center = (start + end) / 2
                ax1.text(center, 0.01, label,
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), fontsize=18)

            ax1.set_xlim(0, 2344) # Note: this looks like a hardcoded limit from original code

            ax2.plot(left_angle, color='blue', label='Left Eye Angle')
            ax2.plot(right_angle, color='red', label='Right Eye Angle')

            for segment in convergence_segments:
                start, end = segment
                ax2.axvspan(start, end, color='yellow', alpha=0.3)

            ax2.legend()
            ax2.set_xlim(0, left_angle.shape[0])

            ax3.plot(left_angle + right_angle, color='blue', label='Binocular Eye Angle')
            for segment in convergence_segments:
                start, end = segment
                ax3.axvspan(start, end, color='yellow', alpha=0.3)
            ax3.set_xlim(0, left_angle.shape[0])

            plt.savefig(f'D:/2p astrocyte/behaviour/convergence periods/{fish_name}_{set_id}_eye_angles.png')
            plt.close(fig)

if __name__ == "__main__":
    # Load configuration
    try:
        with open(BOUTS_INFO_PATH, 'r') as file:
            bouts_info = json.load(file)
        
        periods_1, stimuli_1 = get_stimulus(1)
        periods_2, stimuli_2 = get_stimulus(10)
        hunting_periods = [[30, 32], [150, 152], [270, 272]]
        
        # Uncomment the functions you want to run
        # calculate_threshold(bouts_info, periods_1, periods_2)
        # find_convergence_periods(bouts_info, hunting_periods)
        # plot_eye_angles(bouts_info, periods_1, stimuli_1, periods_2, stimuli_2)
        
        pass 
    except FileNotFoundError as e:
        print(f"Error loading configuration files: {e}")
