"""
Spatial p-value Analysis for Astrocyte Patterns

This script calculates spatial p-values for various astrocyte activity patterns (Hunting, Passivity,
Struggle, Loom, Locomotion) and visualizes the results on a brain template.

It includes functions to:
- Load coordinate and mask data.
- Calculate spatial p-values using permutation tests.
- Visualize significant neurons on a 3D brain template (Napari/BrainGlobe).
- Generate summary plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.patches import Circle
from scipy.stats import norm
from skimage.segmentation import flood_fill
from joblib import Parallel, delayed
from tqdm import tqdm

# Add project root to path if necessary
sys.path.append("..")

try:
    from napari import Viewer
    from utils.brainglobe import AnatomicalPoints, AnatomicalSpace, AnatomicalStack, Atlas
    from utils.brain import BrainGrid
except ImportError as e:
    print(f"Warning: Could not import some specialized libraries (napari, utils.brainglobe, etc.): {e}")
    # Define dummy classes/functions if strictly necessary for static analysis,
    # but runtime will fail if these are missing.

# =============================================================================
# Constants and Configuration
# =============================================================================

BASE_DIR = "D:/2p astrocyte"
IMAGING_DIR = os.path.join(BASE_DIR, "imaging")
TEMPLATE_COORDS_DIR = os.path.join(IMAGING_DIR, "final_template_coordinates")
REGISTERED_DIR = os.path.join(IMAGING_DIR, "Registered 10-fish")
P_VALS_DIR = os.path.join(IMAGING_DIR, "p_vals_final")
HUC_TEMPLATE_DIR = os.path.join(IMAGING_DIR, "huc_template_coords")
PLOT_OUTPUT_DIR = os.path.join(IMAGING_DIR, "huc_template_registered_plot_final")
COMBINED_PLOT_DIR = os.path.join(IMAGING_DIR, "huc_template_registered_plot/Combined")
NEURON_COORDS_PATH = "D:/posters/CSHL/neuron_xyz.npy"

# Ensure directories exist
for d in [P_VALS_DIR, PLOT_OUTPUT_DIR, COMBINED_PLOT_DIR, HUC_TEMPLATE_DIR]:
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError:
            pass

# Pattern Definitions
HUNTING_DICT = {
    '20250807-F1': '90_component_29_intersection_3_4',
    '20250807-F2': '90_component_50_intersection_1_2',
    '20250807-F3': '90_component_76_intersection_2_3',
    '20250807-F4': '90_component_61_intersection_2_3',
    '20250808-F1': '70_component_30_intersection_2_3'
}

PASSIVITY_DICT = {
    '20250807-F1': '90_component_43_intersection_2_3',
    '20250807-F3': '90_component_7_intersection_1_2',
    '20250807-F4': '90_component_37_intersection_2_3',
    '20250808-F1': '70_component_23_intersection_1_2'
}

STRUGGLE_DICT = {
    '20250807-F2': '90_component_16_intersection_3_4',
    '20250807-F3': '90_component_18_intersection_1_2',
    '20250808-F1': '70_component_7_intersection_2_3'
}

LOOM_DICT = {
    '20250807-F1': '90_component_73_intersection_2_3',
    '20250807-F2': '90_component_9_intersection_2_3',
    '20250807-F3': '90_component_42_intersection_2_3',
    '20250807-F4': '90_component_42_intersection_1_2',
    '20250808-F1': '70_component_9_intersection_2_3'
}

LOCOMOTION_1_DICT = {
    '20250807-F1': '90_component_84_intersection_3_4',
    '20250807-F2': '90_component_18_intersection_3_4',
    '20250807-F3': '90_component_16_intersection_3_4',
    '20250807-F4': '90_component_26_intersection_2_3',
    '20250808-F1': '70_component_30_intersection_2_3'
}

LOCOMOTION_2_DICT = {
    '20250807-F1': '90_component_26_intersection_3_4',
    '20250807-F2': '90_component_68_intersection_2_3',
    '20250807-F3': '90_component_26_intersection_2_3',
    '20250807-F4': '90_component_13_intersection_3_4',
    '20250808-F1': '70_component_11_intersection_2_3'
}

PATTERN_COLORS = {
    'Hunting': 'red',
    'Passivity': 'deepskyblue',
    'Struggle': 'fuchsia',
    'Loom': 'orange',
    'locomotion': 'green'
}


# =============================================================================
# Helper Functions
# =============================================================================

def check_is_in_brain(points: AnatomicalPoints):
    """
    Check if points are within the brain structure, excluding specific regions.
    """
    exclude_structures = [
        "olfactory epithelium",
        "retina",
        "anterior lateral line ganglion",
        "trigeminal ganglion",
        "posterior lateral line ganglion",
        "octaval ganglion",
        "glossopharyngeal ganglion",
        "peripheral nervous system",
    ]

    annotation = AnatomicalSpace("asr").map_stack_to("sal", Atlas().annotation.copy())
    annotation = flood_fill(annotation, (0, 0, 0), 1)

    for i in exclude_structures:
        annotation[annotation == i] = 1

    annotation = AnatomicalStack(AnatomicalSpace("sal"), annotation)
    return annotation.sel(points, out_of_range_value=1) != 1


def shuffle(x):
    """Shuffle an array in place and return it."""
    np.random.shuffle(x)
    return x


def shuffle_coords(coords, neuron_fish_id):
    """Shuffle coordinates within each fish group."""
    coords_random = coords.copy()
    fish_list = np.unique(neuron_fish_id)

    for i_fish in fish_list:
        mask = neuron_fish_id == i_fish
        coords_random[mask] = shuffle(coords_random[mask])

    return coords_random


def get_distance(coords, accepted, neuron_fish_id, randomize=False, symmetric=False, ml=282):
    """
    Calculate distances between accepted points and other points.
    """
    fish_list = np.unique(neuron_fish_id)
    dr = []

    if randomize:
        coords2 = shuffle_coords(coords, neuron_fish_id)
    else:
        coords2 = coords

    coords2_flip = coords2.copy()
    coords2_flip[:, 2] = 2 * ml - coords2_flip[:, 2]

    for i_fish in fish_list:
        coords_i = coords[accepted & (neuron_fish_id == i_fish)]

        def get_dist_j(c_i, j_fish):
            bidx = accepted & (neuron_fish_id == j_fish)
            if symmetric:
                return np.minimum(
                    np.linalg.norm(c_i[:, None] - coords2[bidx], axis=-1).min(-1, initial=np.inf),
                    np.linalg.norm(c_i[:, None] - coords2_flip[bidx], axis=-1).min(-1, initial=np.inf),
                )
            return np.linalg.norm(
                c_i[:, None] - coords2[bidx],
                axis=-1,
            ).min(-1, initial=np.inf)

        dr.append(np.column_stack([get_dist_j(coords_i, j_fish) for j_fish in fish_list]))

    return np.concatenate(dr)


def proj(arr, n_outliers=1):
    """Project array by removing outliers and taking mean."""
    arr = np.sort(arr, axis=-1)
    return arr[..., : arr.shape[-1] - n_outliers].mean(-1)


def norm_fit(x):
    """Fit a normal distribution to finite values in x."""
    return norm.fit(x[np.isfinite(x)])


def get_p_value(
    coords,
    accepted,
    neuron_fish_id,
    n_samples=1000,
    n_outliers=2,
    verbose=0,
    random_state=None,
    symmetric=False,
    ml=282,
):
    """
    Calculate spatial p-values using permutation testing.
    """
    if random_state is not None:
        np.random.seed(random_state)

    coords = np.asarray(coords)
    accepted = np.asarray(accepted)
    neuron_fish_id = np.asarray(neuron_fish_id)

    Dr = get_distance(coords, accepted, neuron_fish_id, False, symmetric, ml)
    it = tqdm(range(n_samples)) if verbose else range(n_samples)
    
    Dn = np.array(
        Parallel(max_nbytes=None, n_jobs=-1)(
            delayed(get_distance)(coords, accepted, neuron_fish_id, True, symmetric, ml) for _ in it
        )
    )
    
    dr = proj(Dr, n_outliers)
    dn = proj(Dn, n_outliers).T
    norm_args = np.apply_along_axis(norm_fit, axis=-1, arr=dn)
    pval = np.full(len(accepted), np.nan)
    pval[accepted] = norm(*norm_args.T).cdf(dr)
    return pval


def collect_points_for_pattern(dicts_or_tuple, pattern_name, pattern_types=None):
    """
    Collect coordinates and calculate p-values for a specific pattern across all fish.
    
    Args:
        dicts_or_tuple: A dictionary {fish_name: component_info} or a tuple of two such dictionaries.
        pattern_name: Name of the pattern (e.g., 'Hunting').
        pattern_types: Optional list of patterns to save specific mask files.
    
    Returns:
        List of tuples: (fish_name, coordinate) for points with p_val < threshold (filtering done later usually,
        but this function returns data structure to be processed). 
        WAIT: The original function returned a specific structure. 
        Let's stick to the logic: calculate p-values and return filtered points.
    """
    # Determine fish names
    if isinstance(dicts_or_tuple, tuple):
        fish_names = list(dicts_or_tuple[0].keys())
    else:
        fish_names = list(dicts_or_tuple.keys())

    coord_list = []
    mask_list = []
    fish_ids = []
    
    fish_id_counter = 0
    
    for fish_n in fish_names:
        # Load coordinates
        coord_file = os.path.join(TEMPLATE_COORDS_DIR, fish_n, 'coordinates.csv')
        df = pd.read_csv(coord_file)
        coords = df[['x', 'y', 'z']].values
        
        # Load SNR mask
        snr_mask_file = os.path.join(REGISTERED_DIR, fish_n, 'snr_mask.npy')
        snr_mask = np.load(snr_mask_file)
        
        # Load Brain mask
        is_in_brain_file = os.path.join(TEMPLATE_COORDS_DIR, fish_n, 'is_in_brain.npy')
        is_in_brain = np.load(is_in_brain_file)
        
        # Combine masks: SNR & InBrain
        # Save intermediate mask if needed
        if pattern_types and len(pattern_types) == 1:
            new_mask = [is_in_brain[k] for k in range(len(snr_mask)) if snr_mask[k]]
            np.save(os.path.join(P_VALS_DIR, f'{fish_n}_{pattern_types[0]}_inbrain.npy'), new_mask)

        combined_base = snr_mask & is_in_brain
        coords_cb = coords[combined_base]
        coord_list.append(coords_cb)
        
        # Load Component Mask(s)
        if isinstance(dicts_or_tuple, tuple):
            d1, d2 = dicts_or_tuple
            
            def get_mask(d, fn):
                v = d[fn]
                c_num = v.split('_')[0]
                c_idx = v.split(f'{c_num}_')[1]
                path = os.path.join(REGISTERED_DIR, fn, 'intersection_masks_full_sets_Energy ratio_time_corrected', c_num, f'{c_idx}.npy')
                return np.load(path)
            
            m1 = get_mask(d1, fish_n)
            m2 = get_mask(d2, fish_n)
            
            # Combine component masks, then filter by InBrain (aligned with SNR)
            is_in_brain_snr = is_in_brain[snr_mask]
            mask = (m1 | m2)[is_in_brain_snr]
            
        else:
            v = dicts_or_tuple[fish_n]
            c_num = v.split('_')[0]
            c_idx = v.split(f'{c_num}_')[1]
            path = os.path.join(REGISTERED_DIR, fish_n, 'intersection_masks_full_sets_Energy ratio_time_corrected', c_num, f'{c_idx}.npy')
            raw_mask = np.load(path)
            
            is_in_brain_snr = is_in_brain[snr_mask]
            mask = raw_mask[is_in_brain_snr]

        mask_list.append(mask)
        fish_ids.append(np.full(len(mask), fish_id_counter))
        fish_id_counter += 1

    # Calculate p-values globally
    coords_all = np.concatenate(coord_list, axis=0)
    masks_all = np.concatenate(mask_list, axis=0)
    fish_ids_all = np.concatenate(fish_ids, axis=0)

    p_val = get_p_value(coords_all, masks_all, fish_ids_all)

    # Re-distribute p-values to fish and return detailed info
    # Return structure: list of (fish_name, coords, mask, p_vals_for_this_fish)
    # But to match original logic which filtered immediately:
    
    results = []
    fish_id_counter = 0
    for fish_n in fish_names:
        p_val_temp = p_val[fish_ids_all == fish_id_counter]
        
        # Save p-values if single pattern
        if pattern_types and len(pattern_types) == 1:
            np.save(os.path.join(P_VALS_DIR, f'{fish_n}_{pattern_types[0]}_p_vals.npy'), p_val_temp)
            
        coords_snr = coord_list[fish_id_counter]
        mask = mask_list[fish_id_counter]
        
        results.append({
            'fish_name': fish_n,
            'coords': coords_snr,
            'mask': mask,
            'p_vals': p_val_temp
        })
        fish_id_counter += 1
        
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_astroglia_in_template(astroglia_xyz, p_vals, figure_name, pattern_type=None):
    """
    Plot astroglia on a brain template with colors based on p-values.
    """
    atlas = Atlas()
    mesh = atlas.root_mesh()

    viewer = Viewer()
    viewer.theme = 'light'
    viewer.dims.ndisplay = 3
    viewer.add_surface((atlas.space.map_points_to("sal", mesh.points), mesh.cells_dict["triangle"]),
                       blending="translucent_no_depth", shading="smooth", opacity=0.09)

    # Normalize p-values and generate colors
    p_vals = np.asarray(p_vals)
    n_points = astroglia_xyz.shape[0]
    if p_vals.shape[0] != n_points:
        n = min(p_vals.shape[0], n_points)
        p_vals = p_vals[:n]
        astroglia_xyz = astroglia_xyz[:n]

    vmin = np.nanmin(p_vals)
    vmax = np.nanmax(p_vals)
    
    cmap = cm.get_cmap('hot', 256)
    trunc = cmap(np.linspace(0.05, 0.95, 256))
    cmap_trunc = mcolors.LinearSegmentedColormap.from_list('hot_trunc', trunc)
    
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        norm_obj = mcolors.Normalize(vmin=vmin - 1e-12, vmax=vmax + 1e-12)
        colors_hot = np.tile(np.array([[1.0, 0.5, 0.0, 1.0]]), (p_vals.shape[0], 1))
    else:
        norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors_hot = cmap_trunc(norm_obj(p_vals))

    # Blend with blue based on p-value
    blue = np.array([0.0, 0.0, 1.0])
    if p_vals.shape[0] == colors_hot.shape[0]:
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            w = np.zeros(p_vals.shape[0])
        else:
            w = (p_vals - vmin) / (vmax - vmin)
        rgb = (1 - w)[:, None] * colors_hot[:, :3] + w[:, None] * blue[None, :]
        face_colors = np.concatenate([rgb, colors_hot[:, 3:4]], axis=1)
    else:
        face_colors = colors_hot

    # Add points to viewer
    viewer.add_points(
        astroglia_xyz[:, ::-1],
        size=15,
        opacity=0.5,
        face_color=face_colors,
        border_width=0,
        name="Astroglia",
    )

    # Capture screenshots
    viewer.camera.zoom = 1
    viewer.camera.angles = (0, 0, 0)
    im_front = viewer.screenshot(size=(1500, 1500))
    
    viewer.camera.angles = (0, 0, -90)
    im_top = viewer.screenshot(size=(1500, 1500))
    
    viewer.camera.angles = (-90, 90, 0)
    im_left = viewer.screenshot(size=(1500, 1500))

    # Save logic
    save_dir = os.path.join(PLOT_OUTPUT_DIR, str(pattern_type))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.imsave(f"{save_dir}/{figure_name}_top.png", im_top)
    plt.imsave(f"{save_dir}/{figure_name}_front.png", im_front)
    plt.imsave(f"{save_dir}/{figure_name}_left.png", im_left)

    # Create 4-grid plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
        fig.suptitle(str(figure_name), fontsize=25, fontweight='bold')

        axes[0, 0].imshow(im_front)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Front', fontsize=18)

        # Colorbar
        from matplotlib.cm import ScalarMappable
        
        # Create blended colormap for colorbar
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            cmap_points = mcolors.ListedColormap([[1.0, 0.5, 0.0, 1.0]], name='points_constant')
        else:
            t = np.linspace(0.0, 1.0, 256)
            base = cmap_trunc(t)
            rgb_blend = (1 - t)[:, None] * base[:, :3] + t[:, None] * blue[None, :]
            colors_for_cbar = np.concatenate([rgb_blend, base[:, 3:4]], axis=1)
            cmap_points = mcolors.ListedColormap(colors_for_cbar, name='hot_blue_blend')

        sm = ScalarMappable(cmap=cmap_points, norm=norm_obj)
        sm.set_array([])
        
        try:
            cbar = plt.colorbar(sm, ax=axes[0, 1], fraction=0.046, pad=0.04, location='left')
        except Exception:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axes[0, 1])
            cax = divider.append_axes("left", size="5%", pad=0.2)
            cbar = plt.colorbar(sm, cax=cax)
            cax.yaxis.set_ticks_position('left')
            cax.yaxis.set_label_position('left')
            
        cbar.set_label('p-values', fontsize=18)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(im_top)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Top', fontsize=18)

        axes[1, 1].imshow(im_left)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Left', fontsize=18)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_dir}/{figure_name}_4grid.png", bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"[4-grid] Failed to compose 2x2 figure: {e}")
    
    viewer.close()


def plot_all_patterns_pvals_threshold_on_template(p_threshold=0.5, figure_name=None, pattern_types=None, background='white'):
    """
    Plot significant points for multiple patterns on a single template.
    """
    if figure_name is None:
        figure_name = f"Threshold_p_val_{p_threshold}"

    # Select patterns
    allowed_patterns = {'Hunting', 'Passivity', 'Struggle', 'Loom', 'locomotion'}
    if pattern_types is None:
        selected_patterns = list(allowed_patterns)
    else:
        selected_patterns = [p for p in pattern_types if p in allowed_patterns]
        if not selected_patterns:
            print("[Select] No valid pattern_types provided.")
            return

    # Collect points for each pattern
    points_by_pattern = {}
    
    mapping = {
        'Hunting': HUNTING_DICT,
        'Passivity': PASSIVITY_DICT,
        'Struggle': STRUGGLE_DICT,
        'Loom': LOOM_DICT,
        'locomotion': (LOCOMOTION_1_DICT, LOCOMOTION_2_DICT)
    }

    for pat in selected_patterns:
        data_source = mapping[pat]
        results = collect_points_for_pattern(data_source, pat, pattern_types=pattern_types)
        
        # Filter by threshold and mask
        valid_points = []
        for res in results:
            mask = res['mask']
            p_vals = res['p_vals']
            coords = res['coords']
            
            # Select points where mask is True AND p_val < threshold
            sel = mask & (p_vals < p_threshold)
            valid_points.append(coords[sel])
            
        points_by_pattern[pat] = np.concatenate(valid_points, axis=0) if valid_points else np.empty((0, 3))

    # Aggregate all coordinates and colors
    all_coords = []
    all_colors = []
    
    for pat, pts in points_by_pattern.items():
        if len(pts) == 0:
            continue
        color_name = PATTERN_COLORS.get(pat, 'gray')
        rgba = mcolors.to_rgba(color_name)
        all_coords.append(pts)
        all_colors.append(np.tile(rgba, (len(pts), 1)))

    if not all_coords:
        print(f"[Combined] No points found with p_val < {p_threshold}.")
        return

    all_coords = np.concatenate(all_coords, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Plotting setup
    atlas = Atlas()
    mesh = atlas.root_mesh()
    
    bg = str(background).lower()
    theme = 'dark' if bg == 'black' else 'light'
    
    viewer = Viewer()
    viewer.theme = theme
    viewer.dims.ndisplay = 3
    viewer.add_surface((atlas.space.map_points_to("sal", mesh.points), mesh.cells_dict["triangle"]),
                       blending="translucent_no_depth", shading="smooth", opacity=0.09)

    # Add Hunting Neuron (if available)
    if os.path.exists(NEURON_COORDS_PATH):
        try:
            neuron_coords = np.load(NEURON_COORDS_PATH)
            # Map to brain logic (simplified from original)
            neuron_points = AnatomicalPoints("sal", neuron_coords[:, :])
            # Check if in brain logic...
            # For simplicity, just adding points as original code did after check
            # Assuming pre-check or simple add for visualization
            
            # Original code had logic to filter neurons. 
            # Re-implementing simplified version:
            c = AnatomicalPoints("sal", neuron_coords[:, [2, 1, 0]])
            neuron_is_in_brain = check_is_in_brain(c)
            
            atlas_obj = Atlas()
            no_eye_mask = ~np.isin(
                atlas_obj.points_to_structures(c),
                atlas_obj.get_structures_descendants(["peripheral nervous system", "retina"]),
            )
            main_mask = neuron_is_in_brain & no_eye_mask
            in_brain_neuron_coord = neuron_points[main_mask]
            
            # Save for record
            np.save(os.path.join(HUC_TEMPLATE_DIR, 'in_brain_neuron_coord.npy'), in_brain_neuron_coord)
            
            viewer.add_points(
                in_brain_neuron_coord[:, ::-1],
                size=10,
                opacity=0.9,
                face_color='#00FE00',
                border_width=0,
                name="Hunting neuron",
            )
        except Exception as e:
            print(f"Could not load/process neuron coords: {e}")

    # Add Astroglia points
    viewer.add_points(
        all_coords[:, ::-1],
        size=10,
        opacity=0.5,
        face_color=all_colors,
        border_width=0,
        name="All patterns (p>threshold)",
    )

    # Screenshots
    viewer.camera.zoom = 1
    viewer.camera.angles = (0, 0, -90)
    im_top = viewer.screenshot(size=(1500, 1500))
    
    viewer.camera.angles = (0, 0, 0)
    im_front = viewer.screenshot(size=(1500, 1500))
    
    viewer.camera.angles = (-90, 90, 0)
    im_left = viewer.screenshot(size=(1500, 1500))

    if not os.path.exists(COMBINED_PLOT_DIR):
        os.makedirs(COMBINED_PLOT_DIR)

    plt.imsave(f"{COMBINED_PLOT_DIR}/hunting_neuron_astroglia_top.png", im_top)
    plt.imsave(f"{COMBINED_PLOT_DIR}/hunting_neuron_astroglia_front.png", im_front)
    plt.imsave(f"{COMBINED_PLOT_DIR}/hunting_neuron_astroglia_left.png", im_left)

    # 2x2 Grid with Legend
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    if theme == 'dark':
        fig.patch.set_facecolor('black')
        for ax in axes.flat:
            ax.set_facecolor('black')

    title_color = 'white' if theme == 'dark' else 'black'
    
    axes[0, 0].imshow(im_front)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Front', fontsize=18, color=title_color)

    axes[1, 0].imshow(im_top)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Top', fontsize=18, color=title_color)

    axes[1, 1].imshow(im_left)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Left', fontsize=18, color=title_color)

    # Legend
    axes[0, 1].axis('off')
    if theme == 'dark':
        axes[0, 1].set_facecolor('black')
        
    legend_lines = [(name, PATTERN_COLORS.get(name, 'gray')) for name in selected_patterns]
    y0 = 0.1
    dy = 0.13
    dot_x = 0.05
    text_x = 0.08
    dot_r = 0.015
    legend_text_color = 'white' if theme == 'dark' else 'black'
    
    for i, (name, col) in enumerate(legend_lines):
        y = y0 + i * dy
        axes[0, 1].add_patch(Circle((dot_x, y), dot_r, transform=axes[0, 1].transAxes,
                                     facecolor=col, edgecolor='none'))
        axes[0, 1].text(text_x, y, name, color=legend_text_color, fontsize=16, transform=axes[0, 1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{COMBINED_PLOT_DIR}/hunting_neuron_astroglia.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    viewer.close()


def calculate_and_plot_single_pattern(dicts, pattern_name):
    """
    Wrapper to calculate p-values and plot for a single pattern (legacy support).
    """
    results = collect_points_for_pattern(dicts, pattern_name)
    for res in results:
        fish_n = res['fish_name']
        mask = res['mask']
        p_vals = res['p_vals']
        coords = res['coords']
        
        plot_astroglia_in_template(coords[mask], p_vals[mask], fish_n, pattern_name)


if __name__ == "__main__":
    # Example usage:
    # 1. Calculate and plot individual patterns
    # calculate_and_plot_single_pattern(HUNTING_DICT, 'Hunting')
    # calculate_and_plot_single_pattern(PASSIVITY_DICT, 'Passivity')
    
    # 2. Plot combined patterns
    plot_all_patterns_pvals_threshold_on_template(p_threshold=0.05, pattern_types=['Hunting'])
