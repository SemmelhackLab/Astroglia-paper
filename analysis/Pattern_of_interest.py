import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Constants for file paths
BASE_DIR = os.environ.get("ASTROCYTE_DATA_ROOT", r"D:/2p astrocyte")
IMAGING_DIR = os.path.join(BASE_DIR, "imaging")
BEHAVIOUR_DIR = os.path.join(BASE_DIR, "behaviour")
REGISTERED_FISH_DIR = os.path.join(IMAGING_DIR, "Registered 10-fish")
OUTPUT_BASE_DIR = os.path.join(IMAGING_DIR, "pattern_of_interest_significance_analysis (shuffle)")

STIMULUS_INFO_TEST = os.path.join(BEHAVIOUR_DIR, "2p-test-20250718-info.xlsx")
STIMULUS_INFO_10_FISH = os.path.join(BEHAVIOUR_DIR, "10 fish data/stimulus_info.xlsx")
CONVERGENCE_PERIODS_PATH = os.path.join(BEHAVIOUR_DIR, "convergence periods/convergence_periods.json")
PASSIVITY_PATH = os.path.join(BEHAVIOUR_DIR, "passivity/passivity.json")


def get_stimulus(fish_name):
    """
    根据鱼的编号读取刺激信息，返回所有刺激的时间段与类型列表。

    Args:
        fish_name (str): 鱼的唯一标识。

    Returns:
        tuple:
            - periods (list): 刺激时间段列表，每项为 [start, end]（单位秒）。
            - stimuli (list): 刺激类型列表（如 'loom' 等），与 periods 对齐。
    """
    if fish_name in ['20250718-F1', '20250718-F3', '20250718-F4']:
        file_path = STIMULUS_INFO_TEST
        df = pd.read_excel(file_path, engine="openpyxl")
        
        start = df["start"].tolist()
        end = df["end"].tolist()
        stimuli = df["stimuli"].tolist()

        periods = []
        for i in range(len(stimuli)):
            periods.append([start[i], end[i]])

        periods.append([450, 510])
        periods.append([750, 810])
    else:
        file_path = STIMULUS_INFO_10_FISH
        df = pd.read_excel(file_path, engine="openpyxl")

        start = df["start"].tolist()
        end = df["end"].tolist()
        stimuli = df["stimuli"].tolist()

        periods = []
        for i in range(len(stimuli)):
            periods.append([start[i], end[i]])

    return periods, stimuli


def find_nearest_indices(time_stamps, start, end):
    """
    在时间戳数组中找到最接近区间 [start, end] 的索引范围。

    Args:
        time_stamps (np.ndarray): 1D 数组，递增时间点。
        start (float): 期望的开始时间（秒）。
        end (float): 期望的结束时间（秒）。

    Returns:
        tuple: (start_index, end_index) 分别是截取的起止索引（闭区间）。
    """
    start_index = 0
    for i in range(time_stamps.shape[0]):
        if time_stamps[i] >= start:
            start_index = i
            break

    end_index = 0
    for i in range(time_stamps.shape[0]):
        if time_stamps[i] >= end:
            end_index = i - 1
            break

    return start_index, end_index


def slice_matrix_by_time(X, time_stamps, periods):
    """
    按给定时间段在矩阵第二维进行切片。

    Args:
        X (np.ndarray): 2D 数组，形状 (N, T)。
        time_stamps (np.ndarray): 1D 数组，对应 X 的时间轴。
        periods (list): [start, end] 时间段（秒）。

    Returns:
        np.ndarray: X 在该时间段上的子矩阵（复制）。
    """
    time_stamps = np.asarray(time_stamps)
    start, end = periods
    start_idx, end_idx = find_nearest_indices(time_stamps, start, end)
    slice_data = X[:, start_idx:end_idx + 1].copy()
    return slice_data


def benjamini_hochberg(pvals, alpha):
    """
    Benjamini-Hochberg FDR 控制，返回被判定为显著的掩码。

    Args:
        pvals (np.ndarray): 1D p 值数组。
        alpha (float): 显著性水平。

    Returns:
        np.ndarray: 1D 布尔数组，True 表示对应位置被认为显著。
    """
    pvals = np.asarray(pvals)
    m = pvals.size
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    is_below = p_sorted <= thresholds
    
    if not np.any(is_below):
        mask_sorted = np.zeros(m, dtype=bool)
    else:
        k = np.max(np.where(is_below)[0])
        mask_sorted = np.zeros(m, dtype=bool)
        mask_sorted[:k + 1] = True
    
    mask = np.zeros(m, dtype=bool)
    mask[order] = mask_sorted
    return mask


def detect_significant_points(data, n_shuffles=10000, alpha=0.01, two_tailed=True, verbose=True, apply_fdr=False):
    """
    检测神经元时序数据中每个时间点的显著性。

    Args:
        data (np.ndarray): 原始数据矩阵，形状为 (N, T)。
        n_shuffles (int): 随机重排次数，默认 10000 次。
        alpha (float): 显著性水平，默认 0.05。
        two_tailed (bool): 是否使用双侧检验，默认 True。
        verbose (bool): 是否显示进度条，默认 True。
        apply_fdr (bool): 是否应用 FDR 校正。

    Returns:
        tuple:
            - significance (np.ndarray): 显著性矩阵 (-1: 低, 0: 不显著, 1: 高)。
            - p_values (np.ndarray): p 值矩阵。
            - null_means (np.ndarray): null 分布的均值矩阵。
            - null_stds (np.ndarray): null 分布的标准差矩阵。
    """
    N, T = data.shape
    significance = np.zeros((N, T), dtype=int)
    p_values = np.zeros((N, T))
    null_means = np.zeros((N, T))
    null_medians = np.zeros((N, T))
    null_stds = np.zeros((N, T))

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    iterator = tqdm(range(N), desc="处理样本") if verbose else range(N)

    for i in iterator:
        original_trace = data[i, :]
        null_distribution = np.zeros((n_shuffles, T))

        for shuffle_idx in range(n_shuffles):
            shuffled_trace = np.random.permutation(original_trace)
            null_distribution[shuffle_idx, :] = shuffled_trace

        null_means[i, :] = np.mean(null_distribution, axis=0)
        null_medians[i, :] = np.median(null_distribution, axis=0)
        null_stds[i, :] = np.std(null_distribution, axis=0)

        for t in range(T):
            if two_tailed:
                percentile = np.sum(null_distribution[:, t] < original_trace[t]) / n_shuffles
                if percentile > 0.5:
                    p_val = 2 * (1 - percentile)
                else:
                    p_val = 2 * percentile

                if p_val < alpha:
                    if original_trace[t] > np.percentile(null_distribution[:, t], 100 * (1 - alpha / 2)):
                        significance[i, t] = 1
                    else:
                        significance[i, t] = -1
                else:
                    significance[i, t] = 0
            else:
                p_val_high = np.sum(null_distribution[:, t] >= original_trace[t]) / n_shuffles
                if p_val_high < alpha:
                    significance[i, t] = 1
                else:
                    p_val_low = np.sum(null_distribution[:, t] <= original_trace[t]) / n_shuffles
                    if p_val_low < alpha:
                        significance[i, t] = -1
                    else:
                        significance[i, t] = 0
                p_val = min(p_val_high, p_val_low)

            p_values[i, t] = p_val
            
        if apply_fdr:
            mask = benjamini_hochberg(p_values[i, :], alpha)
            row = np.zeros(T, dtype=int)
            pos = mask & (original_trace > null_medians[i, :])
            neg = mask & (original_trace < null_medians[i, :])
            row[pos] = 1
            row[neg] = -1
            significance[i, :] = row

    return significance, p_values, null_means, null_stds


def visualize_results(data, significance, sample_idx=0):
    """
    可视化单个样本的结果。
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 原始数据和显著性标记
    time_points = np.arange(data.shape[1])
    axes[0].plot(time_points, data[sample_idx, :], 'b-', label='原始数据', alpha=0.7)

    # 标记显著点
    sig_high = np.where(significance[sample_idx, :] == 1)[0]
    sig_low = np.where(significance[sample_idx, :] == -1)[0]

    if len(sig_high) > 0:
        axes[0].scatter(sig_high, data[sample_idx, sig_high],
                        color='red', s=50, label='显著高', zorder=5)
    if len(sig_low) > 0:
        axes[0].scatter(sig_low, data[sample_idx, sig_low],
                        color='green', s=50, label='显著低', zorder=5)

    axes[0].set_xlabel('时间点')
    axes[0].set_ylabel('神经活动强度')
    axes[0].set_title(f'样本 {sample_idx} 的显著性检测结果')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 显著性热图（前几个样本）
    n_samples_to_show = min(5, data.shape[0])
    im = axes[1].imshow(significance[:n_samples_to_show, :],
                        aspect='auto', cmap='coolwarm',
                        vmin=-1, vmax=1)
    axes[1].set_xlabel('时间点')
    axes[1].set_ylabel('样本索引')
    axes[1].set_title('显著性矩阵（-1:低, 0:不显著, 1:高）')
    plt.colorbar(im, ax=axes[1], ticks=[-1, 0, 1])

    plt.tight_layout()


def visualize_set_significance(sig_set, temp_indices, fish_name,
                               set_name, component_number, start, end, mode, event_index, highlight_rows=None):
    """
    可视化某个 set 在给定事件窗口内的显著性切片。
    """
    s_idx, e_idx = temp_indices
    s_idx = max(0, s_idx)
    e_idx = min(sig_set.shape[1] - 1, e_idx)
    
    sig_slice = sig_set[:, s_idx:e_idx + 1]
    sig_slice = np.where(sig_slice == -1, 0, sig_slice)
    
    cmap = ListedColormap(['white', 'red'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(sig_slice, aspect='auto', cmap=cmap, norm=norm)
    ax = plt.gca()
    
    L = sig_slice.shape[1]
    mid = L // 2
    ws = (end - start) / 2
    
    plt.xticks([0, mid, L - 1], [f'-{ws:.3g}', '0', f'{ws:.3g}'])
    plt.xlabel(f'Time from {mode} onset (s)', fontsize=15)
    plt.ylabel('Patterns', fontsize=15)
    plt.title(f'{fish_name} {set_name} {component_number} {mode} event {event_index}', fontsize=18)
    
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0', '1'])
    
    rows_with_high = np.where((sig_slice == 1).any(axis=1))[0]
    ax.set_yticks(rows_with_high)
    ax.set_yticklabels([str(r) for r in rows_with_high])
    
    if highlight_rows is not None:
        try:
            highlight_set = set(highlight_rows.tolist()) if hasattr(highlight_rows, "tolist") else set(highlight_rows)
        except TypeError:
            highlight_set = set()
        
        labels = ax.get_yticklabels()
        for idx, lbl in enumerate(labels):
            r = rows_with_high[idx]
            if r in highlight_set:
                lbl.set_color('red')
            else:
                lbl.set_color('black')
                
    plt.tight_layout()
    
    save_dir = os.path.join(OUTPUT_BASE_DIR, "heatmaps", mode, fish_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'{set_name}_{component_number}_{mode}_event_{event_index}.png'
    plt.savefig(os.path.join(save_dir, file_name), dpi=300)
    plt.close()


def visualize_random_null_distribution(data, n_shuffles=2000, alpha=0.05, sample_idx=None, t_idx=None):
    """
    随机选择一个样本与时间点，构造打乱分布并绘制直方图对比。
    """
    rng_i = np.random.randint(0, data.shape[0]) if sample_idx is None else sample_idx
    rng_t = np.random.randint(0, data.shape[1]) if t_idx is None else t_idx
    
    x = data[rng_i, rng_t]
    null_vals = np.empty(n_shuffles)
    
    for k in range(n_shuffles):
        shuffled = np.random.permutation(data[rng_i, :])
        null_vals[k] = shuffled[rng_t]
        
    percentile = (np.sum(null_vals < x) + 0.5 * np.sum(null_vals == x)) / n_shuffles
    p_val = 2 * (1 - percentile) if percentile > 0.5 else 2 * percentile
    
    q_low = np.quantile(null_vals, alpha / 2)
    q_high = np.quantile(null_vals, 1 - alpha / 2)
    med = np.median(null_vals)
    
    plt.figure(figsize=(10, 6))
    plt.hist(null_vals, bins=100, color='gray', alpha=0.7, label=f'Null distribution (n={n_shuffles})')
    
    sig_note = (
        f'significant high' if (p_val < alpha and x >= q_high) else
        f'significant low' if (p_val < alpha and x <= q_low) else
        f'not significant'
    )
    
    plt.axvline(x, color='black', linewidth=2, label=f'Observed value = {x:.3g} ({sig_note}, p={p_val:.3g})')
    plt.axvline(med, color='orange', linestyle='--', label=f'Null median = {med:.3g}')
    plt.axvline(q_low, color='blue', linestyle=':', label=f'Lower {alpha/2:.3g} quantile = {q_low:.3g}')
    plt.axvline(q_high, color='red', linestyle=':', label=f'Upper {1 - alpha/2:.3g} quantile = {q_high:.3g}')
    
    plt.xlabel('Value', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title(f'Null Distribution (sample {rng_i}, time {rng_t}), n_shuffles={n_shuffles}', fontsize=18)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_BASE_DIR, 'null_distribution.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return rng_i, rng_t, p_val


def compute_low_variability_flags(X_avg, X_full, bg_mask, threshold=0.25):
    """
    计算每个样本的“低波动”标记。
    """
    n = X_avg.shape[0]
    flags = np.zeros(n, dtype=bool)
    
    if np.count_nonzero(bg_mask) <= 1:
        return flags
        
    for i in range(n):
        bg = X_full[i, bg_mask]
        if bg.size <= 1:
            flags[i] = False
            continue
            
        std_bg = bg.std(ddof=1)
        amp_avg = np.mean(np.abs(X_avg[i])) if X_avg.shape[1] > 0 else 0.0
        denom = amp_avg if amp_avg > 1e-12 else 1e-12
        ratio = std_bg / denom
        print(f'{i}: ', ratio)
        flags[i] = ratio <= threshold
        
    return flags


def example_usage(fish_name, component_number):
    """
    示例用法：加载数据，运行显著性检测。
    """
    # 构造文件路径
    base_path = os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets")
    file_pattern = "Temporal_components_prey-loom-omr-set00{}_{}.npy"
    
    H_set1 = np.load(os.path.join(base_path, file_pattern.format(1, component_number)))
    H_set2 = np.load(os.path.join(base_path, file_pattern.format(2, component_number)))
    H_set3 = np.load(os.path.join(base_path, file_pattern.format(3, component_number)))
    H_set4 = np.load(os.path.join(base_path, file_pattern.format(4, component_number)))

    H_concatenated = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)

    # 运行算法
    visualize_random_null_distribution(H_concatenated, n_shuffles=5000, alpha=0.05)

    significance, p_values, null_means, null_stds = detect_significant_points(
        H_concatenated,
        n_shuffles=5000,
        alpha=0.05,
        two_tailed=True,
        verbose=True
    )

    # 输出结果统计
    print(f"\n结果统计:")
    print(f"显著高的点: {np.sum(significance == 1)}")
    print(f"显著低的点: {np.sum(significance == -1)}")
    print(f"不显著的点: {np.sum(significance == 0)}")
    
    visualize_random_null_distribution(H_concatenated, n_shuffles=2000, alpha=0.05)

    return significance, p_values


def find_low_variability_high_significance_samples(fish_name, component_number, significance, time_stamps, window_size=10, threshold=0.25, mode='hunting'):
    """
    筛选同时满足高显著性和低波动的样本。
    """
    base_path = os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets")
    file_pattern = "Temporal_components_prey-loom-omr-set00{}_{}.npy"
    
    H_set1 = np.load(os.path.join(base_path, file_pattern.format(1, component_number)))
    H_set2 = np.load(os.path.join(base_path, file_pattern.format(2, component_number)))
    H_set3 = np.load(os.path.join(base_path, file_pattern.format(3, component_number)))
    H_set4 = np.load(os.path.join(base_path, file_pattern.format(4, component_number)))
    
    H_concatenated = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)
    
    T1 = H_set1.shape[1]
    T2 = H_set2.shape[1]
    T3 = H_set3.shape[1]
    
    offsets = {
        'prey-loom-omr-set001': 0,
        'prey-loom-omr-set002': T1,
        'prey-loom-omr-set003': T1 + T2,
        'prey-loom-omr-set004': T1 + T2 + T3
    }
    
    event_mask = np.zeros(H_concatenated.shape[1], dtype=bool)
    
    if mode == 'hunting':
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            convergence_periods = json.load(file)
        sets = list(convergence_periods[fish_name].keys())
        for set_name in sets:
            current_convergence_periods = convergence_periods[fish_name][set_name]
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for c in current_convergence_periods:
                start = c[0] / 300 - window_size
                end = c[0] / 300 + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True
                
    elif mode == 'passivity':
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = list(passivity_json[fish_name].keys())
        for set_name in sets:
            current_passivity_periods = passivity_json[fish_name][set_name]
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for bout in current_passivity_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True
                
    elif mode == 'loom':
        periods, stimuli = get_stimulus(fish_name)
        looming_periods = []
        for i in range(len(stimuli)):
            if stimuli[i] == 'loom':
                looming_periods.append(periods[i])
        
        # Looming 同样需要遍历所有 sets 来匹配时间
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = list(passivity_json[fish_name].keys())
        
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for bout in looming_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True
                
    else:
        # Default to hunting behavior if mode unknown
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            convergence_periods = json.load(file)
        sets = list(convergence_periods[fish_name].keys())
        for set_name in sets:
            current_convergence_periods = convergence_periods[fish_name][set_name]
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for c in current_convergence_periods:
                start = c[0] / 300 - window_size
                end = c[0] / 300 + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True

    bg_mask = ~event_mask
    X_avg_event = H_concatenated[:, event_mask] if np.any(event_mask) else H_concatenated[:, :0]
    low_var_flags = compute_low_variability_flags(X_avg_event, H_concatenated, bg_mask, threshold=threshold)
    
    has_sig_flags = ((significance == 1) & event_mask).any(axis=1)
    flagged_indices = np.where(low_var_flags & has_sig_flags)[0]
    
    return flagged_indices, low_var_flags, event_mask


def compute_eventwise_low_variability_scores(fish_name, component_number, time_stamps, window_size=10, threshold=0.25, mode='hunting'):
    """
    逐事件窗口计算低波动 ratio。
    """
    base_path = os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets")
    file_pattern = "Temporal_components_prey-loom-omr-set00{}_{}.npy"
    
    H_set1 = np.load(os.path.join(base_path, file_pattern.format(1, component_number)))
    H_set2 = np.load(os.path.join(base_path, file_pattern.format(2, component_number)))
    H_set3 = np.load(os.path.join(base_path, file_pattern.format(3, component_number)))
    H_set4 = np.load(os.path.join(base_path, file_pattern.format(4, component_number)))
    
    H_concatenated = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)
    
    T1 = H_set1.shape[1]
    T2 = H_set2.shape[1]
    T3 = H_set3.shape[1]
    
    offsets = {
        'prey-loom-omr-set001': 0,
        'prey-loom-omr-set002': T1,
        'prey-loom-omr-set003': T1 + T2,
        'prey-loom-omr-set004': T1 + T2 + T3
    }
    
    if mode == 'hunting':
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        events_by_set = {s: [[c[0] / 300 - window_size, c[0] / 300 + window_size] for c in periods_json[fish_name][s]] for s in sets}
    elif mode == 'passivity':
        with open(PASSIVITY_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        events_by_set = {s: [[b[0] - window_size, b[0] + window_size] for b in periods_json[fish_name][s]] for s in sets}
    else:
        periods, stimuli = get_stimulus(fish_name)
        looming = []
        for i in range(len(stimuli)):
            if stimuli[i] == 'loom':
                looming.append([periods[i][0] - window_size, periods[i][0] + window_size])
        
        with open(PASSIVITY_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        events_by_set = {s: looming[:] for s in sets}
        
    event_mask_all = np.zeros(H_concatenated.shape[1], dtype=bool)
    for set_name, win_list in events_by_set.items():
        off = offsets.get(set_name, None)
        if off is None:
            continue
        for start, end in win_list:
            s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
            s_idx = max(0, s_idx)
            e_idx = min(time_stamps.shape[0] - 1, e_idx)
            event_mask_all[off + s_idx: off + e_idx + 1] = True
            
    bg_mask_all = ~event_mask_all
    n = H_concatenated.shape[0]
    
    if np.count_nonzero(bg_mask_all) > 1:
        std_bg_all = H_concatenated[:, bg_mask_all].std(axis=1, ddof=1)
    else:
        std_bg_all = np.zeros(n)
        
    results = []
    for set_name, win_list in events_by_set.items():
        off = offsets.get(set_name, None)
        if off is None:
            continue
        for idx, (start, end) in enumerate(win_list):
            s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
            s_idx = max(0, s_idx)
            e_idx = min(time_stamps.shape[0] - 1, e_idx)
            
            if e_idx < s_idx:
                ratios = np.full(n, np.inf)
                flags = np.zeros(n, dtype=bool)
                results.append({'set': set_name, 'event_index': idx, 'ratios': ratios, 'flags': flags, 'window': (start, end)})
                continue
                
            X_win = H_concatenated[:, off + s_idx: off + e_idx + 1]
            if X_win.shape[1] == 0:
                amp = np.zeros(n)
            else:
                amp = np.mean(np.abs(X_win), axis=1)
                
            denom = np.maximum(amp, 1e-12)
            ratios = std_bg_all / denom
            flags = ratios <= threshold
            results.append({'set': set_name, 'event_index': idx, 'ratios': ratios, 'flags': flags, 'window': (start, end)})
            
    return results


def save_eventwise_scores_json(eventwise, fish_name, component_number, mode, base_dir):
    """
    将逐事件的 ratio/flags 保存为 JSON。
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f'{fish_name}_{component_number}_{mode}_eventwise.json')
    payload = {
        'fish_name': fish_name,
        'component_number': component_number,
        'mode': mode,
        'events': []
    }
    for item in eventwise:
        payload['events'].append({
            'set': item['set'],
            'event_index': item['event_index'],
            'window': [item['window'][0], item['window'][1]],
            'ratios': [float(x) for x in item['ratios']],
            'flags': [bool(x) for x in item['flags']]
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_eventwise_scores_csv(eventwise, fish_name, component_number, mode, base_dir):
    """
    将逐事件的 ratio/flags 按行展开保存为 CSV。
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f'{fish_name}_{component_number}_{mode}_eventwise.csv')
    rows = []
    for item in eventwise:
        set_name = item['set']
        event_index = item['event_index']
        w0, w1 = item['window']
        ratios = item['ratios']
        flags = item['flags']
        for sample_idx in range(len(ratios)):
            rows.append({
                'fish_name': fish_name,
                'component_number': component_number,
                'mode': mode,
                'set': set_name,
                'event_index': event_index,
                'sample_index': sample_idx,
                'ratio': float(ratios[sample_idx]),
                'low_variability': int(flags[sample_idx]),
                'window_start': w0,
                'window_end': w1
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')


def compute_mode_union_low_variability_scores(fish_name, component_number, time_stamps, window_size=10, threshold=0.25, mode='hunting'):
    """
    计算“事件并集版”的低波动 ratio/flag。
    """
    base_path = os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets")
    file_pattern = "Temporal_components_prey-loom-omr-set00{}_{}.npy"
    
    H_set1 = np.load(os.path.join(base_path, file_pattern.format(1, component_number)))
    H_set2 = np.load(os.path.join(base_path, file_pattern.format(2, component_number)))
    H_set3 = np.load(os.path.join(base_path, file_pattern.format(3, component_number)))
    H_set4 = np.load(os.path.join(base_path, file_pattern.format(4, component_number)))
    
    H_concatenated = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)
    
    T1 = H_set1.shape[1]
    T2 = H_set2.shape[1]
    T3 = H_set3.shape[1]
    
    offsets = {
        'prey-loom-omr-set001': 0,
        'prey-loom-omr-set002': T1,
        'prey-loom-omr-set003': T1 + T2,
        'prey-loom-omr-set004': T1 + T2 + T3
    }
    
    event_mask = np.zeros(H_concatenated.shape[1], dtype=bool)
    
    if mode == 'hunting':
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for c in periods_json[fish_name][set_name]:
                start = c[0] / 300 - window_size
                end = c[0] / 300 + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True
    elif mode == 'passivity':
        with open(PASSIVITY_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for b in periods_json[fish_name][set_name]:
                start = b[0] - window_size
                end = b[0] + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True
    else:
        periods, stimuli = get_stimulus(fish_name)
        looming_periods = []
        for i in range(len(stimuli)):
            if stimuli[i] == 'loom':
                looming_periods.append(periods[i])
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = list(passivity_json[fish_name].keys())
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for bout in looming_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask[off + s_idx: off + e_idx + 1] = True
                
    bg_mask = ~event_mask
    X_avg_event = H_concatenated[:, event_mask] if np.any(event_mask) else H_concatenated[:, :0]
    n = H_concatenated.shape[0]
    
    if np.count_nonzero(bg_mask) > 1:
        std_bg = H_concatenated[:, bg_mask].std(axis=1, ddof=1)
    else:
        std_bg = np.zeros(n)
        
    if X_avg_event.shape[1] == 0:
        amp_avg = np.zeros(n)
    else:
        amp_avg = np.mean(np.abs(X_avg_event), axis=1)
        
    denom = np.maximum(amp_avg, 1e-12)
    ratios = std_bg / denom
    flags = ratios <= threshold
    return ratios, flags


def save_mode_union_scores_json(ratios, flags, fish_name, component_number, mode, base_dir):
    """
    保存 Union 版的每样本 ratio/flag 为 JSON。
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f'{fish_name}_{component_number}_{mode}_union.json')
    payload = {
        'fish_name': fish_name,
        'component_number': component_number,
        'mode': mode,
        'ratios': [float(x) for x in ratios],
        'flags': [bool(x) for x in flags]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_mode_union_scores_csv(ratios, flags, fish_name, component_number, mode, base_dir):
    """
    保存 Union 版的每样本 ratio/flag 为 CSV。
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f'{fish_name}_{component_number}_{mode}_union.csv')
    rows = []
    for i in range(len(ratios)):
        rows.append({
            'fish_name': fish_name,
            'component_number': component_number,
            'mode': mode,
            'sample_index': i,
            'ratio': float(ratios[i]),
            'low_variability': int(flags[i])
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')


def compute_mode_average_low_variability_scores(fish_name, component_number, time_stamps, window_size=10, threshold=0.25, mode='hunting'):
    """
    计算“事件平均版”的低波动 ratio/flag。
    """
    base_path = os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets")
    file_pattern = "Temporal_components_prey-loom-omr-set00{}_{}.npy"
    
    H_set1 = np.load(os.path.join(base_path, file_pattern.format(1, component_number)))
    H_set2 = np.load(os.path.join(base_path, file_pattern.format(2, component_number)))
    H_set3 = np.load(os.path.join(base_path, file_pattern.format(3, component_number)))
    H_set4 = np.load(os.path.join(base_path, file_pattern.format(4, component_number)))
    
    H_concatenated = np.concatenate((H_set1, H_set2, H_set3, H_set4), axis=1)
    
    traces = []
    if mode == 'hunting':
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        for set_name in sets:
            H = np.load(os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets", 
                                   f"Temporal_components_{set_name}_{component_number}.npy"))
            for c in periods_json[fish_name][set_name]:
                start = c[0] / 300 - window_size
                end = c[0] / 300 + window_size
                slices = slice_matrix_by_time(H, time_stamps, [start, end])
                traces.append(slices)
    elif mode == 'passivity':
        with open(PASSIVITY_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        for set_name in sets:
            H = np.load(os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets", 
                                   f"Temporal_components_{set_name}_{component_number}.npy"))
            for b in periods_json[fish_name][set_name]:
                start = b[0] - window_size
                end = b[0] + window_size
                slices = slice_matrix_by_time(H, time_stamps, [start, end])
                traces.append(slices)
    else:
        periods, stimuli = get_stimulus(fish_name)
        looming_periods = []
        for i in range(len(stimuli)):
            if stimuli[i] == 'loom':
                looming_periods.append(periods[i])
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = list(passivity_json[fish_name].keys())
        for set_name in sets:
            H = np.load(os.path.join(REGISTERED_FISH_DIR, fish_name, "Temporal_components_full_sets", 
                                   f"Temporal_components_{set_name}_{component_number}.npy"))
            for bout in looming_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                slices = slice_matrix_by_time(H, time_stamps, [start, end])
                traces.append(slices)
                
    if len(traces) == 0:
        averaged = np.zeros((H_concatenated.shape[0], 0))
    else:
        try:
            averaged = np.mean(traces, axis=0)
        except Exception:
            min_len = min([tr.shape[1] for tr in traces])
            trimmed = [tr[:, :min_len] for tr in traces]
            averaged = np.mean(trimmed, axis=0)
            
    # Calculate background mask (same logic as union)
    T1 = H_set1.shape[1]
    T2 = H_set2.shape[1]
    T3 = H_set3.shape[1]
    offsets = {
        'prey-loom-omr-set001': 0,
        'prey-loom-omr-set002': T1,
        'prey-loom-omr-set003': T1 + T2,
        'prey-loom-omr-set004': T1 + T2 + T3
    }
    
    event_mask_all = np.zeros(H_concatenated.shape[1], dtype=bool)
    
    if mode == 'hunting':
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for c in periods_json[fish_name][set_name]:
                start = c[0] / 300 - window_size
                end = c[0] / 300 + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask_all[off + s_idx: off + e_idx + 1] = True
    elif mode == 'passivity':
        with open(PASSIVITY_PATH, 'r') as file:
            periods_json = json.load(file)
        sets = list(periods_json[fish_name].keys())
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for b in periods_json[fish_name][set_name]:
                start = b[0] - window_size
                end = b[0] + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask_all[off + s_idx: off + e_idx + 1] = True
    else:
        periods, stimuli = get_stimulus(fish_name)
        looming_periods = []
        for i in range(len(stimuli)):
            if stimuli[i] == 'loom':
                looming_periods.append(periods[i])
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = list(passivity_json[fish_name].keys())
        for set_name in sets:
            off = offsets.get(set_name, None)
            if off is None:
                continue
            for bout in looming_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                s_idx, e_idx = find_nearest_indices(time_stamps, start, end)
                s_idx = max(0, s_idx)
                e_idx = min(time_stamps.shape[0] - 1, e_idx)
                event_mask_all[off + s_idx: off + e_idx + 1] = True
                
    bg_mask_all = ~event_mask_all
    n = H_concatenated.shape[0]
    
    if np.count_nonzero(bg_mask_all) > 1:
        std_bg_all = H_concatenated[:, bg_mask_all].std(axis=1, ddof=1)
    else:
        std_bg_all = np.zeros(n)
        
    if averaged.shape[1] == 0:
        amp = np.zeros(n)
    else:
        amp = np.mean(np.abs(averaged), axis=1)
        
    denom = np.maximum(amp, 1e-12)
    ratios = std_bg_all / denom
    flags = ratios <= threshold
    return ratios, flags


def save_mode_average_scores_json(ratios, flags, fish_name, component_number, mode, base_dir):
    """
    保存 Average 版的每样本 ratio/flag 为 JSON。
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f'{fish_name}_{component_number}_{mode}_average.json')
    payload = {
        'fish_name': fish_name,
        'component_number': component_number,
        'mode': mode,
        'ratios': [float(x) for x in ratios],
        'flags': [bool(x) for x in flags]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_mode_average_scores_csv(ratios, flags, fish_name, component_number, mode, base_dir):
    """
    保存 Average 版的每样本 ratio/flag 为 CSV。
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f'{fish_name}_{component_number}_{mode}_average.csv')
    rows = []
    for i in range(len(ratios)):
        rows.append({
            'fish_name': fish_name,
            'component_number': component_number,
            'mode': mode,
            'sample_index': i,
            'ratio': float(ratios[i]),
            'low_variability': int(flags[i])
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')


def save_selected_samples_summary(fish_name, component_number, mode, flagged_indices, base_dir):
    """
    保存最终选中的样本汇总表。
    """
    save_dir = os.path.join(base_dir, mode, fish_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{fish_name}_{component_number}_{mode}_selected_samples.csv')
    rows = []
    for idx in (flagged_indices.tolist() if hasattr(flagged_indices, "tolist") else list(flagged_indices)):
        rows.append({
            'fish_name': fish_name,
            'component_number': component_number,
            'mode': mode,
            'sample_index': int(idx),
            'significant_high': 1,
            'low_variability': 1
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')


def process_mode_detection(fish_name, component_number, window_size, mode, threshold=0.4):
    """
    通用的模式检测处理函数，合并 hunting/passivity/loom 的重复逻辑。
    """
    print(f"Start detecting {mode} for {fish_name}...")
    
    # Load timestamps
    time_stamps = np.load(os.path.join(REGISTERED_FISH_DIR, fish_name, 't.npy'))
    
    # Run significance analysis
    significance, p_values = example_usage(fish_name, component_number)
    
    # Identify flagged samples
    flagged_indices, low_var_flags, event_mask = find_low_variability_high_significance_samples(
        fish_name, component_number, significance, time_stamps, 
        window_size=window_size, threshold=threshold, mode=mode
    )
    
    print(f"在事件窗口出现显著高且非事件区方差低的样本索引: {flagged_indices.tolist()}")
    
    # Save selected samples
    summary_base = os.path.join(OUTPUT_BASE_DIR, 'summaries')
    save_selected_samples_summary(fish_name, component_number, mode, flagged_indices, summary_base)
    
    # Compute and save Union scores
    ratios_union, flags_union = compute_mode_union_low_variability_scores(
        fish_name, component_number, time_stamps, window_size=window_size, threshold=threshold, mode=mode
    )
    union_base = os.path.join(OUTPUT_BASE_DIR, 'mode_union_low_variability')
    save_mode_union_scores_json(ratios_union, flags_union, fish_name, component_number, mode, union_base)
    save_mode_union_scores_csv(ratios_union, flags_union, fish_name, component_number, mode, union_base)
    
    # Compute and save Average scores
    ratios_avg, flags_avg = compute_mode_average_low_variability_scores(
        fish_name, component_number, time_stamps, window_size=window_size, threshold=threshold, mode=mode
    )
    avg_base = os.path.join(OUTPUT_BASE_DIR, 'mode_average_low_variability')
    save_mode_average_scores_json(ratios_avg, flags_avg, fish_name, component_number, mode, avg_base)
    save_mode_average_scores_csv(ratios_avg, flags_avg, fish_name, component_number, mode, avg_base)
    
    # Compute and save Eventwise scores
    eventwise = compute_eventwise_low_variability_scores(
        fish_name, component_number, time_stamps, window_size=window_size, threshold=threshold, mode=mode
    )
    eventwise_base = os.path.join(OUTPUT_BASE_DIR, 'eventwise_low_variability')
    save_eventwise_scores_json(eventwise, fish_name, component_number, mode, eventwise_base)
    save_eventwise_scores_csv(eventwise, fish_name, component_number, mode, eventwise_base)
    
    # Visualize heatmaps
    significance_sets = np.array_split(significance, 4, axis=1)
    significance_by_set = {
        'prey-loom-omr-set001': significance_sets[0],
        'prey-loom-omr-set002': significance_sets[1],
        'prey-loom-omr-set003': significance_sets[2],
        'prey-loom-omr-set004': significance_sets[3]
    }
    
    event_count = 0
    
    if mode == 'hunting':
        with open(CONVERGENCE_PERIODS_PATH, 'r') as file:
            convergence_periods = json.load(file)
        sets = convergence_periods[fish_name].keys()
        for set_name in sets:
            current_convergence_periods = convergence_periods[fish_name][set_name]
            convergence_periods_seconds = [[c[0] / 300, c[1] / 300] for c in current_convergence_periods]
            
            for bout in convergence_periods_seconds:
                start = bout[0] - window_size
                end = bout[0] + window_size
                temp_indices = find_nearest_indices(time_stamps, start, end)
                visualize_set_significance(significance_by_set[set_name], temp_indices, fish_name, set_name, 
                                           component_number, start, end, mode, event_count, highlight_rows=flagged_indices)
                event_count += 1
                
    elif mode == 'passivity':
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = passivity_json[fish_name].keys()
        for set_name in sets:
            current_passivity_periods = passivity_json[fish_name][set_name]
            for bout in current_passivity_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                temp_indices = find_nearest_indices(time_stamps, start, end)
                visualize_set_significance(significance_by_set[set_name], temp_indices, fish_name, set_name, 
                                           component_number, start, end, mode, event_count, highlight_rows=flagged_indices)
                event_count += 1
                
    elif mode == 'loom':
        with open(PASSIVITY_PATH, 'r') as file:
            passivity_json = json.load(file)
        sets = passivity_json[fish_name].keys()
        
        periods, stimulus = get_stimulus(fish_name)
        looming_periods = [periods[l] for l in range(len(stimulus)) if stimulus[l] == 'loom']
        
        for set_name in sets:
            for bout in looming_periods:
                start = bout[0] - window_size
                end = bout[0] + window_size
                temp_indices = find_nearest_indices(time_stamps, start, end)
                visualize_set_significance(significance_by_set[set_name], temp_indices, fish_name, set_name, 
                                           component_number, start, end, mode, event_count, highlight_rows=flagged_indices)
                event_count += 1


def detect_hunting(fish_name, component_number, window_size=10):
    """
    狩猎模式入口。
    """
    process_mode_detection(fish_name, component_number, window_size, 'hunting', threshold=0.4)


def detect_passivity(fish_name, component_number, window_size=10):
    """
    被动模式入口。
    """
    process_mode_detection(fish_name, component_number, window_size, 'passivity', threshold=0.55)


def detect_loom(fish_name, component_number, window_size=10):
    """
    惊吓（loom）模式入口。
    """
    process_mode_detection(fish_name, component_number, window_size, 'loom', threshold=0.4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("--fish", required=True, help="Fish name, e.g. 20250807-F1")
    parser.add_argument("--component", required=True, type=int, help="Component number, e.g. 90")
    parser.add_argument("--mode", required=True, choices=["hunting", "passivity", "loom"])
    parser.add_argument("--window-size", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "hunting":
        detect_hunting(args.fish, args.component, window_size=int(args.window_size))
    elif args.mode == "passivity":
        detect_passivity(args.fish, args.component, window_size=int(args.window_size))
    elif args.mode == "loom":
        detect_loom(args.fish, args.component, window_size=int(args.window_size))
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")
