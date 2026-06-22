"""
Behavior Classification Module

This module provides functionality for classifying zebrafish behaviors using
tail angle time-series data. It includes data loading, feature extraction using
ROCKET (Random Convolutional Kernel Transform), and classification using
LightGBM with a Label Powerset approach for multi-label classification.

The module supports:
1. Loading and preprocessing behavioral data (tail angles).
2. Training a classifier with hyperparameter optimization (Optuna).
3. Cross-validation and evaluation of the model.
4. Reloading trained models for prediction and fine-tuning.

Dependencies:
    - numpy, pandas, matplotlib, seaborn
    - scikit-learn, sktime, lightgbm, joblib
    - pycatch22, wordcloud, optuna, torch (for custom dataset if needed)
"""

import os
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import torch

from datetime import datetime
from collections import Counter
from pycatch22 import catch22_all
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    classification_report, hamming_loss, multilabel_confusion_matrix
)
from skmultilearn.problem_transform import LabelPowerset
from sktime.transformations.panel.rocket import Rocket
from torch.utils.data import Dataset

# =============================================================================
# Configuration & Constants
# =============================================================================

# Base directories
ASTROCYTE_DATA_ROOT = os.environ.get("ASTROCYTE_DATA_ROOT", r"D:/2p astrocyte")
BEHAVIOUR_DIR = os.path.join(ASTROCYTE_DATA_ROOT, "behaviour")
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
RESULTS_DIR = os.path.join(BEHAVIOUR_DIR, "classification models and results")
OPTO_DATA_DIR = os.path.join(BEHAVIOUR_DIR, "opto data", "bouts_sampled_for_labeling")

# Input data paths
BOUTS_INFO_PATH = os.path.join(BEHAVIOUR_DIR, "bouts_info_appended.json")

# Output paths for models and logs
MODELS_BASE_DIR = os.path.join(RESULTS_DIR, 'models')
TRAINING_LOG_PATH = os.path.join(RESULTS_DIR, 'training_log_v3.txt')
BEST_PARAMS_PATH_DEFAULT = os.path.join(RESULTS_DIR, 'best_params_latest_combined.json')
ROCKET_MODEL_PATH_DEFAULT = os.path.join(PROJECT_ROOT, 'rocket_model.joblib')
CLASSIFIER_MODEL_PATH_DEFAULT = os.path.join(PROJECT_ROOT, 'rocket_multilabel_classifier.pkl')

# Plotting output paths
BASE_LABEL_PLOT_PATH = os.path.join(RESULTS_DIR, 'base_label_bar_plot_v2.png')

# Classification Constants
ORIGINAL_BASE_LABELS = [
    'J-turn right', 'escape', 'struggle', 'slow swim', 'turn left', 'turn right'
]
TARGET_BASE_LABELS = ['J-turn right', 'escape', 'struggle', 'spontaneous swim']
OMR_LABELS = ['slow swim', 'turn left', 'turn right']


# =============================================================================
# Helper Functions
# =============================================================================

def extract_features(series):
    """
    Extract features from a time series using catch22.

    Args:
        series (np.ndarray): Input time series data of shape (n_samples, n_channels).

    Returns:
        list: Extracted features as a flat list.
    """
    features = []
    for channel in range(series.shape[1]):
        channel_data = series[:, channel]
        result = catch22_all(channel_data)
        features.extend(result['values'])
    return features


def get_tail_angles(df_tail, heading):
    """
    Calculate tail angles relative to the fish's heading.

    Args:
        df_tail (pd.DataFrame): DataFrame containing tail coordinates.
                                Columns are assumed to be interleaved x, y.
        heading (np.ndarray): Array of heading angles in degrees.

    Returns:
        np.ndarray: Calculated tail angles.
    """
    # Convert x, y coordinates to complex numbers for easier manipulation
    xy = df_tail.values[:, ::2] + df_tail.values[:, 1::2] * 1j
    
    # Create midline vector based on heading
    midline = -np.exp(1j * np.deg2rad(np.asarray(heading)))

    # Calculate angles relative to midline
    # Note: np.diff computes discrete difference along axis 1 (segments)
    return -np.angle(np.diff(xy, axis=1) / midline[:, None])


def encode_label(label, base_labels):
    """
    Encode a text label into a binary vector (multi-hot encoding).

    Args:
        label (str): The label string (e.g., "escape + struggle").
        base_labels (list): List of all possible base labels.

    Returns:
        np.ndarray: Binary vector indicating presence of each base label.
    """
    vector = np.zeros(len(base_labels), dtype=int)
    parts = label.split(' + ')
    for part in parts:
        if part in base_labels:
            index = base_labels.index(part)
            vector[index] = 1
    return vector


def decode_label(vector, base_labels):
    """
    Decode a binary vector back into a text label.

    Args:
        vector (np.ndarray): Binary vector.
        base_labels (list): List of all possible base labels.

    Returns:
        str: Reconstructed label string joined by ' + ', or 'No Label'.
    """
    labels = [base_labels[i] for i, val in enumerate(vector) if val == 1]
    return ' + '.join(labels) if labels else 'No Label'


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for variable-length time series.
    """
    def __init__(self, series, labels):
        self.lengths = [len(ts) for ts in series]
        # Pad sequences to the same length
        self.X = torch.nn.utils.rnn.pad_sequence(
            [torch.FloatTensor(ts) for ts in series],
            batch_first=True
        )
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]


# =============================================================================
# Data Loading & Processing
# =============================================================================

def load_and_process_data():
    """
    Load behavioral data from JSON and HDF5 files, process tail angles,
    and generate labels.

    Returns:
        tuple: (time_series, new_labels, metadata, base_labels)
            - time_series (list of np.ndarray): List of tail angle arrays.
            - new_labels (np.ndarray): Encoded multi-label vectors.
            - metadata (list of dict): Metadata for each sample.
            - base_labels (list): List of base label names used for encoding.
    """
    print(f"Loading bouts info from {BOUTS_INFO_PATH}...")
    if not os.path.exists(BOUTS_INFO_PATH):
        raise FileNotFoundError(f"Bouts info file not found: {BOUTS_INFO_PATH}")

    with open(BOUTS_INFO_PATH, 'r') as file:
        bouts_info = json.load(file)

    time_series = []
    original_labels = []
    metadata = []
    fish_names = list(bouts_info.keys())

    # Skip first 3 fish as per original logic
    for fish_name in fish_names[3:]:
        set_ids = list(bouts_info[fish_name].keys())

        for set_id in set_ids:
            try:
                bouts = bouts_info[fish_name][set_id]['bout']
                bout_labels = bouts_info[fish_name][set_id]['behaviour']
                
                if len(bouts) != len(bout_labels):
                    print(f"Warning: Mismatch in bouts ({len(bouts)}) and labels ({len(bout_labels)}) "
                          f"for {fish_name}/{set_id}. Skipping.")
                    continue
                
                # Construct path to H5 file
                # Assuming standard naming convention based on fish_name and set_id
                h5_filename = f"{fish_name.replace('-', '_')}_Trial1.h5"
                h5_path = os.path.join(BASE_DATA_DIR, fish_name, 'TOPCAMERA', set_id, h5_filename)
                
                if not os.path.exists(h5_path):
                    print(f"Warning: H5 file not found: {h5_path}. Skipping.")
                    continue

                # Load tail and eye data
                df_tail = pd.read_hdf(h5_path, "tail")
                heading = pd.read_hdf(h5_path, "eye")["heading"].values
                
                tail_angles = get_tail_angles(df_tail, heading)
                # Drop first 5 points (likely noise or setup artifacts)
                tail_angles = tail_angles[:, 5:]

                for i, bout in enumerate(bouts):
                    # Extract bout segment
                    segment = tail_angles[bout[0]:bout[1], :]
                    if segment.size == 0:
                        continue
                        
                    time_series.append(segment)
                    original_labels.append(bout_labels[i])
                    metadata.append({
                        'fish_name': fish_name,
                        'set_id': set_id,
                        'bout': bout,
                    })
            except Exception as e:
                print(f"Error processing {fish_name}/{set_id}: {e}")
                continue

    # Process Labels
    # Determine base labels counts (logic preserved from original)
    original_base_label_counts = {label: 0 for label in ORIGINAL_BASE_LABELS}
    for label in original_labels:
        for base_label in ORIGINAL_BASE_LABELS:
            if base_label in label:
                original_base_label_counts[base_label] += 1

    # Map specific swim types to 'spontaneous swim'
    base_labels = TARGET_BASE_LABELS
    
    # Remap labels
    processed_labels = []
    for label in original_labels:
        parts = label.split(' + ')
        parts = ['spontaneous swim' if part in OMR_LABELS else part for part in parts]
        # Deduplicate and sort to ensure consistency if needed, though ' + ' join order matters if not sorted
        # Here we just join them back as per original logic
        new_label = ' + '.join(parts)
        processed_labels.append(new_label)

    # Encode labels
    new_labels_encoded = np.array([encode_label(label, base_labels) for label in processed_labels])
    
    print(f"Loaded {len(time_series)} samples.")
    return time_series, new_labels_encoded, metadata, base_labels


# =============================================================================
# Modeling & Classification
# =============================================================================

def classifier_combined_labels(time_series, new_labels, base_labels,
                               use_saved_best_params=False, best_params_path=None, 
                               n_trials=1, reload_models=False, 
                               rocket_model_path=None, classifier_model_path=None):
    """
    Train and evaluate a classifier using ROCKET features and LightGBM.

    Args:
        time_series (list): List of time series arrays.
        new_labels (np.ndarray): Encoded labels.
        base_labels (list): List of label names.
        use_saved_best_params (bool): Whether to load best params from file.
        best_params_path (str): Path to best params JSON.
        n_trials (int): Number of Optuna trials for hyperparameter tuning.
        reload_models (bool): Whether to reload existing ROCKET/Classifier models.
        rocket_model_path (str): Path to saved ROCKET model.
        classifier_model_path (str): Path to saved Classifier model.
    """
    # Setup paths
    training_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join(MODELS_BASE_DIR, training_time)
    os.makedirs(models_dir, exist_ok=True)
    
    bp_path = best_params_path if best_params_path else BEST_PARAMS_PATH_DEFAULT
    rm_path = rocket_model_path if rocket_model_path else ROCKET_MODEL_PATH_DEFAULT
    cm_path = classifier_model_path if classifier_model_path else CLASSIFIER_MODEL_PATH_DEFAULT

    # Pad time series for ROCKET
    max_len = max(len(ts) for ts in time_series)
    X_padded = [np.pad(ts, ((0, max_len - len(ts)), (0, 0))) for ts in time_series]
    X_padded = np.array(X_padded)
    
    # ROCKET expects (n_instances, n_dims, n_timesteps)
    # Our data is (n_instances, n_timesteps, n_dims), so transpose
    X_padded_transposed = X_padded.transpose(0, 2, 1)

    # --- Hyperparameter Optimization ---
    if use_saved_best_params and os.path.exists(bp_path):
        print(f"Reloading best params from {bp_path}...")
        with open(bp_path, 'r') as f:
            best_params = json.load(f)
    else:
        def objective(trial):
            num_kernels = trial.suggest_int('num_kernels', 1000, 10000, step=100)
            rocket = Rocket(num_kernels=num_kernels, random_state=42)
            X_rocket = rocket.fit_transform(X_padded_transposed)
            
            lgbm_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_hamming_losses = []
            X_rocket_values = X_rocket.values
            
            for _, (train_idx, test_idx) in enumerate(kf.split(X_rocket_values)):
                X_tr, X_te = X_rocket_values[train_idx], X_rocket_values[test_idx]
                y_tr, y_te = new_labels[train_idx], new_labels[test_idx]
                
                clf = LabelPowerset(lgb.LGBMClassifier(**lgbm_params))
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                y_pred_dense = y_pred.toarray() if hasattr(y_pred, 'toarray') else y_pred
                fold_hamming_losses.append(hamming_loss(y_te, y_pred_dense))
                
            return np.mean(fold_hamming_losses)

        print(f"Running {n_trials} trials for hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params
        
        # Save params
        with open(bp_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best params saved to {bp_path}")

    best_num_kernels = best_params.pop('num_kernels')
    best_lgbm_params = best_params

    # --- ROCKET Transformation ---
    if reload_models and os.path.exists(rm_path):
        print(f"Reloading ROCKET model from {rm_path}...")
        rocket = joblib.load(rm_path)
        X_rocket = rocket.transform(X_padded_transposed)
    else:
        print(f"Applying ROCKET transform with {best_num_kernels} kernels...")
        rocket = Rocket(num_kernels=best_num_kernels, random_state=42)
        X_rocket = rocket.fit_transform(X_padded_transposed)
        joblib.dump(rocket, rm_path)
        joblib.dump(rocket, os.path.join(models_dir, f'rocket_model_{training_time}.pkl'))
        print(f"ROCKET model saved to {rm_path}")

    # --- Cross-Validation Evaluation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_reports = []
    fold_hamming_losses = []
    X_rocket_values = X_rocket.values

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_rocket_values)):
        print(f"--- Fold {fold + 1}/5 ---")
        X_train, X_test = X_rocket_values[train_idx], X_rocket_values[test_idx]
        y_train, y_test = new_labels[train_idx], new_labels[test_idx]

        # Use the global classifier path for the main model, but don't overwrite it in CV loops usually
        # The original code logic was slightly unusual: it checked for reload inside the loop.
        # Here we will train fresh for CV to get accurate metrics, 
        # unless specifically asked to reload a pre-trained model for *all* folds (which implies data leakage or static model).
        # Assuming we want to evaluate the *method*, we should train new models per fold.
        
        classifier = LabelPowerset(
            lgb.LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, class_weight='balanced')
        )
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        y_pred_dense = y_pred.toarray() if hasattr(y_pred, 'toarray') else y_pred
        
        # Save fold-specific confusion matrices
        plot_confusion_matrices(y_test, y_pred_dense, base_labels, 
                                f'Fold {fold + 1} Test Set', 
                                os.path.join(models_dir, f'fold_{fold+1}_cm.png'))

        report = classification_report(y_test, y_pred_dense, target_names=base_labels, output_dict=True)
        fold_reports.append(report)
        fold_hamming_losses.append(hamming_loss(y_test, y_pred_dense))

    print("\n--- Cross-Validation Summary ---")
    print(f"Average Hamming Loss: {np.mean(fold_hamming_losses):.4f}")
    
    # Train Final Model on All Data
    print("Training final model on all data...")
    final_classifier = LabelPowerset(
        lgb.LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, class_weight='balanced')
    )
    final_classifier.fit(X_rocket_values, new_labels)
    joblib.dump(final_classifier, cm_path)
    joblib.dump(final_classifier, os.path.join(models_dir, f'classifier_final_{training_time}.pkl'))
    print(f"Final classifier saved to {cm_path}")


def plot_confusion_matrices(y_true, y_pred, labels, title, save_path):
    """
    Helper to plot multilabel confusion matrices.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    num_labels = len(labels)
    cols = 2
    rows = (num_labels + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle(f'{title} Confusion Matrices', fontsize=16)
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, (matrix, label) in enumerate(zip(mcm, labels)):
        if i >= len(axes): break
        matrix_percent = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        annot = np.asarray([[f'{p:.2%}\n({v})' for p, v in zip(row_p, row_v)] 
                            for row_p, row_v in zip(matrix_percent, matrix)])
        sns.heatmap(matrix_percent, annot=annot, fmt='', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred Neg', 'Pred Pos'],
                    yticklabels=['Actual Neg', 'Actual Pos'])
        axes[i].set_title(f'{label}', color=colors[i % len(colors)])
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def reload_predict_and_finetune(new_series, new_labels=None, fine_tune=False, 
                                best_params_path=None, rocket_model_path=None, 
                                classifier_model_path=None, base_labels=None,
                                time_series_train=None, labels_train=None):
    """
    Reload existing models to predict on new data or fine-tune the model.
    """
    training_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join(MODELS_BASE_DIR, training_time)
    
    bp_path = best_params_path if best_params_path else BEST_PARAMS_PATH_DEFAULT
    rm_path = rocket_model_path if rocket_model_path else ROCKET_MODEL_PATH_DEFAULT
    cm_path = classifier_model_path if classifier_model_path else CLASSIFIER_MODEL_PATH_DEFAULT
    
    if base_labels is None:
        base_labels = TARGET_BASE_LABELS

    # Load params
    best_lgbm_params = {}
    if os.path.exists(bp_path):
        with open(bp_path, 'r') as f:
            best_params = json.load(f)
        best_lgbm_params = {k: v for k, v in best_params.items() if k != 'num_kernels'}

    # Load models
    if not os.path.exists(rm_path) or not os.path.exists(cm_path):
        print("Error: Models not found for reloading.")
        return None

    rocket = joblib.load(rm_path)
    classifier = joblib.load(cm_path)
    
    # Transform new data
    max_len_new = max(len(ts) for ts in new_series)
    X_new_padded = [np.pad(ts, ((0, max_len_new - len(ts)), (0, 0))) for ts in new_series]
    X_new = np.array(X_new_padded).transpose(0, 2, 1)
    
    X_new_rocket = rocket.transform(X_new)
    Xn = X_new_rocket.values if hasattr(X_new_rocket, 'values') else X_new_rocket
    
    # Predict
    y_pred = classifier.predict(Xn)
    y_pred_dense = y_pred.toarray() if hasattr(y_pred, 'toarray') else y_pred
    y_pred_text = [decode_label(vec, base_labels) for vec in y_pred_dense]
    
    saved_classifier_path = None
    
    # Fine-tune
    if fine_tune and new_labels is not None and time_series_train is not None:
        print("Fine-tuning model...")
        os.makedirs(models_dir, exist_ok=True)
        
        # Prepare labels
        if isinstance(new_labels, (list, tuple)) and (isinstance(new_labels[0], str) or isinstance(new_labels[0], np.str_)):
            y_new = np.array([encode_label(lbl, base_labels) for lbl in new_labels])
        else:
            y_new = np.asarray(new_labels)
            
        # Transform training data (needed for combined training)
        max_len_train = max(len(ts) for ts in time_series_train)
        X_train_padded = [np.pad(ts, ((0, max_len_train - len(ts)), (0, 0))) for ts in time_series_train]
        X_train = np.array(X_train_padded).transpose(0, 2, 1)
        X_train_rocket = rocket.transform(X_train)
        Xt = X_train_rocket.values if hasattr(X_train_rocket, 'values') else X_train_rocket
        
        # Combine data
        X_all = np.vstack([Xt, Xn])
        y_all = np.vstack([labels_train, y_new])
        
        # Re-train
        params = classifier.classifier.get_params() if hasattr(classifier, 'classifier') else {}
        if not params and best_lgbm_params:
            params = best_lgbm_params
            
        tuned_clf = LabelPowerset(lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, class_weight='balanced'))
        tuned_clf.fit(X_all, y_all)
        
        # Save
        joblib.dump(tuned_clf, cm_path)
        saved_classifier_path = os.path.join(models_dir, f'classifier_finetuned_{training_time}.pkl')
        joblib.dump(tuned_clf, saved_classifier_path)
        
        # Re-predict
        y_pred = tuned_clf.predict(Xn)
        y_pred_dense = y_pred.toarray() if hasattr(y_pred, 'toarray') else y_pred
        y_pred_text = [decode_label(vec, base_labels) for vec in y_pred_dense]
        
    return {
        'pred_dense': y_pred_dense, 
        'pred_text': y_pred_text, 
        'saved_classifier_path': saved_classifier_path
    }


def load_opto(shuffle_idx=None):
    """
    Load optogenetics data for labeling.
    """
    if shuffle_idx is None:
        if not os.path.exists(OPTO_DATA_DIR):
            return []
        runs = [d for d in os.listdir(OPTO_DATA_DIR) 
                if os.path.isdir(os.path.join(OPTO_DATA_DIR, d)) and d.isdigit()]
        if runs:
            shuffle_idx = max(int(d) for d in runs)
        else:
            return []
            
    json_path = os.path.join(OPTO_DATA_DIR, str(shuffle_idx), "sampled_infos.json")
    if not os.path.exists(json_path):
        print(f"Opto data file not found: {json_path}")
        return []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("--train", action="store_true", help="Train/evaluate the classifier.")
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna trial count (only used when not reusing saved params).")
    parser.add_argument("--use-saved-best-params", action="store_true", help="Reuse best params JSON if available.")
    parser.add_argument("--reload-models", action="store_true", help="Reload saved ROCKET/classifier models if available.")
    args = parser.parse_args()

    if not args.train:
        raise SystemExit("No action selected. Use --train to run training/evaluation.")

    time_series, new_labels, metadata, base_labels = load_and_process_data()
    if len(time_series) == 0:
        raise SystemExit("No data available for processing. Check BOUTS_INFO_PATH / ASTROCYTE_DATA_ROOT.")

    classifier_combined_labels(
        time_series,
        new_labels,
        base_labels,
        use_saved_best_params=bool(args.use_saved_best_params),
        n_trials=int(args.n_trials),
        reload_models=bool(args.reload_models),
    )
