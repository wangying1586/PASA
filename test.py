import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
import pandas as pd
from datetime import datetime
from tabulate import tabulate

from PASA import (
    get_dataset_class_names,
    CustomEfficientNet,
    load_efficientnet_model,
    calculate_circor_metrics
)


def load_preprocessed_data(args):
    """Load data from preprocessed .npy files"""
    print(f"Loading {args.dataset} dataset, task type: {args.task_type}")

    # Determine file paths based on dataset and task type
    if args.dataset == "SPRSound":
        data_path = os.path.join(args.data_dir, "SPRSound_2022+2023_processed_data", f"task{args.task_type}")
    elif args.dataset == "ICBHI":
        data_path = os.path.join(args.data_dir, "ICBHI2017_processed_data", args.task_type)
    elif args.dataset == "CirCor":
        data_path = os.path.join(args.data_dir, "CirCor_DigiScope_2022_processed_data")
        print(f"CirCor DigiScope path: {data_path}")
        print("Loading heart murmur detection data (Present/Absent/Unknown)")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    specs_path = os.path.join(data_path, "test_specs.npy")
    labels_path = os.path.join(data_path, "test_labels.npy")

    if not os.path.exists(specs_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Data files not found: {specs_path} or {labels_path}")

    try:
        specs = np.load(specs_path)
        labels = np.load(labels_path)

        # Ensure labels are integers
        if not np.issubdtype(labels.dtype, np.integer):
            print("Warning: Labels are not integer type, converting...")
            labels = labels.astype(np.int64)

        # Ensure correct shape for features
        if specs.ndim == 2:
            print("Warning: Features are 2D, reshaping to [samples, height, width]")
            n_samples = labels.shape[0]
            total_features = specs.shape[0] * specs.shape[1]
            feature_per_sample = total_features // n_samples
            height = int(np.sqrt(feature_per_sample))
            width = feature_per_sample // height
            specs = specs.reshape(n_samples, height, width)

        # Check sample count match
        if specs.shape[0] != labels.shape[0]:
            print(f"Warning: Sample count mismatch - specs:{specs.shape[0]}, labels:{labels.shape[0]}")
            min_samples = min(specs.shape[0], labels.shape[0])
            specs = specs[:min_samples]
            labels = labels[:min_samples]
            print(f"Trimmed to {min_samples} samples")

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    num_classes = len(np.unique(labels))
    print(f"Detected {num_classes} classes: {np.unique(labels)}")

    if args.dataset == "CirCor":
        print(f"CirCor DigiScope heart murmur detection dataset")
        print(f"Classes: {get_dataset_class_names(args.dataset, args.task_type)}")
        print("Evaluation metrics: W.acc (Weighted Accuracy), UAR (Unweighted Average Recall)")

    # Verify labels start from 0
    unique_labels = np.unique(labels)
    expected_labels = np.arange(num_classes)
    if not np.array_equal(unique_labels, expected_labels):
        print(f"Warning: Labels are not consecutive from 0! Actual: {unique_labels}, Expected: {expected_labels}")
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        print(f"Remapped labels, new unique values: {np.unique(labels)}")

    dataset = PreprocessedDataset(specs, labels)
    return dataset, num_classes


class PreprocessedDataset(Dataset):
    """Custom dataset for preprocessed data"""

    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        if len(spec.shape) == 2:
            spec = np.expand_dims(spec, axis=0)
        return spec, self.labels[idx]


def collate_fn_preprocessed(batch):
    """Collate function for preprocessed data"""
    specs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    if getattr(collate_fn_preprocessed, 'first_call', True):
        collate_fn_preprocessed.first_call = False

    specs = [torch.from_numpy(spec).float() if isinstance(spec, np.ndarray) else spec for spec in specs]
    labels = torch.tensor(labels, dtype=torch.long)
    stacked_specs = torch.stack(specs)
    return stacked_specs, labels


def load_test_dataset(args, test_set_name):
    """Load specific test dataset"""
    print(f"Loading {test_set_name} test dataset...")

    if args.dataset == 'SPRSound':
        task_type = int(args.task_type) if isinstance(args.task_type, str) else args.task_type

        if test_set_name == 'test_2022_intra':
            specs_path = os.path.join(args.data_dir,
                                      f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_intra_specs.npy')
            labels_path = os.path.join(args.data_dir,
                                       f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_intra_labels.npy')
        elif test_set_name == 'test_2022_inter':
            specs_path = os.path.join(args.data_dir,
                                      f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_inter_specs.npy')
            labels_path = os.path.join(args.data_dir,
                                       f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_inter_labels.npy')
        elif test_set_name == 'test_2023':
            specs_path = os.path.join(args.data_dir,
                                      f'SPRSound_2022+2023_processed_data/task{task_type}/test_2023_specs.npy')
            labels_path = os.path.join(args.data_dir,
                                       f'SPRSound_2022+2023_processed_data/task{task_type}/test_2023_labels.npy')
        else:
            raise ValueError(f"Unknown test set name: {test_set_name}")

        if not os.path.exists(specs_path) or not os.path.exists(labels_path):
            print(f"Warning: Could not find data for {test_set_name} at {specs_path} or {labels_path}")
            return None, None

        specs = np.load(specs_path)
        labels = np.load(labels_path)
        dataset = PreprocessedDataset(specs, labels)
        num_classes = len(np.unique(labels))
        print(f"Loaded {test_set_name} dataset with {len(dataset)} samples and {num_classes} classes")
        return dataset, num_classes

    elif args.dataset == 'CirCor':
        dataset, num_classes = load_preprocessed_data(args)
        print(f"Loaded CirCor DigiScope test dataset with {len(dataset)} samples and {num_classes} classes")
        return dataset, num_classes
    else:
        dataset, num_classes = load_preprocessed_data(args)
        print(f"Loaded {args.dataset} test dataset with {len(dataset)} samples and {num_classes} classes")
        return dataset, num_classes


def create_combined_dataset(datasets):
    """Merge multiple test sets into one dataset"""
    if not datasets or any(d is None for d in datasets):
        return None, None

    all_specs = []
    all_labels = []

    for dataset in datasets:
        for i in range(len(dataset)):
            spec, label = dataset[i]
            all_specs.append(spec)
            all_labels.append(label)

    all_specs = np.array(all_specs)
    all_labels = np.array(all_labels)
    combined_dataset = PreprocessedDataset(all_specs, all_labels)
    num_classes = len(np.unique(all_labels))

    print(f"Created combined dataset with {len(combined_dataset)} samples and {num_classes} classes")
    return combined_dataset, num_classes


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Visualize confusion matrix"""
    plt.figure(figsize=(12, 10))
    num_classes = len(class_names)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm_present = confusion_matrix(y_true, y_pred, labels=unique_labels)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cm[true_label, pred_label] = cm_present[i, j]

    row_sums = cm.sum(axis=1)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] != 0:
            cm_percent[i] = (cm[i] / row_sums[i]) * 100

    labels = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percent[i, j]
            if row_sums[i] == 0:
                row.append("N/A")
            else:
                row.append(f'{count}\n({percentage:.1f}%)')
        labels.append(row)

    df_cm = pd.DataFrame(labels, index=class_names, columns=class_names)
    cmap = sns.light_palette("#4169E1", as_cmap=True)
    annot_kws_size = 20 if num_classes <= 4 else 16 if num_classes <= 6 else 12

    plt.rcParams.update({'font.family': 'serif', 'font.size': 18})
    ax = sns.heatmap(cm_percent, annot=df_cm, fmt='', cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={'size': annot_kws_size})

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.title(title, fontsize=24, pad=20, fontfamily='serif')
    plt.xlabel('Predicted Label', fontsize=22, fontfamily='serif')
    plt.ylabel('True Label', fontsize=22, fontfamily='serif')
    plt.xticks(rotation=45, ha='right', fontsize=20, fontfamily='serif')
    plt.yticks(rotation=0, fontsize=20, fontfamily='serif')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_prob, class_names, title, save_path):
    """Visualize ROC curve"""
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 18,
                         'axes.linewidth': 1.5, 'lines.linewidth': 2.0,
                         'xtick.major.width': 1.5, 'ytick.major.width': 1.5})

    ax = plt.gca()
    ax.set_facecolor('#F5F8FA')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    unique_labels = np.unique(y_true)

    if n_classes == 2:
        if 1 in unique_labels:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[0], lw=3, label=f'{class_names[1]} (AUC = {roc_auc:.3f})')
    else:
        for i in range(n_classes):
            if i in unique_labels and np.sum(y_true == i) > 0:
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], lw=3, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], '--', color='#8B0000', alpha=0.8, lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='normal', fontfamily='serif')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='normal', fontfamily='serif')
    plt.title(title, fontsize=22, pad=10, fontweight='normal', fontfamily='serif')
    plt.xticks(fontsize=18, fontfamily='serif')
    plt.yticks(fontsize=18, fontfamily='serif')
    plt.legend(loc='lower right', fontsize=16, frameon=True, edgecolor='none', facecolor='white', ncol=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_pr_curve(y_true, y_pred_prob, class_names, title, save_path):
    """Visualize PR curve"""
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 18,
                         'axes.linewidth': 1.5, 'lines.linewidth': 2.0,
                         'xtick.major.width': 1.5, 'ytick.major.width': 1.5})

    ax = plt.gca()
    ax.set_facecolor('#F5F8FA')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    unique_labels = np.unique(y_true)

    if n_classes == 2:
        if 1 in unique_labels:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
            ap = average_precision_score(y_true, y_pred_prob[:, 1])
            plt.plot(recall, precision, color=colors[0], lw=3, label=f'{class_names[1]} (AP = {ap:.3f})')
    else:
        for i in range(n_classes):
            if i in unique_labels and np.sum(y_true == i) > 0:
                precision, recall, _ = precision_recall_curve(y_true == i, y_pred_prob[:, i])
                ap = average_precision_score(y_true == i, y_pred_prob[:, i])
                plt.plot(recall, precision, color=colors[i], lw=3, label=f'{class_names[i]} (AP={ap:.3f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=20, fontweight='normal', fontfamily='serif')
    plt.ylabel('Precision', fontsize=20, fontweight='normal', fontfamily='serif')
    plt.title(title, fontsize=22, pad=10, fontweight='normal', fontfamily='serif')
    plt.xticks(fontsize=18, fontfamily='serif')
    plt.yticks(fontsize=18, fontfamily='serif')
    plt.legend(loc='lower left', fontsize=16, frameon=True, edgecolor='none', facecolor='white', ncol=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def evaluate_model(model, data_loader, device, criterion, task_type=None, dataset=None):
    """Model evaluation - supports CirCor special metrics"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)

    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(data_loader)
    metrics = {'accuracy': accuracy, 'loss': avg_loss}

    class_names = get_dataset_class_names(dataset, task_type)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    f1 = f1_score(all_labels, all_preds, average='macro')
    metrics['f1'] = f1

    if dataset == "CirCor":
        circor_metrics = calculate_circor_metrics(all_labels, all_preds)
        metrics.update({
            'w_acc': circor_metrics['w_acc'],
            'uar': circor_metrics['uar'],
            'recall_present': circor_metrics['recall_present'],
            'recall_absent': circor_metrics['recall_absent'],
            'recall_unknown': circor_metrics['recall_unknown']
        })
        overall_score = circor_metrics['w_acc']
        macro_sensitivity = circor_metrics['recall_present']
        macro_specificity = circor_metrics['recall_absent']
        print(f"CirCor metrics - W.acc: {circor_metrics['w_acc']:.4f}, UAR: {circor_metrics['uar']:.4f}")
    else:
        sensitivities, specificities = [], []
        total_samples = np.sum(conf_matrix)

        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            fn = np.sum(conf_matrix[i, :]) - tp
            fp = np.sum(conf_matrix[:, i]) - tp
            tn = total_samples - (tp + fn + fp)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        macro_sensitivity = np.mean(sensitivities)
        macro_specificity = np.mean(specificities)
        average_score = (macro_sensitivity + macro_specificity) / 2
        harmonic_score = 2 * macro_sensitivity * macro_specificity / (macro_sensitivity + macro_specificity + 1e-9)
        overall_score = (average_score + harmonic_score) / 2

        metrics.update({
            'sensitivities': sensitivities,
            'specificities': specificities,
            'average_score': average_score,
            'harmonic_score': harmonic_score
        })

        if len(class_names) <= len(sensitivities):
            for i, class_name in enumerate(class_names):
                if i < len(sensitivities):
                    metrics[f'recall_{class_name.lower()}'] = sensitivities[i]

    metrics.update({
        'macro_sensitivity': macro_sensitivity,
        'macro_specificity': macro_specificity,
        'overall_score': overall_score,
        'confusion_matrix': conf_matrix.tolist()
    })

    return all_preds, all_labels, all_probs, metrics, class_names


def load_model_weights(model, weights_path, device):
    """Load model weights"""
    print(f"Loading model weights from: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                print("Warning: Checkpoint doesn't have model_state_dict or state_dict keys")
            if 'epoch' in checkpoint:
                print(f"Model was saved at epoch: {checkpoint['epoch']}")
        else:
            state_dict = checkpoint
            print("Note: Checkpoint is not a dictionary (direct state_dict)")

        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model weights")
        epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
        return model, epoch

    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise e


def create_ensemble_prediction(all_fold_probs, labels, class_names, vis_dir, test_set_name, dataset=None):
    """Create ensemble model predictions"""
    print(f"\n===== Creating Ensemble Prediction for {test_set_name} =====")

    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    correct = (ensemble_preds == labels).sum()
    accuracy = 100 * correct / len(labels) if len(labels) > 0 else 0
    ensemble_f1 = f1_score(labels, ensemble_preds, average='macro')
    ensemble_cm = confusion_matrix(labels, ensemble_preds, labels=range(len(class_names)))

    ensemble_metrics = {
        'accuracy': float(accuracy),
        'f1': float(ensemble_f1),
        'confusion_matrix': ensemble_cm.tolist()
    }

    if dataset == "CirCor":
        circor_metrics = calculate_circor_metrics(labels, ensemble_preds)
        ensemble_metrics.update({
            'w_acc': float(circor_metrics['w_acc']),
            'uar': float(circor_metrics['uar']),
            'recall_present': float(circor_metrics['recall_present']),
            'recall_absent': float(circor_metrics['recall_absent']),
            'recall_unknown': float(circor_metrics['recall_unknown']),
            'overall_score': float(circor_metrics['w_acc'])
        })

        print(f"Ensemble Accuracy: {accuracy:.2f}%")
        print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
        print(f"Ensemble W.acc: {circor_metrics['w_acc']:.4f}")
        print(f"Ensemble UAR: {circor_metrics['uar']:.4f}")

        ensemble_macro_sensitivity = circor_metrics['recall_present']
        ensemble_macro_specificity = circor_metrics['recall_absent']
    else:
        ensemble_sensitivities, ensemble_specificities = [], []
        total_samples = np.sum(ensemble_cm)

        for i in range(ensemble_cm.shape[0]):
            tp = ensemble_cm[i, i]
            fn = np.sum(ensemble_cm[i, :]) - tp
            fp = np.sum(ensemble_cm[:, i]) - tp
            tn = total_samples - (tp + fn + fp)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ensemble_sensitivities.append(sensitivity)
            ensemble_specificities.append(specificity)

        ensemble_macro_sensitivity = np.mean(ensemble_sensitivities)
        ensemble_macro_specificity = np.mean(ensemble_specificities)
        ensemble_average_score = (ensemble_macro_sensitivity + ensemble_macro_specificity) / 2
        ensemble_harmonic_score = 2 * ensemble_macro_sensitivity * ensemble_macro_specificity / (
                ensemble_macro_sensitivity + ensemble_macro_specificity + 1e-9)
        ensemble_overall_score = (ensemble_average_score + ensemble_harmonic_score) / 2

        ensemble_metrics.update({
            'sensitivities': [float(s) for s in ensemble_sensitivities],
            'specificities': [float(s) for s in ensemble_specificities],
            'average_score': float(ensemble_average_score),
            'harmonic_score': float(ensemble_harmonic_score),
            'overall_score': float(ensemble_overall_score)
        })

        print(f"Ensemble Accuracy: {accuracy:.2f}%")
        print(f"Ensemble F1: {ensemble_f1:.4f}, Sensitivity: {ensemble_macro_sensitivity:.4f}")
        print(f"Ensemble Specificity: {ensemble_macro_specificity:.4f}, Overall: {ensemble_overall_score:.4f}")

    ensemble_metrics.update({
        'macro_sensitivity': float(ensemble_macro_sensitivity),
        'macro_specificity': float(ensemble_macro_specificity)
    })

    plot_confusion_matrix(labels, ensemble_preds, class_names,
                          f'Confusion Matrix - Ensemble Model - {test_set_name}',
                          os.path.join(vis_dir, f'confusion_matrix_ensemble_{test_set_name}.png'))
    plot_roc_curve(labels, ensemble_probs, class_names,
                   f'ROC Curve - Ensemble Model - {test_set_name}',
                   os.path.join(vis_dir, f'roc_curve_ensemble_{test_set_name}.png'))
    plot_pr_curve(labels, ensemble_probs, class_names,
                  f'Precision-Recall Curve - Ensemble Model - {test_set_name}',
                  os.path.join(vis_dir, f'pr_curve_ensemble_{test_set_name}.png'))

    return ensemble_preds, ensemble_probs, ensemble_metrics


def compile_results_table(all_test_sets_results, test_set_names, dataset=None):
    """Compile all results into a table"""
    if dataset == "CirCor":
        headers = ["Dataset", "Model", "Accuracy (%)", "F1 Score", "W.acc", "UAR",
                   "Recall Present", "Recall Absent", "Recall Unknown", "Overall Score"]
    else:
        headers = ["Dataset", "Model", "Accuracy (%)", "F1 Score", "Sensitivity", "Specificity",
                   "Average Score", "Harmonic Score", "Overall Score"]

    table_data = []

    for test_set_name in test_set_names:
        results = all_test_sets_results.get(test_set_name, {})

        # Add fold results
        for fold in range(1, 6):
            fold_results = results.get(f'fold_{fold}', None)
            if fold_results:
                fold_metrics = fold_results.get('metrics', {})
                if dataset == "CirCor":
                    table_data.append([test_set_name, f"Fold {fold}",
                                       f"{fold_metrics.get('accuracy', 0):.2f}",
                                       f"{fold_metrics.get('f1', 0):.4f}",
                                       f"{fold_metrics.get('w_acc', 0):.4f}",
                                       f"{fold_metrics.get('uar', 0):.4f}",
                                       f"{fold_metrics.get('recall_present', 0):.4f}",
                                       f"{fold_metrics.get('recall_absent', 0):.4f}",
                                       f"{fold_metrics.get('recall_unknown', 0):.4f}",
                                       f"{fold_metrics.get('overall_score', 0):.4f}"])
                else:
                    table_data.append([test_set_name, f"Fold {fold}",
                                       f"{fold_metrics.get('accuracy', 0):.2f}",
                                       f"{fold_metrics.get('f1', 0):.4f}",
                                       f"{fold_metrics.get('macro_sensitivity', 0):.4f}",
                                       f"{fold_metrics.get('macro_specificity', 0):.4f}",
                                       f"{fold_metrics.get('average_score', 0):.4f}",
                                       f"{fold_metrics.get('harmonic_score', 0):.4f}",
                                       f"{fold_metrics.get('overall_score', 0):.4f}"])

        # Add average results
        avg_metrics = results.get('avg_metrics', None)
        if avg_metrics:
            if dataset == "CirCor":
                table_data.append([test_set_name, "Average Folds",
                                   f"{avg_metrics.get('accuracy', 0):.2f}",
                                   f"{avg_metrics.get('f1', 0):.4f}",
                                   f"{avg_metrics.get('w_acc', 0):.4f}",
                                   f"{avg_metrics.get('uar', 0):.4f}",
                                   f"{avg_metrics.get('recall_present', 0):.4f}",
                                   f"{avg_metrics.get('recall_absent', 0):.4f}",
                                   f"{avg_metrics.get('recall_unknown', 0):.4f}",
                                   f"{avg_metrics.get('overall_score', 0):.4f}"])
            else:
                table_data.append([test_set_name, "Average Folds",
                                   f"{avg_metrics.get('accuracy', 0):.2f}",
                                   f"{avg_metrics.get('f1', 0):.4f}",
                                   f"{avg_metrics.get('macro_sensitivity', 0):.4f}",
                                   f"{avg_metrics.get('macro_specificity', 0):.4f}",
                                   f"{avg_metrics.get('average_score', 0):.4f}",
                                   f"{avg_metrics.get('harmonic_score', 0):.4f}",
                                   f"{avg_metrics.get('overall_score', 0):.4f}"])

        # Add final model results
        final_model_results = results.get('final_model', None)
        if final_model_results:
            final_metrics = final_model_results.get('metrics', {})
            if dataset == "CirCor":
                table_data.append([test_set_name, "Final Model",
                                   f"{final_metrics.get('accuracy', 0):.2f}",
                                   f"{final_metrics.get('f1', 0):.4f}",
                                   f"{final_metrics.get('w_acc', 0):.4f}",
                                   f"{final_metrics.get('uar', 0):.4f}",
                                   f"{final_metrics.get('recall_present', 0):.4f}",
                                   f"{final_metrics.get('recall_absent', 0):.4f}",
                                   f"{final_metrics.get('recall_unknown', 0):.4f}",
                                   f"{final_metrics.get('overall_score', 0):.4f}"])
            else:
                table_data.append([test_set_name, "Final Model",
                                   f"{final_metrics.get('accuracy', 0):.2f}",
                                   f"{final_metrics.get('f1', 0):.4f}",
                                   f"{final_metrics.get('macro_sensitivity', 0):.4f}",
                                   f"{final_metrics.get('macro_specificity', 0):.4f}",
                                   f"{final_metrics.get('average_score', 0):.4f}",
                                   f"{final_metrics.get('harmonic_score', 0):.4f}",
                                   f"{final_metrics.get('overall_score', 0):.4f}"])

        # Add ensemble results
        ensemble_metrics = results.get('ensemble_metrics', None)
        if ensemble_metrics:
            if dataset == "CirCor":
                table_data.append([test_set_name, "Ensemble",
                                   f"{ensemble_metrics.get('accuracy', 0):.2f}",
                                   f"{ensemble_metrics.get('f1', 0):.4f}",
                                   f"{ensemble_metrics.get('w_acc', 0):.4f}",
                                   f"{ensemble_metrics.get('uar', 0):.4f}",
                                   f"{ensemble_metrics.get('recall_present', 0):.4f}",
                                   f"{ensemble_metrics.get('recall_absent', 0):.4f}",
                                   f"{ensemble_metrics.get('recall_unknown', 0):.4f}",
                                   f"{ensemble_metrics.get('overall_score', 0):.4f}"])
            else:
                table_data.append([test_set_name, "Ensemble",
                                   f"{ensemble_metrics.get('accuracy', 0):.2f}",
                                   f"{ensemble_metrics.get('f1', 0):.4f}",
                                   f"{ensemble_metrics.get('macro_sensitivity', 0):.4f}",
                                   f"{ensemble_metrics.get('macro_specificity', 0):.4f}",
                                   f"{ensemble_metrics.get('average_score', 0):.4f}",
                                   f"{ensemble_metrics.get('harmonic_score', 0):.4f}",
                                   f"{ensemble_metrics.get('overall_score', 0):.4f}"])

        table_data.append([''] * len(headers))

    if table_data:
        table_data.pop()

    return headers, table_data


def main():
    parser = argparse.ArgumentParser(description='Test Five-fold Cross Validation Models')
    parser.add_argument('--dataset', type=str, default='SPRSound',
                        choices=['SPRSound', 'ICBHI', 'CirCor'], help='Dataset name')
    parser.add_argument('--task_type', type=str, default='22',
                        help='Task type (11/12/21/22 for SPRSound, binary/multiclass for ICBHI)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory containing preprocessed data')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory containing trained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory for data loader')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor for data loader')

    args = parser.parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'test_results/CirCor/heart_murmur' if args.dataset == 'CirCor' else f'test_results/{args.dataset}/task_{args.task_type}'
    vis_dir = os.path.join(base_dir, timestamp)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Created visualization directory: {vis_dir}")

    config = vars(args)
    config.update({'timestamp': timestamp, 'vis_dir': vis_dir})
    with open(os.path.join(vis_dir, 'test_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_set_names = ['test_2022_intra', 'test_2022_inter', 'test_2023',
                      'test_2022_combined'] if args.dataset == 'SPRSound' else ['test']

    # Load test datasets
    test_datasets = {}
    for test_set_name in test_set_names:
        if args.dataset == 'SPRSound':
            if test_set_name != 'test_2022_combined':
                dataset, num_classes = load_test_dataset(args, test_set_name)
                if dataset:
                    test_datasets[test_set_name] = dataset
                    print(f"Loaded {test_set_name} dataset with {len(dataset)} samples")
        else:
            dataset, num_classes = load_test_dataset(args, test_set_name)
            if dataset:
                test_datasets[test_set_name] = dataset
                print(f"Loaded {args.dataset} {test_set_name} dataset with {len(dataset)} samples")

    # Create combined dataset for SPRSound
    if args.dataset == 'SPRSound' and 'test_2022_intra' in test_datasets and 'test_2022_inter' in test_datasets:
        combined_dataset, num_classes = create_combined_dataset(
            [test_datasets['test_2022_intra'], test_datasets['test_2022_inter']])
        if combined_dataset:
            test_datasets['test_2022_combined'] = combined_dataset
            print(f"Created test_2022_combined dataset with {len(combined_dataset)} samples")

    if not test_datasets:
        print("Error: No test datasets found. Please check the data paths.")
        return

    class_names = get_dataset_class_names(args.dataset, args.task_type)
    print(f"Class names: {class_names}")

    # Determine number of classes
    if args.dataset == 'SPRSound':
        task_map = {'11': 2, '12': 7, '21': 3, '22': 5}
        num_classes = task_map.get(str(args.task_type), len(np.unique(
            [next(iter(test_datasets.values()))[i][1] for i in range(len(next(iter(test_datasets.values()))))])))
    elif args.dataset == 'CirCor':
        num_classes = 3
    else:
        first_dataset = next(iter(test_datasets.values()))
        num_classes = len(np.unique([first_dataset[i][1] for i in range(len(first_dataset))]))

    all_test_sets_results = {test_name: {} for test_name in test_datasets.keys()}
    fold_probs_by_dataset = {test_name: [] for test_name in test_datasets.keys()}
    criterion = nn.CrossEntropyLoss()

    # Test fold models
    for fold in range(1, 6):
        model_path = os.path.join(args.exp_dir, 'models', f'best_fold_{fold}.pth')
        if not os.path.exists(model_path):
            print(f"Warning: Model file for fold {fold} not found at {model_path}")
            continue

        print(f"\n{'=' * 50}\nTesting fold {fold} model on all datasets\n{'=' * 50}")

        base_model = load_efficientnet_model(num_classes=num_classes)
        model = CustomEfficientNet(base_model)
        model, _ = load_model_weights(model, model_path, device)
        model = model.to(device)

        for test_set_name, test_dataset in test_datasets.items():
            print(f"\n--- Evaluating fold {fold} on {test_set_name} ---")
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate_fn_preprocessed, num_workers=args.num_workers,
                                     pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)

            preds, labels, probs, metrics, _ = evaluate_model(model, test_loader, device, criterion, args.task_type,
                                                              args.dataset)

            all_test_sets_results[test_set_name][f'fold_{fold}'] = {
                'metrics': metrics, 'preds': preds.tolist(), 'labels': labels.tolist()
            }
            fold_probs_by_dataset[test_set_name].append(probs)

            print(f"\n===== Fold {fold} Results on {test_set_name} =====")
            print(f"Accuracy: {metrics['accuracy']:.2f}%, F1: {metrics['f1']:.4f}")
            if args.dataset == "CirCor":
                print(f"W.acc: {metrics.get('w_acc', 0):.4f}, UAR: {metrics.get('uar', 0):.4f}")
            else:
                print(
                    f"Sensitivity: {metrics['macro_sensitivity']:.4f}, Specificity: {metrics['macro_specificity']:.4f}")

            plot_confusion_matrix(labels, preds, class_names, f'Confusion Matrix - Fold {fold} - {test_set_name}',
                                  os.path.join(vis_dir, f'confusion_matrix_fold_{fold}_{test_set_name}.png'))
            plot_roc_curve(labels, probs, class_names, f'ROC Curve - Fold {fold} - {test_set_name}',
                           os.path.join(vis_dir, f'roc_curve_fold_{fold}_{test_set_name}.png'))
            plot_pr_curve(labels, probs, class_names, f'PR Curve - Fold {fold} - {test_set_name}',
                          os.path.join(vis_dir, f'pr_curve_fold_{fold}_{test_set_name}.png'))

    # Create ensemble predictions
    for test_set_name, fold_probs in fold_probs_by_dataset.items():
        if len(fold_probs) > 1:
            print(f"\n{'=' * 50}\nCreating ensemble prediction for {test_set_name}\n{'=' * 50}")
            labels = np.array(all_test_sets_results[test_set_name]['fold_1']['labels'])
            ensemble_preds, ensemble_probs, ensemble_metrics = create_ensemble_prediction(
                fold_probs, labels, class_names, vis_dir, test_set_name, args.dataset)
            all_test_sets_results[test_set_name]['ensemble_metrics'] = ensemble_metrics

    # Test final model
    final_model_path = os.path.join(args.exp_dir, 'models', 'final_model.pth')
    if os.path.exists(final_model_path):
        print(f"\n{'=' * 50}\nTesting final model on all datasets\n{'=' * 50}")
        base_model = load_efficientnet_model(num_classes=num_classes)
        model = CustomEfficientNet(base_model)
        model, epoch = load_model_weights(model, final_model_path, device)
        model = model.to(device)

        for test_set_name, test_dataset in test_datasets.items():
            print(f"\n--- Evaluating final model on {test_set_name} ---")
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate_fn_preprocessed, num_workers=args.num_workers,
                                     pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)

            preds, labels, probs, metrics, _ = evaluate_model(model, test_loader, device, criterion, args.task_type,
                                                              args.dataset)

            all_test_sets_results[test_set_name]['final_model'] = {
                'metrics': metrics, 'preds': preds.tolist(), 'labels': labels.tolist(), 'saved_epoch': epoch
            }

            print(f"\n===== Final Model Results on {test_set_name} =====")
            print(f"Accuracy: {metrics['accuracy']:.2f}%, F1: {metrics['f1']:.4f}")

            plot_confusion_matrix(labels, preds, class_names, f'Confusion Matrix - Final Model - {test_set_name}',
                                  os.path.join(vis_dir, f'confusion_matrix_final_{test_set_name}.png'))
            plot_roc_curve(labels, probs, class_names, f'ROC Curve - Final Model - {test_set_name}',
                           os.path.join(vis_dir, f'roc_curve_final_{test_set_name}.png'))
            plot_pr_curve(labels, probs, class_names, f'PR Curve - Final Model - {test_set_name}',
                          os.path.join(vis_dir, f'pr_curve_final_{test_set_name}.png'))
    else:
        print(f"Warning: Final model file not found at {final_model_path}")

    # Calculate average metrics
    for test_set_name, results in all_test_sets_results.items():
        fold_metrics = [results[f'fold_{fold}']['metrics'] for fold in range(1, 6) if
                        f'fold_{fold}' in results and 'metrics' in results[f'fold_{fold}']]

        if fold_metrics:
            avg_metrics = {
                'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
                'f1': np.mean([m['f1'] for m in fold_metrics]),
                'macro_sensitivity': np.mean([m['macro_sensitivity'] for m in fold_metrics]),
                'macro_specificity': np.mean([m['macro_specificity'] for m in fold_metrics]),
                'overall_score': np.mean([m['overall_score'] for m in fold_metrics])
            }

            if args.dataset == "CirCor":
                avg_metrics.update({
                    'w_acc': np.mean([m.get('w_acc', 0) for m in fold_metrics]),
                    'uar': np.mean([m.get('uar', 0) for m in fold_metrics]),
                    'recall_present': np.mean([m.get('recall_present', 0) for m in fold_metrics]),
                    'recall_absent': np.mean([m.get('recall_absent', 0) for m in fold_metrics]),
                    'recall_unknown': np.mean([m.get('recall_unknown', 0) for m in fold_metrics])
                })
            else:
                avg_metrics.update({
                    'average_score': np.mean([m.get('average_score', 0) for m in fold_metrics]),
                    'harmonic_score': np.mean([m.get('harmonic_score', 0) for m in fold_metrics])
                })

            results['avg_metrics'] = avg_metrics
            print(f"\n===== Average Metrics for {test_set_name} =====")
            print(f"Avg Accuracy: {avg_metrics['accuracy']:.2f}%, Avg F1: {avg_metrics['f1']:.4f}")

    # Compile and print results table
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL TEST RESULTS")
    print("=" * 80)
    headers, table_data = compile_results_table(all_test_sets_results, test_set_names, args.dataset)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    with open(os.path.join(vis_dir, 'results_summary.txt'), 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save detailed predictions
    detailed_results_dir = os.path.join(vis_dir, 'detailed_predictions')
    os.makedirs(detailed_results_dir, exist_ok=True)

    for test_set_name, results in all_test_sets_results.items():
        if 'final_model' in results:
            df = pd.DataFrame({
                'true_label': results['final_model']['labels'],
                'pred_label': results['final_model']['preds'],
                'correct': np.array(results['final_model']['labels']) == np.array(results['final_model']['preds'])
            })
            df.to_csv(os.path.join(detailed_results_dir, f'{test_set_name}_final_model_predictions.csv'), index=False)

        if 'ensemble_metrics' in results and len(fold_probs_by_dataset[test_set_name]) > 0:
            labels = np.array(all_test_sets_results[test_set_name]['fold_1']['labels'])
            ensemble_probs = np.mean(fold_probs_by_dataset[test_set_name], axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            df = pd.DataFrame({'true_label': labels, 'pred_label': ensemble_preds, 'correct': labels == ensemble_preds})
            df.to_csv(os.path.join(detailed_results_dir, f'{test_set_name}_ensemble_predictions.csv'), index=False)

    # Save all results to JSON
    results_filename = f'CirCor_heart_murmur_test_results.json' if args.dataset == "CirCor" else f'{args.dataset}_task_{args.task_type}_test_results.json'

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(os.path.join(vis_dir, results_filename), 'w') as f:
        json.dump(all_test_sets_results, f, indent=4, cls=NumpyEncoder)

    print(f"\nTesting completed successfully!")
    print(f"All results saved to: {vis_dir}")
    print(f"Detailed predictions: {detailed_results_dir}")
    print(f"Visualizations saved as PNG files")
    print(f"Summary table: results_summary.txt")
    print(f"Results saved to: {results_filename}")


if __name__ == "__main__":
    main()