import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from efficientnet_pytorch import EfficientNet
import time
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings
from PASA import TwoPhaseSACStrategy, calculate_balanced_accuracy, apply_simple_anti_memorization_fix
import json
from sklearn.metrics import confusion_matrix, f1_score

warnings.filterwarnings('ignore')


def calculate_circor_metrics(y_true, y_pred):
    """Calculate CirCor DigiScope evaluation metrics (W.acc and UAR)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cp = np.sum((y_true == 0) & (y_pred == 0))
    ca = np.sum((y_true == 1) & (y_pred == 1))
    cu = np.sum((y_true == 2) & (y_pred == 2))

    tp, ta, tu = np.sum(y_true == 0), np.sum(y_true == 1), np.sum(y_true == 2)

    w_acc = (5 * cp + 3 * cu + ca) / (5 * tp + 3 * tu + ta) if (5 * tp + 3 * tu + ta) > 0 else 0.0

    recall_present = cp / tp if tp > 0 else 0.0
    recall_absent = ca / ta if ta > 0 else 0.0
    recall_unknown = cu / tu if tu > 0 else 0.0
    uar = (recall_present + recall_absent + recall_unknown) / 3.0

    return {
        'w_acc': w_acc,
        'uar': uar,
        'recall_present': recall_present,
        'recall_absent': recall_absent,
        'recall_unknown': recall_unknown
    }


class IndexAwareDataset(torch.utils.data.Dataset):
    """Dataset wrapper that returns sample indices along with data"""

    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            data, label = self.dataset[idx] if hasattr(self.dataset, '__getitem__') else self.dataset.__getitem__(idx)
            return data, label, idx
        except Exception:
            return torch.zeros(1, 128, 1000), torch.tensor(0, dtype=torch.long), idx

    def get_targets(self):
        if hasattr(self.dataset, 'get_targets'):
            return self.dataset.get_targets()
        elif hasattr(self.dataset, 'labels'):
            return self.dataset.labels.tolist() if hasattr(self.dataset.labels, 'tolist') else list(self.dataset.labels)
        else:
            return [int(self.dataset[i][1]) for i in range(len(self.dataset))]


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Sampler for imbalanced datasets"""

    def __init__(self, dataset, num_samples=None, replacement=True):
        self.dataset = dataset
        self.replacement = replacement

        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            self.dataset_size = len(dataset)
            self.subset_indices = dataset.indices
            all_labels = dataset.dataset.labels if hasattr(dataset.dataset, 'labels') else dataset.dataset.get_targets()
            self.labels = [all_labels[i] for i in self.subset_indices]
        else:
            self.dataset_size = len(dataset)
            self.labels = dataset.labels.tolist() if hasattr(dataset, 'labels') else dataset.get_targets()

        self.class_counts = {}
        for label in self.labels:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

        total_samples = len(self.labels)
        num_classes = len(self.class_counts)
        self.weights = [total_samples / (num_classes * self.class_counts[label]) for label in self.labels]
        self.num_samples = num_samples if num_samples is not None else self.dataset_size

    def __iter__(self):
        if self.replacement:
            sampled_indices = torch.multinomial(torch.tensor(self.weights, dtype=torch.float),
                                                self.num_samples, replacement=True).tolist()
        else:
            sampled_indices = torch.multinomial(torch.tensor(self.weights, dtype=torch.float),
                                                min(self.num_samples, len(self.weights)),
                                                replacement=False).tolist()
        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples


def get_dataset_class_names(dataset, task_type):
    """Get class names for different datasets"""
    if dataset == "SPRSound":
        if task_type == "11":
            return ['Normal', 'Adventitious']
        elif task_type == "12":
            return ['Normal', 'Rhonchi', 'Wheeze', 'Stridor', 'Coarse Crackle', 'Fine Crackle', 'Wheeze+Crackle']
        elif task_type == "21":
            return ['Normal', 'Poor Quality', 'Adventitious']
        elif task_type == "22":
            return ['Normal', 'Poor Quality', 'CAS', 'DAS', 'CAS & DAS']
    elif dataset == "ICBHI":
        if task_type == "binary":
            return ['Normal', 'Abnormal']
        elif task_type == "multiclass":
            return ['Normal', 'Crackle', 'Wheeze', 'Wheeze+Crackle']
    elif dataset == "CirCor":
        return ['Present', 'Absent', 'Unknown']
    return [f"Class {i}" for i in range(10)]


class PreprocessedDataset(torch.utils.data.Dataset):
    """Dataset for loading preprocessed .npy files"""

    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = torch.from_numpy(self.specs[idx].astype(np.float32))
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        return spec, label

    def get_targets(self):
        return self.labels.tolist()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    """Cosine annealing LR scheduler with warmup"""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 target_lr=0.0001, warmup_start_lr=0.001, min_lr=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = max(total_epochs, warmup_epochs + 10)
        self.target_lr = target_lr
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr if min_lr is not None else target_lr / 100
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return [self.target_lr for _ in self.optimizer.param_groups]
            progress = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * progress
                    for _ in self.optimizer.param_groups]
        else:
            remaining_epochs = self.total_epochs - self.warmup_epochs
            if remaining_epochs <= 0:
                return [self.target_lr for _ in self.optimizer.param_groups]
            progress = min((self.last_epoch - self.warmup_epochs) / remaining_epochs, 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (self.target_lr - self.min_lr) * cosine
                    for _ in self.optimizer.param_groups]


def load_efficientnet_model(num_classes=None):
    """Load EfficientNet-B4 model and adapt for audio input"""
    model = EfficientNet.from_pretrained('efficientnet-b4')

    first_conv_layer = model._conv_stem
    weight_data = first_conv_layer.weight.data
    original_in_channels = weight_data.shape[1]
    new_in_channels = 1

    if new_in_channels < original_in_channels:
        weight_data = weight_data[:, :new_in_channels, :, :]
    elif new_in_channels > original_in_channels:
        additional_channels_weight = torch.randn(weight_data.shape[0], new_in_channels - original_in_channels,
                                                 weight_data.shape[2], weight_data.shape[3])
        weight_data = torch.cat([weight_data, additional_channels_weight], dim=1)

    first_conv_layer.weight.data = weight_data

    if first_conv_layer.bias is not None:
        bias_data = first_conv_layer.bias.data
        if new_in_channels < original_in_channels:
            bias_data = bias_data[:new_in_channels]
        elif new_in_channels > original_in_channels:
            additional_bias = torch.zeros(new_in_channels - original_in_channels)
            bias_data = torch.cat([bias_data, additional_bias], dim=0)
        first_conv_layer.bias.data = bias_data

    if num_classes is not None:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)

    return model


class CustomEfficientNet(nn.Module):
    """Custom EfficientNet with feature extraction capability"""

    def __init__(self, base_model):
        super(CustomEfficientNet, self).__init__()
        self.base_model = base_model
        self.feature_dim = self.base_model._fc.in_features

    def forward(self, x):
        return self.base_model(x)

    def extract_features(self, x):
        original_fc = self.base_model._fc
        self.base_model._fc = nn.Identity()
        with torch.no_grad():
            features = self.base_model(x)
        self.base_model._fc = original_fc
        return features


def create_model(num_classes, args):
    """Create and initialize model"""
    base_model = load_efficientnet_model(num_classes)
    return CustomEfficientNet(base_model)


def load_preprocessed_data(args):
    """Load data from preprocessed .npy files"""
    if args.dataset == "SPRSound":
        data_path = os.path.join(args.data_dir, "SPRSound_2022+2023_processed_data", f"task{args.task_type}")
    elif args.dataset == "ICBHI":
        data_path = os.path.join(args.data_dir, "ICBHI2017_processed_data", args.task_type)
    elif args.dataset == "CirCor":
        data_path = os.path.join(args.data_dir, "CirCor_DigiScope_2022_processed_data")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    specs_path = os.path.join(data_path, "train_specs.npy")
    labels_path = os.path.join(data_path, "train_labels.npy")

    if not os.path.exists(specs_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Data files not found: {specs_path} and {labels_path}")

    specs = np.load(specs_path)
    labels = np.load(labels_path)

    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)

    if specs.ndim == 2:
        n_samples = labels.shape[0]
        total_features = specs.shape[0] * specs.shape[1]
        feature_per_sample = total_features // n_samples
        height = int(np.sqrt(feature_per_sample))
        width = feature_per_sample // height
        specs = specs.reshape(n_samples, height, width)

    if specs.shape[0] != labels.shape[0]:
        min_samples = min(specs.shape[0], labels.shape[0])
        specs = specs[:min_samples]
        labels = labels[:min_samples]

    num_classes = len(np.unique(labels))
    unique_labels = np.unique(labels)
    expected_labels = np.arange(num_classes)

    if not np.array_equal(unique_labels, expected_labels):
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])

    dataset = PreprocessedDataset(specs, labels)
    return dataset, num_classes


def collate_fn_with_indices(batch):
    """Collate function supporting indices"""
    try:
        specs, labels, indices = zip(*batch)

        processed_specs = []
        for spec in specs:
            if isinstance(spec, torch.Tensor):
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)
                elif spec.dim() == 1:
                    length = spec.shape[0]
                    height = int(np.sqrt(length))
                    width = length // height
                    spec = spec.reshape(1, height, width)
                processed_specs.append(spec)
            else:
                spec = torch.from_numpy(spec.astype(np.float32))
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)
                processed_specs.append(spec)

        if len(processed_specs) > 0:
            max_h = max(s.shape[-2] for s in processed_specs)
            max_w = max(s.shape[-1] for s in processed_specs)
            channels = processed_specs[0].shape[0]
            batch_size = len(processed_specs)

            unified_specs = torch.zeros(batch_size, channels, max_h, max_w)
            for i, spec in enumerate(processed_specs):
                h, w = spec.shape[-2], spec.shape[-1]
                unified_specs[i, :, :h, :w] = spec

            processed_labels = [torch.tensor(int(label), dtype=torch.long) if not isinstance(label, torch.Tensor)
                                else label for label in labels]
            labels_tensor = torch.stack(processed_labels)
            indices_list = list(indices)

            return unified_specs, labels_tensor, indices_list
        else:
            raise ValueError("Empty batch")

    except Exception:
        batch_size = len(batch)
        return (torch.zeros(batch_size, 1, 128, 1000),
                torch.zeros(batch_size, dtype=torch.long),
                list(range(batch_size)))


def train(args, device):
    """
    Main training function using two-phase SAC strategy with 5 key steps:
    (1) Feature extraction and state construction
    (2) Action sampling via SAC network
    (3) HBS execution strategy
    (4) Performance evaluation and reward computation
    (5) Joint optimization of augmentation policies and diagnostic models
    """
    set_seed(args.seed)

    train_dataset, num_classes = load_preprocessed_data(args)
    indexed_train_dataset = IndexAwareDataset(train_dataset)

    all_labels = train_dataset.labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    indices = np.arange(len(train_dataset))
    fold_splits = list(skf.split(indices, all_labels))

    exp_dir = create_experiment_dir(args)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    all_folds_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n===== Fold {fold + 1}/5 =====")

        train_subset = Subset(indexed_train_dataset, train_idx)
        val_subset = Subset(indexed_train_dataset, val_idx)

        balanced_sampler = ImbalancedDatasetSampler(train_subset)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=balanced_sampler,
            collate_fn=collate_fn_with_indices,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn_with_indices,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )

        model = create_model(num_classes, args).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epoch,
            total_epochs=args.epoch,
            target_lr=args.lr,
            warmup_start_lr=args.warmup_lr,
            min_lr=1e-6
        )

        patience = 30
        best_val_score = 0.0
        best_epoch = -1
        early_stop_counter = 0

        # Initialize two-phase SAC strategy if enabled
        sac_strategy = None
        if args.use_adaptive_aug:
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 128, 1000).to(device)
                feature_dim = model.feature_dim

            aug_operations = ["time_mask", "frequency_mask", "noise_injection", "random_quantization",
                              "spectral_contrast", "harmonic_perturbation", "breathing_cycle_stretch",
                              "low_freq_emphasis"]

            sac_strategy = TwoPhaseSACStrategy(
                n_classes=num_classes,
                feature_dim=feature_dim,
                aug_operations=aug_operations,
                device=device,
                buffer_capacity=args.sac_buffer_size,
                gamma=args.gamma,
                tau=args.sac_tau,
                actor_lr=args.sac_lr_actor,
                critic_lr=args.sac_lr_critic,
                validation_set=val_loader,
                n_magnitude_levels=args.n_magnitude_levels,
                phase2_trigger_patience=args.phase2_trigger_patience,
                phase2_trigger_threshold=args.phase2_trigger_threshold,
                debug_mode=True
            )
            sac_strategy = apply_simple_anti_memorization_fix(sac_strategy)

        # Training loop
        for epoch in range(args.epoch):
            model.train()
            running_loss = 0.0
            batch_count = 0

            for batch_idx, batch_data in enumerate(train_loader):
                if len(batch_data) == 3:
                    inputs, labels, sample_indices = batch_data
                else:
                    inputs, labels = batch_data
                    sample_indices = None

                inputs, labels = inputs.to(device), labels.to(device)

                # Step 1: Feature extraction and state construction
                # Step 2: Action sampling via SAC network
                # Step 3: HBS execution strategy
                # Step 4: Performance evaluation and reward computation
                if args.use_adaptive_aug and sac_strategy is not None:
                    augmented_inputs, augmented_labels, avg_reward, sac_losses = sac_strategy.train_step(
                        model=model,
                        criterion=criterion,
                        x=inputs,
                        y=labels,
                        epoch=epoch,
                        total_epochs=args.epoch,
                        sample_indices=sample_indices
                    )

                    if 'early_stop' in sac_losses and sac_losses.get('early_stop', False):
                        break

                    optimizer.zero_grad()
                    outputs = model(augmented_inputs)
                    loss = criterion(outputs, augmented_labels)
                else:
                    augmented_inputs, augmented_labels = inputs, labels
                    optimizer.zero_grad()
                    outputs = model(augmented_inputs)
                    loss = criterion(outputs, augmented_labels)

                # Step 5: Joint optimization of augmentation policies and diagnostic models
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            if args.use_adaptive_aug and sac_strategy is not None:
                sac_strategy.check_epoch_end_phase_transition(model, criterion, epoch)

            train_loss = running_loss / max(1, batch_count)
            scheduler.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_results = evaluate(
                    model=model,
                    data_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    val_set_name=f"fold_{fold + 1}_val",
                    epoch=epoch,
                    task_type=args.task_type,
                    dataset=args.dataset
                )

            current_score = val_results['overall_score']
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                early_stop_counter = 0

                best_model_path = os.path.join(exp_dir, 'models', f'best_fold_{fold + 1}.pth')
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': best_val_score,
                    'sac_stats': sac_strategy.get_statistics() if sac_strategy else None
                }, best_model_path)
            else:
                early_stop_counter += 1

            if args.early_stop and epoch >= args.warmup_epoch + 10:
                if early_stop_counter >= patience:
                    break

        fold_result = {
            'fold': fold + 1,
            'best_score': best_val_score,
            'best_epoch': best_epoch
        }
        all_folds_results.append(fold_result)

        print(f"Fold {fold + 1} completed - Best score: {best_val_score:.4f} at epoch {best_epoch}")

    # Save summary
    summary_path = os.path.join(exp_dir, 'cv_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'all_folds': all_folds_results,
            'avg_score': sum([r['best_score'] for r in all_folds_results]) / len(all_folds_results),
            'std_score': np.std([r['best_score'] for r in all_folds_results]),
            'sac_enabled': args.use_adaptive_aug,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)

    return exp_dir, all_folds_results


def evaluate(model, data_loader, criterion, device, val_set_name, epoch, task_type, dataset):
    """Evaluate model performance"""
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data
            else:
                inputs, labels = batch_data

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total if total > 0 else 0

    if dataset == "CirCor":
        circor_metrics = calculate_circor_metrics(all_labels, all_preds)
        overall_score = circor_metrics['w_acc']
        macro_sensitivity = circor_metrics['recall_present']
        macro_specificity = circor_metrics['recall_absent']

        print(
            f"{val_set_name} - Acc: {accuracy:.2f}%, W.acc: {circor_metrics['w_acc']:.4f}, UAR: {circor_metrics['uar']:.4f}")

        return {
            'accuracy': accuracy,
            'loss': running_loss / len(data_loader),
            'macro_sensitivity': macro_sensitivity,
            'macro_specificity': macro_specificity,
            'overall_score': overall_score,
            'w_acc': circor_metrics['w_acc'],
            'uar': circor_metrics['uar']
        }
    else:
        try:
            conf_matrix = confusion_matrix(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')

            if (dataset == "SPRSound" and task_type == "11") or (dataset == "ICBHI" and task_type == "binary"):
                if conf_matrix.shape == (2, 2):
                    tn, fp, fn, tp = conf_matrix.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
                    macro_sensitivity = sensitivity
                    macro_specificity = specificity
                else:
                    macro_sensitivity = macro_specificity = 0.5
            else:
                sensitivities = []
                specificities = []
                total_samples = np.sum(conf_matrix)

                for i in range(conf_matrix.shape[0]):
                    tp = conf_matrix[i, i]
                    row_sum = np.sum(conf_matrix[i, :])
                    fn = row_sum - tp
                    col_sum = np.sum(conf_matrix[:, i])
                    fp = col_sum - tp
                    tn = total_samples - (row_sum + col_sum - tp)

                    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

                    sensitivities.append(sensitivity)
                    specificities.append(specificity)

                macro_sensitivity = np.mean(sensitivities)
                macro_specificity = np.mean(specificities)

        except Exception:
            f1 = 0.0
            macro_sensitivity = 0.5
            macro_specificity = 0.5

        average_score = (macro_sensitivity + macro_specificity) / 2
        harmonic_score = 2 * macro_sensitivity * macro_specificity / (macro_sensitivity + macro_specificity + 1e-9)
        overall_score = (average_score + harmonic_score) / 2

        print(
            f"{val_set_name} - Acc: {accuracy:.2f}%, Sens: {macro_sensitivity:.4f}, Spec: {macro_specificity:.4f}, Score: {overall_score:.4f}")

        return {
            'accuracy': accuracy,
            'loss': running_loss / len(data_loader),
            'macro_sensitivity': macro_sensitivity,
            'macro_specificity': macro_specificity,
            'overall_score': overall_score,
            'f1': f1
        }


def create_experiment_dir(args):
    """Create experiment directory"""
    base_dir = './experiments'
    os.makedirs(base_dir, exist_ok=True)

    if args.dataset == "CirCor":
        aug_suffix = f"TwoPhaseSAC_P{args.phase2_trigger_patience}_T{args.phase2_trigger_threshold}" if args.use_adaptive_aug else "NoAug"
        exp_identifier = f"CirCor_DigiScope_HeartMurmur_EfficientNet-B4_{args.feature_type}_{aug_suffix}"
    else:
        aug_suffix = f"TwoPhaseSAC_P{args.phase2_trigger_patience}_T{args.phase2_trigger_threshold}" if args.use_adaptive_aug else "NoAug"
        exp_identifier = f"{args.dataset}_task{args.task_type}_EfficientNet-B4_{args.feature_type}_{aug_suffix}"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{exp_identifier}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

    return exp_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Adaptive Augmentation Strategy for Lung Sound Classification with Two-Phase SAC')

    # Basic parameters
    parser.add_argument("--dataset", type=str, default="SPRSound", choices=["SPRSound", "ICBHI", "CirCor"])
    parser.add_argument("--circor_val_ratio", type=float, default=0.2)
    parser.add_argument("--task_type", type=str, default="22")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--gpu_id", type=int, default=0)

    # Feature and model parameters
    parser.add_argument("--feature_type", type=str, default='log-mel', choices=['MFCC', 'log-mel', 'mel', 'STFT'])

    # SAC parameters
    parser.add_argument("--use_adaptive_aug", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--n_magnitude_levels", type=int, default=5, help="Number of magnitude levels per operation")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--sac_lr_actor", type=float, default=3e-4)
    parser.add_argument("--sac_lr_critic", type=float, default=3e-4)
    parser.add_argument("--sac_buffer_size", type=int, default=10000)
    parser.add_argument("--sac_tau", type=float, default=0.005)

    # Two-phase specific parameters
    parser.add_argument("--phase2_trigger_patience", type=int, default=20, help="Phase 2 trigger patience")
    parser.add_argument("--phase2_trigger_threshold", type=float, default=0.01, help="Phase 2 trigger threshold")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_epoch", type=int, default=25)
    parser.add_argument("--warmup_lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    args = parser.parse_args()

    # Setup CUDA device
    if torch.cuda.is_available():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device_id = 0
        else:
            device_id = args.gpu_id
        global_device = torch.device(f"cuda:{device_id}")
    else:
        global_device = torch.device("cpu")

    print(f"Using device: {global_device}")

    if args.use_adaptive_aug:
        print(f"Two-Phase SAC Configuration:")
        print(f"  - gamma={args.gamma}")
        print(f"  - Phase 2 trigger patience={args.phase2_trigger_patience}")
        print(f"  - Phase 2 trigger threshold={args.phase2_trigger_threshold}")
        print(f"  - Phase 1: Batch-level adaptation")
        print(f"  - Phase 2: Sample-level adaptation")

    try:
        exp_dir, cv_results = train(args, global_device)
        print(f"\nTraining completed! Results saved in: {exp_dir}")
        return exp_dir, cv_results

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()