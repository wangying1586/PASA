import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from sklearn.metrics import confusion_matrix
import copy
from typing import Dict, List, Tuple, Optional


def calculate_balanced_accuracy(y_true, y_pred, n_classes):
    """Calculate balanced accuracy across all classes"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    recalls = []
    for i in range(n_classes):
        tp = cm[i, i]
        total_true_samples = cm[i, :].sum()
        recall = tp / total_true_samples if total_true_samples > 0 else 0.0
        recalls.append(recall)

    return np.mean(recalls)


class SampleConfidenceTracker:
    """Track sample confidence and difficulty based on correct predictions only"""

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.sample_stats = {}
        self.update_count = 0
        self.correct_predictions = 0
        self.total_predictions = 0

    def update_sample_stats(self, sample_indices: List[int], predictions: torch.Tensor,
                            true_labels: torch.Tensor):
        """Update statistics only for correctly predicted samples"""
        if not sample_indices or len(sample_indices) == 0:
            return

        batch_size = min(len(sample_indices), predictions.shape[0], true_labels.shape[0])

        for i in range(batch_size):
            try:
                sample_idx = sample_indices[i]
                pred_logits = predictions[i] if predictions.dim() > 1 else predictions
                true_label = true_labels[i].item()

                probs = F.softmax(pred_logits, dim=0)
                max_prob, predicted_class = torch.max(probs, dim=0)
                predicted_class = predicted_class.item()
                max_prob = max_prob.item()

                self.total_predictions += 1
                is_correct = (predicted_class == true_label)
                if is_correct:
                    self.correct_predictions += 1

                if is_correct:
                    confidence = max_prob
                    if len(probs) >= 2:
                        top2_probs, _ = torch.topk(probs, min(2, len(probs)))
                        top1_top2_diff = (top2_probs[0] - top2_probs[1]).item() if len(top2_probs) == 2 else confidence
                    else:
                        top1_top2_diff = confidence

                    if sample_idx not in self.sample_stats:
                        self.sample_stats[sample_idx] = {
                            'confidence': confidence,
                            'top1_top2_diff': top1_top2_diff,
                            'update_count': 1,
                            'correct_predictions': 1,
                            'total_attempts': 1,
                            'accuracy': 1.0
                        }
                    else:
                        old_stats = self.sample_stats[sample_idx]
                        old_stats['total_attempts'] += 1
                        old_stats['correct_predictions'] += 1
                        old_stats['accuracy'] = old_stats['correct_predictions'] / old_stats['total_attempts']

                        self.sample_stats[sample_idx] = {
                            'confidence': self.momentum * old_stats['confidence'] + (1 - self.momentum) * confidence,
                            'top1_top2_diff': self.momentum * old_stats['top1_top2_diff'] + (
                                        1 - self.momentum) * top1_top2_diff,
                            'update_count': old_stats['update_count'] + 1,
                            'correct_predictions': old_stats['correct_predictions'],
                            'total_attempts': old_stats['total_attempts'],
                            'accuracy': old_stats['accuracy']
                        }
                else:
                    if sample_idx in self.sample_stats:
                        old_stats = self.sample_stats[sample_idx]
                        old_stats['total_attempts'] += 1
                        old_stats['accuracy'] = old_stats['correct_predictions'] / old_stats['total_attempts']
                    else:
                        self.sample_stats[sample_idx] = {
                            'confidence': 0.5,
                            'top1_top2_diff': 0.1,
                            'update_count': 0,
                            'correct_predictions': 0,
                            'total_attempts': 1,
                            'accuracy': 0.0
                        }

                self.update_count += 1
            except Exception:
                continue

    def get_sample_difficulty(self, sample_idx: int) -> str:
        """Classify sample difficulty: easy, medium, or hard"""
        if sample_idx not in self.sample_stats:
            return 'medium'

        stats = self.sample_stats[sample_idx]
        confidence = stats['confidence']
        diff = stats['top1_top2_diff']
        accuracy = stats.get('accuracy', 0.5)
        correct_count = stats.get('correct_predictions', 0)

        if correct_count == 0 or accuracy < 0.3:
            return 'hard'
        elif confidence > 0.8 and diff > 0.3 and accuracy > 0.7:
            return 'easy'
        else:
            return 'medium'

    def get_all_difficulties(self) -> Dict[str, int]:
        """Get difficulty distribution across all samples"""
        difficulties = {'easy': 0, 'medium': 0, 'hard': 0}
        for sample_idx in self.sample_stats:
            difficulty = self.get_sample_difficulty(sample_idx)
            difficulties[difficulty] += 1
        return difficulties

    def get_confidence_statistics(self) -> Dict[str, float]:
        """Get overall confidence statistics"""
        if not self.sample_stats:
            return {
                'total_samples': 0,
                'samples_with_correct_predictions': 0,
                'overall_accuracy': 0.0,
                'avg_confidence': 0.5,
                'avg_accuracy': 0.0
            }

        samples_with_correct = [stats for stats in self.sample_stats.values()
                                if stats.get('correct_predictions', 0) > 0]

        return {
            'total_samples': len(self.sample_stats),
            'samples_with_correct_predictions': len(samples_with_correct),
            'overall_accuracy': self.correct_predictions / max(1, self.total_predictions),
            'avg_confidence': np.mean([s['confidence'] for s in samples_with_correct]) if samples_with_correct else 0.5,
            'avg_accuracy': np.mean([s['accuracy'] for s in self.sample_stats.values()]),
            'confidence_std': np.std([s['confidence'] for s in samples_with_correct]) if samples_with_correct else 0.0
        }


class AdaptiveAlphaStrategy:
    """Adaptive alpha strategy with difficulty-specific exploration strength"""

    def __init__(self, device, alpha_lr=3e-4):
        self.device = device

        # Different alpha values for different difficulty levels
        self.difficulty_log_alphas = {
            'easy': torch.tensor(np.log(1.0), requires_grad=True, device=device),
            'medium': torch.tensor(np.log(0.3), requires_grad=True, device=device),
        }

        self.alpha_optimizers = {}
        for difficulty in ['easy', 'medium']:
            self.alpha_optimizers[difficulty] = optim.Adam(
                [self.difficulty_log_alphas[difficulty]], lr=alpha_lr)

        self.target_entropies = {
            'easy': -np.log(1.0 / 9) * 1.2,
            'medium': -np.log(1.0 / 9) * 0.6,
        }

        self.alpha_update_counts = defaultdict(int)
        self.alpha_losses = defaultdict(list)

    def get_alpha_for_difficulty(self, difficulty: str) -> torch.Tensor:
        """Get alpha value for specified difficulty"""
        if difficulty == 'hard':
            return torch.tensor(0.0, device=self.device)
        if difficulty not in self.difficulty_log_alphas:
            difficulty = 'medium'
        return self.difficulty_log_alphas[difficulty].exp()

    def update_alpha_for_difficulty(self, difficulty: str, log_prob: torch.Tensor):
        """Update alpha for specified difficulty"""
        if difficulty == 'hard':
            return 0.0

        if difficulty not in self.difficulty_log_alphas:
            difficulty = 'medium'

        try:
            target_entropy = self.target_entropies[difficulty]
            alpha_loss = -(self.difficulty_log_alphas[difficulty] *
                           (log_prob + target_entropy).detach()).mean()

            self.alpha_optimizers[difficulty].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[difficulty].step()

            self.alpha_update_counts[difficulty] += 1
            self.alpha_losses[difficulty].append(alpha_loss.item())

            return alpha_loss.item()
        except Exception:
            return 0.0

    def get_all_alphas(self) -> Dict[str, float]:
        """Get all current alpha values"""
        alphas = {}
        for diff, log_alpha in self.difficulty_log_alphas.items():
            alphas[diff] = log_alpha.exp().item()
        alphas['hard'] = 0.0
        return alphas


class ExplorationScheduler:
    """Lightweight exploration scheduler using noise injection"""

    def __init__(self):
        self.exploration_decay = 0.98
        self.min_exploration = 0.02
        self.noise_injection_interval = 15

    def get_exploration_boost(self, epoch: int, total_epochs: int) -> float:
        """Calculate exploration boost based on training progress"""
        progress = epoch / total_epochs
        if progress > 0.8:
            return 0.05
        elif progress > 0.6:
            return 0.02
        else:
            return 0.0

    def add_exploration_noise(self, action_logits: torch.Tensor, epoch: int) -> torch.Tensor:
        """Add exploration noise to action logits"""
        if epoch % self.noise_injection_interval == 0:
            noise_scale = max(self.min_exploration,
                              0.1 * (self.exploration_decay ** (epoch // self.noise_injection_interval)))
            exploration_noise = torch.randn_like(action_logits) * noise_scale
            return action_logits + exploration_noise
        return action_logits

    def should_encourage_exploration(self, epoch: int) -> bool:
        """Check if extra exploration is needed"""
        return (epoch % 50 == 0 and epoch > 100)

    def get_exploration_bonus(self, epoch: int, total_epochs: int) -> float:
        """Get exploration reward bonus"""
        if self.should_encourage_exploration(epoch):
            progress = epoch / total_epochs
            if progress > 0.7:
                return 0.02
            else:
                return 0.01
        return 0.0


class TwoPhaseSACStrategy:
    """
    Two-phase SAC adaptive augmentation strategy with HBS

    Phase 1: Batch-level adaptation
    Phase 2: Sample-level adaptation with fixed hard sample strategy

    Key Components (aligned with the 5-step framework):
    1. Feature extraction and state construction
    2. Action sampling via SAC network
    3. HBS execution strategy (difficulty-based augmentation selection)
    4. Performance evaluation and reward computation
    5. Joint optimization of augmentation policies and diagnostic models
    """

    def __init__(self, n_classes, feature_dim, aug_operations, device,
                 buffer_capacity=10000, gamma=0.95, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 validation_set=None, n_magnitude_levels=5,
                 phase2_trigger_patience=20, phase2_trigger_threshold=0.02,
                 **kwargs):

        self.device = device
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.aug_operations = aug_operations
        self.n_magnitude_levels = n_magnitude_levels

        # SAC parameters
        self.gamma = gamma
        self.tau = tau

        # Two-phase parameters
        self.current_phase = 1
        self.phase2_trigger_patience = phase2_trigger_patience
        self.phase2_trigger_threshold = phase2_trigger_threshold
        self.performance_history = deque(maxlen=phase2_trigger_patience + 5)
        self.phase2_triggered = False

        # Sample confidence tracking
        self.confidence_tracker = SampleConfidenceTracker()

        # Adaptive alpha strategy
        self.adaptive_alpha = AdaptiveAlphaStrategy(device, alpha_lr)

        # Exploration scheduler
        self.exploration_scheduler = ExplorationScheduler()

        # Network dimensions
        self.state_dim = feature_dim * 2 + n_classes
        self.action_dim = len(aug_operations) + 1 + n_magnitude_levels

        # Initialize magnitude ranges
        self.magnitude_ranges = self._init_magnitude_ranges()

        # Build networks
        self._build_networks()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.validation_set = validation_set

        # Statistics
        self.statistics = defaultdict(list)
        self.operation_stats = defaultdict(int)
        self.magnitude_stats = defaultdict(lambda: defaultdict(int))
        self.training_step = 0

        self.phase2_stats = {
            'easy_samples': 0,
            'medium_samples': 0,
            'hard_samples': 0,
            'phase2_epochs': 0,
            'phase2_start_epoch': -1
        }

        self._last_phase2_epoch = -1
        self._last_validation_time = 0
        self._global_sample_counter = 0

    def _init_magnitude_ranges(self):
        """Initialize magnitude ranges for each augmentation operation"""
        ranges = {}
        configs = {
            "time_mask": (0.05, 0.3),
            "frequency_mask": (0.05, 0.3),
            "noise_injection": (0.01, 0.1),
            "random_quantization": (0.1, 0.5),
            "spectral_contrast": (0.1, 0.4),
            "harmonic_perturbation": (0.05, 0.2),
            "breathing_cycle_stretch": (0.05, 0.25),
            "low_freq_emphasis": (0.1, 0.4)
        }

        for operation in self.aug_operations:
            if operation in configs:
                min_val, max_val = configs[operation]
            else:
                min_val, max_val = 0.1, 0.3

            magnitudes = np.linspace(min_val, max_val, self.n_magnitude_levels)
            ranges[operation] = magnitudes.tolist()

        ranges["no_augmentation"] = [0.0] * self.n_magnitude_levels
        return ranges

    def _build_networks(self):
        """Build SAC networks"""
        self.actor = AdaptiveActorNetwork(
            self.state_dim, len(self.aug_operations) + 1, self.n_magnitude_levels
        ).to(self.device)

        max_action_dim = (len(self.aug_operations) + 1) + self.n_magnitude_levels
        self.critic1 = CriticNetwork(self.state_dim, max_action_dim).to(self.device)
        self.critic2 = CriticNetwork(self.state_dim, max_action_dim).to(self.device)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

    @property
    def alpha(self):
        """Default alpha value for compatibility"""
        return self.adaptive_alpha.get_alpha_for_difficulty('medium')

    def generate_stable_sample_indices(self, batch_size: int, sample_indices: List[int] = None) -> List[int]:
        """Generate stable sample indices"""
        if sample_indices is not None and len(sample_indices) == batch_size:
            return sample_indices

        stable_indices = []
        for i in range(batch_size):
            stable_indices.append(self._global_sample_counter + i)

        self._global_sample_counter += batch_size
        return stable_indices

    def calculate_ba_reward(self, model, original_x, augmented_x, y_true):
        """
        Step 4: Performance evaluation and reward computation
        Calculate reward based on Balanced Accuracy improvement
        """
        try:
            model.eval()
            with torch.no_grad():
                orig_outputs = model(original_x)
                orig_preds = torch.argmax(orig_outputs, dim=1)
                orig_ba = calculate_balanced_accuracy(
                    y_true.cpu().numpy(),
                    orig_preds.cpu().numpy(),
                    self.n_classes
                )

                aug_outputs = model(augmented_x)
                aug_preds = torch.argmax(aug_outputs, dim=1)
                aug_ba = calculate_balanced_accuracy(
                    y_true.cpu().numpy(),
                    aug_preds.cpu().numpy(),
                    self.n_classes
                )

                ba_reward = aug_ba - orig_ba

            model.train()
            return ba_reward
        except Exception:
            model.train()
            return 0.0

    def get_difficulty_action_mask(self, difficulty: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 3: HBS execution strategy
        Get action masks based on sample difficulty
        """
        n_operations = len(self.aug_operations) + 1
        operation_mask = torch.ones(n_operations, device=self.device)
        magnitude_mask = torch.ones(self.n_magnitude_levels, device=self.device)

        if difficulty == 'easy':
            pass  # Full search space
        elif difficulty == 'medium':
            magnitude_mask[3:] = 0  # Limit to lighter magnitudes
        elif difficulty == 'hard':
            operation_mask[:-1] = 0  # Only no_augmentation
            magnitude_mask[1:] = 0

        return operation_mask, magnitude_mask

    def select_action_phase1(self, state, epoch=0, deterministic=False):
        """
        Step 2: Action sampling via SAC network (Phase 1)
        Batch-level action selection
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        n_operations = len(self.aug_operations) + 1
        n_magnitudes = self.n_magnitude_levels

        if deterministic:
            operation_logits, magnitude_logits = self.actor(state, n_operations, n_magnitudes)
            operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
            operation_idx = torch.argmax(F.softmax(operation_logits, dim=-1), dim=-1)
            magnitude_idx = torch.argmax(F.softmax(magnitude_logits, dim=-1), dim=-1)
        else:
            operation_logits, magnitude_logits, _ = self.actor.sample(state, n_operations, n_magnitudes)
            operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
            magnitude_logits = self.exploration_scheduler.add_exploration_noise(magnitude_logits, epoch)

            operation_dist = torch.distributions.Categorical(logits=operation_logits)
            magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)
            operation_idx = operation_dist.sample()
            magnitude_idx = magnitude_dist.sample()

        action = torch.zeros(n_operations + n_magnitudes).to(self.device)
        action[operation_idx] = 1.0
        action[n_operations + magnitude_idx] = 1.0

        return action.cpu().numpy(), operation_idx.item(), magnitude_idx.item()

    def select_action_phase2_adaptive(self, state, sample_indices, epoch=0, deterministic=False):
        """
        Step 2 & 3: Action sampling via SAC network (Phase 2) + HBS execution
        Sample-level adaptive action selection with difficulty-based strategy
        """
        batch_actions = []
        batch_operation_indices = []
        batch_magnitude_indices = []
        batch_difficulties = []
        batch_log_probs = []

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        for sample_idx in sample_indices:
            difficulty = self.confidence_tracker.get_sample_difficulty(sample_idx)
            batch_difficulties.append(difficulty)

            # Hard samples: fixed no_augmentation strategy
            if difficulty == 'hard':
                operation_name = 'no_augmentation'
                magnitude_idx_item = 0
                operation_idx_item = len(self.aug_operations)

                batch_actions.append((operation_name, magnitude_idx_item))
                batch_operation_indices.append(operation_idx_item)
                batch_magnitude_indices.append(magnitude_idx_item)
                batch_log_probs.append(torch.tensor(0.0, device=self.device))
                continue

            # Easy and medium samples: SAC-based selection with difficulty masks
            operation_mask, magnitude_mask = self.get_difficulty_action_mask(difficulty)
            current_alpha = self.adaptive_alpha.get_alpha_for_difficulty(difficulty)

            n_operations = operation_mask.shape[0]
            n_magnitudes = magnitude_mask.shape[0]

            if deterministic:
                operation_logits, magnitude_logits = self.actor(state_tensor, n_operations, n_magnitudes)
                operation_logits = operation_logits / (current_alpha + 1e-8)
                magnitude_logits = magnitude_logits / (current_alpha + 1e-8)

                if difficulty == 'easy':
                    operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
                    magnitude_logits = self.exploration_scheduler.add_exploration_noise(magnitude_logits, epoch)

                operation_logits = operation_logits + (operation_mask - 1) * 1e9
                magnitude_logits = magnitude_logits + (magnitude_mask - 1) * 1e9

                operation_idx = torch.argmax(operation_logits, dim=-1)
                magnitude_idx = torch.argmax(magnitude_logits, dim=-1)

                op_dist = torch.distributions.Categorical(logits=operation_logits)
                mag_dist = torch.distributions.Categorical(logits=magnitude_logits)
                log_prob = op_dist.log_prob(operation_idx) + mag_dist.log_prob(magnitude_idx)
            else:
                operation_logits, magnitude_logits, _ = self.actor.sample(state_tensor, n_operations, n_magnitudes)
                operation_logits = operation_logits / (current_alpha + 1e-8)
                magnitude_logits = magnitude_logits / (current_alpha + 1e-8)

                if difficulty == 'easy':
                    operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
                    magnitude_logits = self.exploration_scheduler.add_exploration_noise(magnitude_logits, epoch)
                    operation_logits += torch.randn_like(operation_logits) * 0.1
                    magnitude_logits += torch.randn_like(magnitude_logits) * 0.1

                operation_logits = operation_logits + (operation_mask - 1) * 1e9
                magnitude_logits = magnitude_logits + (magnitude_mask - 1) * 1e9

                operation_dist = torch.distributions.Categorical(logits=operation_logits)
                magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)

                operation_idx = operation_dist.sample()
                magnitude_idx = magnitude_dist.sample()
                log_prob = operation_dist.log_prob(operation_idx) + magnitude_dist.log_prob(magnitude_idx)

            batch_log_probs.append(log_prob)

            operation_idx_item = operation_idx.item()
            magnitude_idx_item = magnitude_idx.item()

            if operation_idx_item >= len(self.aug_operations):
                operation_name = 'no_augmentation'
            else:
                operation_name = self.aug_operations[operation_idx_item]

            batch_actions.append((operation_name, magnitude_idx_item))
            batch_operation_indices.append(operation_idx_item)
            batch_magnitude_indices.append(magnitude_idx_item)

        avg_operation_idx = int(np.mean(batch_operation_indices))
        avg_magnitude_idx = int(np.mean(batch_magnitude_indices))

        return batch_actions, avg_operation_idx, avg_magnitude_idx, batch_difficulties, batch_log_probs

    def apply_sample_level_augmentation(self, x, y, batch_actions, sample_indices):
        """
        Step 3: HBS execution strategy
        Apply sample-level augmentation based on selected actions
        """
        augmented_x = x.clone()
        augmented_y = y.clone()

        for i, ((operation_name, magnitude_idx), sample_idx) in enumerate(zip(batch_actions, sample_indices)):
            try:
                if i >= x.shape[0]:
                    break

                single_x = x[i:i + 1]
                single_y = y[i:i + 1]

                if operation_name == 'no_augmentation':
                    continue
                else:
                    if operation_name in self.aug_operations:
                        operation_idx = self.aug_operations.index(operation_name)
                        aug_x, aug_y = self.apply_augmentation(single_x, single_y, operation_idx, magnitude_idx)
                        augmented_x[i:i + 1] = aug_x
                        augmented_y[i:i + 1] = aug_y
            except Exception:
                continue

        return augmented_x, augmented_y

    def check_phase2_trigger(self, current_performance: float) -> bool:
        """Check if phase 2 should be triggered based on performance plateau"""
        if self.phase2_triggered:
            return True

        self.performance_history.append(current_performance)

        if len(self.performance_history) <= self.phase2_trigger_patience:
            return False

        historical_performances = list(self.performance_history)[:-self.phase2_trigger_patience]
        if len(historical_performances) == 0:
            return False

        historical_best = max(historical_performances)
        recent_performances = list(self.performance_history)[-self.phase2_trigger_patience:]

        no_improvement_count = 0
        for perf in recent_performances:
            if perf <= historical_best + self.phase2_trigger_threshold:
                no_improvement_count += 1

        if no_improvement_count >= self.phase2_trigger_patience:
            self.current_phase = 2
            self.phase2_triggered = True
            return True

        return False

    def calculate_validation_performance(self, model, criterion):
        """Calculate validation performance for phase transition"""
        if self.validation_set is None:
            return 0.5

        import time
        current_time = time.time()
        if hasattr(self, '_last_val_time') and current_time - self._last_val_time < 10.0:
            return getattr(self, '_cached_val_performance', 0.5)

        self._last_val_time = current_time

        was_training = model.training
        model.eval()

        all_preds = []
        all_labels = []

        try:
            with torch.no_grad():
                sample_count = 0
                for batch_data in self.validation_set:
                    if sample_count > 300:
                        break

                    if len(batch_data) == 3:
                        inputs, labels, _ = batch_data
                    else:
                        inputs, labels = batch_data

                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    sample_count += len(labels)

            if len(all_preds) > 0 and len(all_labels) > 0:
                val_performance = calculate_balanced_accuracy(all_labels, all_preds, self.n_classes)
            else:
                val_performance = 0.5

            self._cached_val_performance = val_performance
            return val_performance
        except Exception:
            return 0.5
        finally:
            if was_training:
                model.train()

    def update_sample_confidence(self, model, batch_x, batch_y, sample_indices):
        """Update sample confidence statistics"""
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(batch_x)
                self.confidence_tracker.update_sample_stats(sample_indices, outputs, batch_y)
            model.train()
        except Exception:
            pass

    def check_epoch_end_phase_transition(self, model, criterion, epoch):
        """Check phase transition at epoch end"""
        if not self.phase2_triggered:
            val_performance = self.calculate_validation_performance(model, criterion)

            if not hasattr(self.statistics, 'val_performance'):
                self.statistics['val_performance'] = []
            self.statistics['val_performance'].append(val_performance)

            phase_changed = self.check_phase2_trigger(val_performance)
            if phase_changed and self.phase2_stats['phase2_start_epoch'] == -1:
                self.phase2_stats['phase2_start_epoch'] = epoch
            return phase_changed
        else:
            val_performance = self.calculate_validation_performance(model, criterion)
            if not hasattr(self.statistics, 'val_performance'):
                self.statistics['val_performance'] = []
            self.statistics['val_performance'].append(val_performance)

        return False

    def train_step(self, model, criterion, x, y, epoch, total_epochs, sample_indices=None):
        """
        Step 5: Joint optimization of augmentation policies and diagnostic models
        Main training step integrating all 5 framework components
        """
        # Generate stable sample indices
        sample_indices = self.generate_stable_sample_indices(x.shape[0], sample_indices)

        # Update sample confidence
        self.update_sample_confidence(model, x, y, sample_indices)

        # Step 1: Feature extraction and state construction
        initial_state = self.get_state(model, x, y)

        # Steps 2-3: Action sampling and HBS execution
        if self.current_phase == 1:
            # Phase 1: Batch-level augmentation
            action, operation_idx, magnitude_idx = self.select_action_phase1(initial_state.cpu().numpy(), epoch)
            augmented_x, augmented_y = self.apply_augmentation(x, y, operation_idx, magnitude_idx)

            # Step 4: Performance evaluation
            batch_ba_reward = self.calculate_ba_reward(model, x, augmented_x, y)
            avg_reward = batch_ba_reward
        else:
            # Phase 2: Sample-level augmentation with HBS
            batch_actions, operation_idx, magnitude_idx, batch_difficulties, batch_log_probs = \
                self.select_action_phase2_adaptive(initial_state.cpu().numpy(), sample_indices, epoch)

            augmented_x, augmented_y = self.apply_sample_level_augmentation(x, y, batch_actions, sample_indices)

            # Step 4: Performance evaluation (batch-level reward)
            batch_ba_reward = self.calculate_ba_reward(model, x, augmented_x, y)
            avg_reward = batch_ba_reward

            # Update adaptive alpha for non-hard samples
            for difficulty, log_prob in zip(batch_difficulties, batch_log_probs):
                if difficulty != 'hard':
                    self.adaptive_alpha.update_alpha_for_difficulty(difficulty, log_prob)

        # Step 1: Get augmented state
        next_state = self.get_augmented_state(model, x, y, augmented_x)

        # Update statistics
        operation_name = self.aug_operations[operation_idx] if operation_idx < len(
            self.aug_operations) else "no_augmentation"
        self.operation_stats[operation_name] += 1
        self.magnitude_stats[operation_name][magnitude_idx] += 1

        # Step 5: Train diagnostic model
        model.train()
        outputs = model(augmented_x)
        loss = criterion(outputs, augmented_y)

        # Store experience in replay buffer
        if self.current_phase == 1:
            action_vector = torch.zeros(len(self.aug_operations) + 1 + self.n_magnitude_levels)
            action_vector[operation_idx] = 1.0
            action_vector[len(self.aug_operations) + 1 + magnitude_idx] = 1.0

            self.replay_buffer.push(
                initial_state.cpu().numpy(),
                action_vector.numpy(),
                batch_ba_reward,
                next_state.cpu().numpy(),
                False
            )
        else:
            for i, (operation_name, magnitude_idx_sample) in enumerate(batch_actions):
                try:
                    if operation_name == 'no_augmentation':
                        op_idx = len(self.aug_operations)
                    else:
                        op_idx = self.aug_operations.index(operation_name)

                    action_vector = torch.zeros(len(self.aug_operations) + 1 + self.n_magnitude_levels)
                    action_vector[op_idx] = 1.0
                    action_vector[len(self.aug_operations) + 1 + magnitude_idx_sample] = 1.0

                    self.replay_buffer.push(
                        initial_state.cpu().numpy(),
                        action_vector.numpy(),
                        batch_ba_reward,
                        next_state.cpu().numpy(),
                        False
                    )
                except Exception:
                    pass

        # Step 5: Update SAC networks (policy optimization)
        sac_losses = {}
        if len(self.replay_buffer) > 256:
            sac_losses = self.update_networks()

        # Update statistics
        self.statistics['rewards'].append(avg_reward)

        if 'ba_improvement' not in self.statistics:
            self.statistics['ba_improvement'] = []
        self.statistics['ba_improvement'].append(batch_ba_reward)

        self.training_step += 1

        # Update phase 2 statistics
        if self.current_phase == 2:
            for difficulty in batch_difficulties:
                if difficulty == 'easy':
                    self.phase2_stats['easy_samples'] += 1
                elif difficulty == 'medium':
                    self.phase2_stats['medium_samples'] += 1
                elif difficulty == 'hard':
                    self.phase2_stats['hard_samples'] += 1

            if self._last_phase2_epoch != epoch:
                self.phase2_stats['phase2_epochs'] = epoch
                self._last_phase2_epoch = epoch

        return augmented_x, augmented_y, avg_reward, sac_losses

    def extract_features(self, model, x):
        """
        Step 1: Feature extraction
        Extract features from model for state construction
        """
        if not x.is_cuda:
            x = x.to(self.device, non_blocking=True)

        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                elif hasattr(model, 'base_model'):
                    original_fc = model.base_model._fc
                    model.base_model._fc = nn.Identity()
                    features = model.base_model(x)
                    model.base_model._fc = original_fc
                else:
                    features = model(x)
        except:
            batch_size = x.shape[0]
            features = torch.randn(batch_size, self.feature_dim).to(self.device)
        finally:
            if was_training:
                model.train()

        return features

    def get_class_distribution(self, labels):
        """Calculate class distribution for state construction"""
        batch_size = labels.shape[0]
        class_counts = torch.zeros(self.n_classes, device=self.device)
        for i in range(self.n_classes):
            class_counts[i] = (labels == i).float().sum()
        return class_counts / batch_size

    def get_state(self, model, batch_x, batch_y):
        """
        Step 1: State construction
        Construct state representation from features and class distribution
        """
        original_features = self.extract_features(model, batch_x)
        batch_original_features = torch.mean(original_features, dim=0)
        batch_augmented_features = batch_original_features.clone()

        class_distribution = self.get_class_distribution(batch_y)

        state = torch.cat([
            batch_original_features,
            batch_augmented_features,
            class_distribution
        ])

        return state

    def get_augmented_state(self, model, batch_x, batch_y, augmented_x):
        """
        Step 1: Augmented state construction
        Get state after augmentation
        """
        original_features = self.extract_features(model, batch_x)
        augmented_features = self.extract_features(model, augmented_x)

        batch_original_features = torch.mean(original_features, dim=0)
        batch_augmented_features = torch.mean(augmented_features, dim=0)

        class_distribution = self.get_class_distribution(batch_y)

        state = torch.cat([
            batch_original_features,
            batch_augmented_features,
            class_distribution
        ])

        return state

    def apply_augmentation(self, x, y, operation_idx, magnitude_idx):
        """Apply augmentation operation"""
        if not x.is_cuda:
            x = x.to(self.device, non_blocking=True)
        if not y.is_cuda:
            y = y.to(self.device, non_blocking=True)

        if operation_idx >= len(self.aug_operations):
            return x, y

        operation_name = self.aug_operations[operation_idx]
        magnitude = self.get_magnitude_value(operation_idx, magnitude_idx)

        try:
            if operation_name == "time_mask":
                return self._apply_time_mask(x, y, magnitude)
            elif operation_name == "frequency_mask":
                return self._apply_frequency_mask(x, y, magnitude)
            elif operation_name == "noise_injection":
                return self._apply_noise_injection(x, y, magnitude)
            elif operation_name == "random_quantization":
                return self._apply_random_quantization(x, y, magnitude)
            elif operation_name == "spectral_contrast":
                return self._apply_spectral_contrast(x, y, magnitude)
            elif operation_name == "harmonic_perturbation":
                return self._apply_harmonic_perturbation(x, y, magnitude)
            elif operation_name == "breathing_cycle_stretch":
                return self._apply_breathing_cycle_stretch(x, y, magnitude)
            elif operation_name == "low_freq_emphasis":
                return self._apply_low_freq_emphasis(x, y, magnitude)
            else:
                return x, y
        except Exception:
            return x, y

    def get_magnitude_value(self, operation_idx, magnitude_idx):
        """Get actual magnitude value"""
        if operation_idx >= len(self.aug_operations):
            return 0.0

        operation_name = self.aug_operations[operation_idx]
        if operation_name in self.magnitude_ranges:
            magnitude_values = self.magnitude_ranges[operation_name]
            if magnitude_idx < len(magnitude_values):
                return magnitude_values[magnitude_idx]

        return 0.1

    # Augmentation operation implementations
    def _apply_time_mask(self, x, y, magnitude):
        """Time masking"""
        _, _, _, width = x.shape
        mask_width = int(width * magnitude)
        if mask_width == 0:
            return x, y
        start_pos = torch.randint(0, width - mask_width + 1, (1,)).item()
        augmented_x = x.clone()
        augmented_x[:, :, :, start_pos:start_pos + mask_width] = 0
        return augmented_x, y

    def _apply_frequency_mask(self, x, y, magnitude):
        """Frequency masking"""
        _, _, height, _ = x.shape
        mask_height = int(height * magnitude)
        if mask_height == 0:
            return x, y
        start_pos = torch.randint(0, height - mask_height + 1, (1,)).item()
        augmented_x = x.clone()
        augmented_x[:, :, start_pos:start_pos + mask_height, :] = 0
        return augmented_x, y

    def _apply_noise_injection(self, x, y, magnitude):
        """Noise injection"""
        noise = torch.randn_like(x) * magnitude
        return x + noise, y

    def _apply_random_quantization(self, x, y, magnitude):
        """Random quantization"""
        levels = max(2, int(256 * (1 - magnitude)))
        x_min, x_max = x.min(), x.max()
        x_scaled = (x - x_min) / (x_max - x_min + 1e-8)
        x_quantized = torch.round(x_scaled * (levels - 1)) / (levels - 1)
        return x_quantized * (x_max - x_min) + x_min, y

    def _apply_spectral_contrast(self, x, y, magnitude):
        """Spectral contrast"""
        contrast_factor = 1.0 + magnitude
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - x_mean) * contrast_factor + x_mean, y

    def _apply_harmonic_perturbation(self, x, y, magnitude):
        """Harmonic perturbation"""
        batch_size, channels, height, width = x.shape
        freq = torch.rand(1).item() * 10
        phase = torch.rand(1).item() * 2 * np.pi

        t = torch.linspace(0, 1, width).to(x.device)
        t = t.view(1, 1, 1, -1).expand(batch_size, channels, height, -1)

        harmonic = torch.sin(2 * np.pi * freq * t + phase) * magnitude
        return x + harmonic, y

    def _apply_breathing_cycle_stretch(self, x, y, magnitude):
        """Breathing cycle stretch"""
        stretch_factor = 1.0 + (torch.rand(1).item() - 0.5) * magnitude * 2

        if abs(stretch_factor - 1.0) < 0.01:
            return x, y

        try:
            augmented_x = F.interpolate(x, scale_factor=(1.0, stretch_factor), mode='bilinear', align_corners=False)

            _, _, _, new_width = augmented_x.shape
            _, _, _, orig_width = x.shape

            if new_width > orig_width:
                start = (new_width - orig_width) // 2
                augmented_x = augmented_x[:, :, :, start:start + orig_width]
            elif new_width < orig_width:
                pad_width = orig_width - new_width
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                augmented_x = F.pad(augmented_x, (pad_left, pad_right), mode='constant', value=0)

            return augmented_x, y
        except:
            return x, y

    def _apply_low_freq_emphasis(self, x, y, magnitude):
        """Low frequency emphasis"""
        _, _, height, _ = x.shape
        freq_weights = torch.ones(height, device=x.device)
        low_freq_end = int(height * 0.3)
        freq_weights[:low_freq_end] = 1.0 + magnitude
        freq_weights = freq_weights.view(1, 1, -1, 1)
        return x * freq_weights, y

    def update_networks(self, batch_size=256):
        """
        Step 5: Joint optimization - SAC network update
        Update actor and critic networks
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        try:
            # Sample experience
            batch = self.replay_buffer.sample(batch_size)
            state_batch = torch.FloatTensor(batch.state).to(self.device)
            action_batch = torch.FloatTensor(batch.action).to(self.device)
            reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
            next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
            done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

            n_operations = len(self.aug_operations) + 1
            n_magnitudes = self.n_magnitude_levels

            # Update critics
            with torch.no_grad():
                next_operation_logits, next_magnitude_logits, next_log_prob = self.actor.sample(
                    next_state_batch, n_operations, n_magnitudes)

                next_operation_dist = torch.distributions.Categorical(logits=next_operation_logits)
                next_magnitude_dist = torch.distributions.Categorical(logits=next_magnitude_logits)

                next_operation_idx = next_operation_dist.sample()
                next_magnitude_idx = next_magnitude_dist.sample()

                next_action = torch.zeros_like(action_batch)
                batch_indices = torch.arange(next_action.shape[0])
                next_action[batch_indices, next_operation_idx] = 1.0
                next_action[batch_indices, n_operations + next_magnitude_idx] = 1.0

                target_q1 = self.target_critic1(next_state_batch, next_action)
                target_q2 = self.target_critic2(next_state_batch, next_action)

                default_alpha = self.adaptive_alpha.get_alpha_for_difficulty('medium')
                target_q = torch.min(target_q1, target_q2) - default_alpha * next_log_prob.unsqueeze(1)
                target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

            current_q1 = self.critic1(state_batch, action_batch)
            current_q2 = self.critic2(state_batch, action_batch)

            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_optimizer.step()

            # Update actor
            operation_logits, magnitude_logits, log_prob = self.actor.sample(
                state_batch, n_operations, n_magnitudes)

            new_action = torch.zeros_like(action_batch)
            operation_dist = torch.distributions.Categorical(logits=operation_logits)
            magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)

            operation_idx = operation_dist.sample()
            magnitude_idx = magnitude_dist.sample()

            batch_indices = torch.arange(new_action.shape[0])
            new_action[batch_indices, operation_idx] = 1.0
            new_action[batch_indices, n_operations + magnitude_idx] = 1.0

            q1_new = self.critic1(state_batch, new_action)
            q2_new = self.critic2(state_batch, new_action)
            q_new = torch.min(q1_new, q2_new)

            default_alpha = self.adaptive_alpha.get_alpha_for_difficulty('medium')
            actor_loss = (default_alpha * log_prob.unsqueeze(1) - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)

            return {
                'actor_loss': actor_loss.item(),
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'alpha': default_alpha.item(),
                'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
                'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item()
            }
        except Exception:
            return {
                'actor_loss': 0.0,
                'critic1_loss': 0.0,
                'critic2_loss': 0.0,
                'alpha': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
                'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item()
            }

    def soft_update(self, target_network, source_network):
        """Soft update of target network"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def get_statistics(self):
        """Get training statistics"""
        if not self.statistics['rewards']:
            base_stats = {
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'avg_ba_improvement': 0.0,
                'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
                'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item(),
                'training_step': self.training_step,
                'total_episodes': 0,
                'current_phase': self.current_phase,
                'phase2_triggered': self.phase2_triggered
            }

            confidence_stats = self.confidence_tracker.get_confidence_statistics()
            base_stats.update({f'confidence_{k}': v for k, v in confidence_stats.items()})

            return base_stats

        rewards = self.statistics['rewards']
        ba_improvements = self.statistics.get('ba_improvement', [0.0])

        stats = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_ba_improvement': np.mean(ba_improvements),
            'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
            'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
            'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item(),
            'training_step': self.training_step,
            'total_episodes': len(rewards),
            'current_phase': self.current_phase,
            'phase2_triggered': self.phase2_triggered
        }

        confidence_stats = self.confidence_tracker.get_confidence_statistics()
        stats.update({f'confidence_{k}': v for k, v in confidence_stats.items()})

        if self.current_phase == 2:
            stats.update(self.phase2_stats)
            difficulties = self.confidence_tracker.get_all_difficulties()
            stats.update({f'total_{k}_samples': v for k, v in difficulties.items()})
            stats.update(self.adaptive_alpha.get_all_alphas())

        return stats


class AdaptiveActorNetwork(nn.Module):
    """Adaptive actor network with dynamic action space"""

    def __init__(self, state_dim, max_operations, max_magnitude_levels, hidden_dim=256):
        super().__init__()
        self.max_operations = max_operations
        self.max_magnitude_levels = max_magnitude_levels

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.operation_head = nn.Linear(hidden_dim, max_operations)
        self.magnitude_head = nn.Linear(hidden_dim, max_magnitude_levels)

    def forward(self, state, n_operations=None, n_magnitude_levels=None):
        x = self.shared(state)

        operation_logits = self.operation_head(x)
        magnitude_logits = self.magnitude_head(x)

        if n_operations is not None:
            operation_logits = operation_logits[:, :n_operations]
        if n_magnitude_levels is not None:
            magnitude_logits = magnitude_logits[:, :n_magnitude_levels]

        return operation_logits, magnitude_logits

    def sample(self, state, n_operations=None, n_magnitude_levels=None):
        operation_logits, magnitude_logits = self.forward(state, n_operations, n_magnitude_levels)

        operation_dist = torch.distributions.Categorical(logits=operation_logits)
        magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)

        operation_sample = operation_dist.sample()
        magnitude_sample = magnitude_dist.sample()

        operation_log_prob = operation_dist.log_prob(operation_sample)
        magnitude_log_prob = magnitude_dist.log_prob(magnitude_sample)

        total_log_prob = operation_log_prob + magnitude_log_prob

        return operation_logits, magnitude_logits, total_log_prob


class CriticNetwork(nn.Module):
    """Critic network for Q-value estimation"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        from collections import namedtuple
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        return Transition(state, action, reward, next_state, done)

    def __len__(self):
        return len(self.buffer)