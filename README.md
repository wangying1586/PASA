# PASA: Progressive Adaptive Sample-level Augmentation for Respiratory Sound Classification (AAAI 2026)

### Abstract

Automated auscultation advances the detection of respiratory diseases, especially in areas with limited resources where traditional diagnostic methods are unavailable. On the other hand, the scarcity of auscultation datasets limits the automation performance, prompting the needs for data augmentation methods. However, most of the existing methods neglect the difference in acoustic sounds that requires personalized augmentation strategies. To address this, we propose a Progressive-Adaptive Spectral Augmentation (PASA), which is one of the first paradigms to adaptively select the best augmentation strategy for each sample. The PASA innovatively treats augmentation selection problem as a Markov Decision Process (MDP), creating an alternating loop between the diagnostic model and the augmentation selection. The agent selects the optimal augmentation operations and magnitudes via a task-specific design, including state construction, action sampling, Hybrid Batch-Sample (HBS) strategy execution, and reward guidance. The HBS strategy initially applies uniform augmentation across mini-batches while collecting sample-specific performance statistics. When model performance stabilizes, it transits to sample-level augmentation based on accumulated difficulty assessments. This two-phase design balances computational complexity with personalization. Extensive experiments across three benchmark datasets demonstrate that the PASA outperforms the state-of-the-art methods, pioneering a transformative paradigm for adaptive data augmentation in automated auscultation.

### PASA Architecture
![2.Methodology_Overview.png](2.Methodology_Overview.png "相对路径演示") 

### Five-Step Workflow
1. State Construction: Extract features and construct MDP states
2. Action Sampling: Use SAC to sample augmentation actions  
3. HBS Strategy: Execute Hybrid Batch-Sample augmentation
4. Reward Computation: Calculate performance-based rewards
5. Network Optimization: Joint optimization of all networks
   
---
## Installation
```bash
# Clone repository
git clone https://github.com/wangying1586/PASA.git
cd PASA

# Create conda environment
conda create -n pasa python=3.8
conda activate pasa

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### Download Datasets

- **SPRSound**: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- **ICBHI 2017**: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- **CirCor DigiScope**: https://physionet.org/content/circor-heart-sound/1.0.3/

### Directory Structure

Place downloaded datasets in the following structure:
```
data/
├── SPRSound/
│   ├── Classification/
│   │   ├── restore_train_classification_wav/
│   │   ├── train_classification_json/
│   │   ├── restore_valid_classification_wav/
│   │   └── valid_classification_json/
├── ICBHI/
│   ├── ICBHI_final_database/
│   └── ICBHI_challenge_train_test.txt
└── CirCor/
    └── the-circor-digiscope-phonocardiogram-dataset-1.0.3/
```

### Preprocessing

Run preprocessing scripts to generate `.npy` files [Update dataset paths in preprocessing scripts before running]:
```bash
# SPRSound (year:22, 23 | all tasks: 11, 12, 21, 22)
python data_preprocess/SPRSound2022_2023_preprocessor.py

# ICBHI (tasks: binary & multiclass)
python data_preprocess/ICBHI_preprocessor.py

# CirCor DigiScope (heart murmur detection)
python data_preprocess/CirCor_preprocessor.py
```

After preprocessing, verify generated files:
```
datasets/
├── SPRSound_2022+2023_processed_data/
│   ├── task11/
│   │   ├── train_specs.npy
│   │   ├── train_labels.npy
│   │   └── test_*.npy
│   ├── task12/
│   ├── task21/
│   └── task22/
├── ICBHI2017_processed_data/
│   ├── binary/
│   └── multiclass/
└── CirCor_DigiScope_2022_processed_data/
    ├── train_specs.npy
    └── train_labels.npy
```

---

## Training

### Basic Usage
```bash
./train.sh      
```

### Examples
```bash
# SPRSound Task 22 (5-class recording classification)
./train.sh SPRSound 22 True 20 0.01 0.95

# CirCor DigiScope (3-class heart murmur detection)
./train.sh CirCor heart_murmur True 20 0.01 0.95

# Baseline without augmentation
./train.sh SPRSound 22 False
```

### Model Checkpoints

After training step, the models are saved in `experiments/` directory:
```
experiments/
└── SPRSound_task22_EfficientNet-B4_log-mel_TwoPhaseSAC_P20_T0.01_YYYYMMDD_HHMMSS/
    ├── models/
    │   ├── best_fold_1.pth
    │   ├── best_fold_2.pth
    │   ├── best_fold_3.pth
    │   ├── best_fold_4.pth
    │   ├── best_fold_5.pth
    │   └── final_model.pth
    ├── config.txt
    └── cv_results_summary.json
```

---

## Testing and Evaluation

### Run Testing
```bash
python test.py \
    --dataset SPRSound \
    --task_type 22 \
    --data_dir ./datasets \
    --exp_dir experiments/SPRSound_task22_EfficientNet-B4_log-mel_TwoPhaseSAC_P20_T0.01_YYYYMMDD_HHMMSS \
    --batch_size 32
```

Replace `YYYYMMDD_HHMMSS` with your actual experiment timestamp.

### Test Results

Results are saved in `test_results/` directory:
```
test_results/
└── <DATASET>/task_<TASK_TYPE>/YYYYMMDD_HHMMSS/
    ├── confusion_matrix_fold_*.png
    ├── confusion_matrix_ensemble_*.png
    ├── confusion_matrix_final_*.png
    ├── roc_curve_*.png
    ├── pr_curve_*.png
    ├── results_summary.txt
    ├── <DATASET>_task_<TASK_TYPE>_test_results.json
    ├── test_config.json
    └── detailed_predictions/
        ├── test_*_predictions.csv
        └── ensemble_predictions.csv
```

### Visualizations

The testing script generates:

1. **Confusion Matrices**: Shows prediction distribution across classes
   - Per-fold models (best_fold_1.pth to best_fold_5.pth)
   - Ensemble model (average of all folds)
   - Final model (final_model.pth)

2. **ROC Curves**: Receiver Operating Characteristic with AUC scores per class

3. **Precision-Recall Curves**: PR curves with Average Precision per class

4. **Performance Summary Table**: Comprehensive metrics comparison including:
   - Accuracy, F1-score
   - Sensitivity (Recall), Specificity
   - Overall Score (harmonic and arithmetic mean)
   - For CirCor: W.acc (Weighted Accuracy) and UAR (Unweighted Average Recall)

### Understanding CirCor Metrics

For CirCor DigiScope heart murmur detection:
- **W.acc**: Weighted accuracy based on challenge scoring (5×Present + 3×Unknown + Absent)
- **UAR**: Unweighted Average Recall across three classes
- **Class-specific Recalls**: Present, Absent, Unknown

[THE VISUALIZATION OF SPRSound 2022 and SPRSound 2023 test dataset are in the test_results.]
---

### Adding Custom Augmentation Operations

Edit `PASA.py` in the `TwoPhaseSACStrategy` class:

1. Add operation to `_init_magnitude_ranges()`:
```python
def _init_magnitude_ranges(self):
    configs = {
        "your_augmentation": (min_val, max_val),
        "time_mask": (0.05, 0.3),
        # ... existing operations
    }
```

2. Implement the augmentation function:
```python
def _apply_your_augmentation(self, x, y, magnitude):
    # Your augmentation logic here
    augmented_x = x  # Apply transformation
    return augmented_x, y
```

3. Add to `apply_augmentation()` method:
```python
def apply_augmentation(self, x, y, operation_idx, magnitude_idx):
    operation_name = self.aug_operations[operation_idx]
    
    if operation_name == "your_augmentation":
        return self._apply_your_augmentation(x, y, magnitude)
    # ... existing conditions
```

---

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{pasa2026,
  title={PASA: Progressive-Adaptive Spectral Augmentation for Automated Auscultation in Data-Scarce Environments},
  author={Ying Wang, Guoheng Huang, Xueyuan Gong, Xinxin Wang, Xiaochen Yuan*},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: p2412918@mpu.edu.mo

---
