import os
import json
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional
from sklearn.model_selection import train_test_split
import scipy.signal as signal

warnings.filterwarnings('ignore')


class CirCorDigiScopeProcessor:
    """
    CirCor DigiScope dataset processor for heart murmur detection
    Following the exact methodology from:
    "Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection"
    Niizumi et al., 2024

    Key preprocessing steps from the reference paper:
    1. Convert all recordings to 16 kHz sampling rate to match model input
    2. Uniformly provide 5-second fixed-length audio input to the model for simplicity
    3. Training: split every 5s with a 2.5s stride (overlapping segments)
    4. Testing: split each recording into 5s segments without overlap
    5. Extract log-mel spectrogram features with standard parameters
    """

    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 target_sr: int = 16000,
                 segment_length: float = 5.0,
                 training_stride: float = 2.5,
                 feature_type: str = 'log-mel',
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 test_size: float = 0.25,
                 random_state: int = 42,
                 save_visualizations: bool = True):
        """
        Initialize the CirCor DigiScope dataset processor following the reference paper

        Args:
            data_dir: Directory containing the CirCor DigiScope dataset
            output_dir: Directory to save processed files
            target_sr: Target sampling rate (16000 Hz - paper requirement)
            segment_length: Length of each audio segment in seconds (5.0s - paper requirement)
            training_stride: Stride for training segmentation (2.5s - paper requirement)
            feature_type: Feature extraction type ('log-mel' as per paper)
            n_mels: Number of mel filters for log-mel spectrogram (128 - standard)
            n_fft: FFT window size for spectrogram computation (1024 - standard for 16kHz)
            hop_length: Hop length for spectrogram computation (512 - standard)
            test_size: Test set proportion (0.25 for 75:25 train:test split for 5-fold CV)
            random_state: Random seed for reproducibility
            save_visualizations: Whether to save sample visualizations

        Note: Using 75:25 train:test split as user will perform 5-fold cross-validation on training set
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.training_stride = training_stride
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.test_size = test_size
        self.random_state = random_state
        self.save_visualizations = save_visualizations

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data storage (train and test sets only - user will do 5-fold CV)
        self.data = {
            'train': {'specs': [], 'labels': [], 'patient_ids': [], 'recording_ids': []},
            'test': {'specs': [], 'labels': [], 'patient_ids': [], 'recording_ids': []}
        }

        # Label mapping for heart murmur detection (3-class classification as per paper)
        # Present: 心脏杂音存在, Absent: 心脏杂音不存在, Unknown: 无法确定
        self.label_mapping = {
            'Present': 0,  # 心脏杂音存在
            'Absent': 1,  # 心脏杂音不存在
            'Unknown': 2  # 无法确定/未知
        }

        self.class_names = ['Present', 'Absent', 'Unknown']
        self.metadata = []

        print(f"CirCor DigiScope Processor initialized following reference paper:")
        print(f"Reference: 'Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection'")
        print(f"Task: Heart Murmur Detection (3-class classification)")
        print(f"Classes: {self.class_names}")
        print(f"Target sampling rate: {self.target_sr} Hz (paper requirement)")
        print(f"Segment length: {self.segment_length} seconds (paper requirement)")
        print(f"Training stride: {self.training_stride} seconds (paper requirement)")
        print(f"Testing stride: {self.segment_length} seconds (non-overlapping - paper requirement)")
        print(f"Feature type: {self.feature_type} (paper standard)")
        print(f"Data split: 75:25 (train:test) for 5-fold cross-validation")

    def _load_patient_data(self):
        """Load patient data and murmur labels from the dataset"""
        patient_data = {}

        # Look for patient data file (CSV format)
        possible_files = ['training_data.csv', 'patient_data.csv', 'outcomes.csv']
        patient_file = None

        for filename in possible_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                patient_file = filepath
                break

        if patient_file is None:
            raise FileNotFoundError(f"Could not find patient data file. Looked for: {possible_files}")

        # Load patient data
        df = pd.read_csv(patient_file)
        print(f"Loading patient data from: {os.path.basename(patient_file)}")

        # Extract patient IDs and murmur labels
        for _, row in df.iterrows():
            # Handle different possible column names
            patient_id = None
            for col in ['Patient ID', 'patient_id', 'PatientID']:
                if col in row and pd.notna(row[col]):
                    patient_id = str(row[col])
                    break

            if patient_id is None:
                patient_id = str(row.iloc[0])  # Use first column if no standard name found

            # Handle different possible murmur column names
            murmur_label = None
            for col in ['Murmur', 'murmur', 'Outcome', 'outcome']:
                if col in row and pd.notna(row[col]):
                    murmur_label = row[col]
                    break

            if murmur_label is None:
                print(f"Warning: No murmur label found for patient {patient_id}")
                continue

            patient_data[patient_id] = {
                'murmur': murmur_label,
                'age': row.get('Age', None),
                'sex': row.get('Sex', None),
                'pregnancy_status': row.get('Pregnancy status', None)
            }

        print(f"Loaded data for {len(patient_data)} patients")

        # Print class distribution
        murmur_counts = {}
        for patient_info in patient_data.values():
            murmur = patient_info['murmur']
            murmur_counts[murmur] = murmur_counts.get(murmur, 0) + 1

        print("Class distribution:")
        for label, count in murmur_counts.items():
            print(f"  {label}: {count} patients")

        return patient_data

    def _get_audio_files(self):
        """Get all audio files in the dataset"""
        audio_files = []

        # Look for audio files with common extensions
        for ext in ['*.wav', '*.WAV']:
            audio_files.extend(list(Path(self.data_dir).rglob(ext)))

        print(f"Found {len(audio_files)} audio files")
        return audio_files

    def _extract_patient_and_location(self, filename):
        """Extract patient ID and auscultation location from filename"""
        # CirCor filenames format: PatientID_Location.wav
        # e.g., "12345_AV.wav", "12345_PV.wav", "12345_TV.wav", "12345_MV.wav"
        basename = os.path.basename(filename).replace('.wav', '').replace('.WAV', '')

        if '_' in basename:
            parts = basename.split('_')
            patient_id = parts[0]
            location = parts[1] if len(parts) > 1 else 'Unknown'
        else:
            # Handle cases where there's no underscore
            patient_id = basename
            location = 'Unknown'

        return patient_id, location

    def _preprocess_audio(self, audio_path):
        """
        Load and preprocess audio file according to the reference paper methodology

        Key steps from "Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection":
        1. Convert to 16 kHz sampling rate to match model input requirements
        2. Normalize audio for consistency
        """
        try:
            # Load audio file - convert to 16 kHz to match model input (as per paper)
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

            # Normalize audio - the paper mentions CirCor is already normalized but we ensure consistency
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio

        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return None

    def _segment_audio(self, audio, is_training=True):
        """
        Segment audio according to the reference paper methodology:

        From the paper:
        - Training: split every 5s with a 2.5s stride (overlapping segments for data augmentation)
        - Testing: split each recording into 5s segments without overlap

        Args:
            audio: Audio signal
            is_training: If True, use 2.5s stride; if False, use 5s stride (no overlap)
        """
        segments = []
        segment_samples = int(self.segment_length * self.target_sr)  # 5 seconds = 80,000 samples at 16kHz

        if is_training:
            # Training phase: 2.5s stride as per paper (overlapping segments)
            stride_samples = int(self.training_stride * self.target_sr)  # 2.5 seconds = 40,000 samples
        else:
            # Testing phase: no overlap as per paper
            stride_samples = segment_samples  # 5 seconds (no overlap)

        # Handle short audio: if shorter than 5s, pad with zeros
        if len(audio) < segment_samples:
            padded_audio = np.zeros(segment_samples, dtype=np.float32)
            padded_audio[:len(audio)] = audio
            segments.append(padded_audio)
        else:
            # Create segments with specified stride
            start = 0
            while start + segment_samples <= len(audio):
                segment = audio[start:start + segment_samples].astype(np.float32)
                segments.append(segment)
                start += stride_samples

            # For testing phase, if there's remaining audio, create one more segment
            if not is_training and start < len(audio):
                # Take the last 5 seconds to avoid losing data
                last_segment = audio[-segment_samples:].astype(np.float32)
                segments.append(last_segment)

        return segments

    def _extract_log_mel_features(self, audio):
        """
        Extract log-mel spectrogram features following the reference paper

        Standard parameters for heart sound analysis as used in the paper:
        - Mel spectrogram with 128 mel bins
        - FFT size: 1024 (appropriate for 16kHz)
        - Hop length: 512 (standard)
        - Frequency range: 0 Hz to Nyquist (8000 Hz for 16kHz sampling)
        - Convert to log scale (dB) for better representation
        """
        try:
            # Extract mel spectrogram with parameters suitable for heart sound analysis
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.target_sr,
                n_fft=self.n_fft,  # 1024 - standard for 16kHz
                hop_length=self.hop_length,  # 512 - standard
                n_mels=self.n_mels,  # 128 mel bins - paper standard
                fmin=0,  # Start from 0 Hz (important for heart sounds)
                fmax=self.target_sr // 2,  # Nyquist frequency (8000 Hz)
                power=2.0  # Power spectrogram
            )

            # Convert to log scale (dB) - this is the key step for log-mel
            # Using librosa's power_to_db with ref=np.max for consistent scaling
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            return log_mel_spec.astype(np.float32)

        except Exception as e:
            print(f"Error extracting log-mel features: {e}")
            return None

    def _split_data(self, patient_data):
        """Split patients into train/test sets with stratification (75:25) for 5-fold cross-validation"""
        # Extract patient IDs and labels
        patient_ids = list(patient_data.keys())
        labels = [patient_data[pid]['murmur'] for pid in patient_ids]

        # Convert labels to numeric
        numeric_labels = []
        for label in labels:
            if label in self.label_mapping:
                numeric_labels.append(self.label_mapping[label])
            else:
                print(f"Warning: Unknown label '{label}' found, skipping patient")
                continue

        # Filter out patients with unknown labels
        valid_patients = [(pid, label) for pid, label in zip(patient_ids, labels)
                          if label in self.label_mapping]

        if len(valid_patients) != len(patient_ids):
            print(f"Filtered {len(patient_ids) - len(valid_patients)} patients with invalid labels")

        patient_ids = [pid for pid, _ in valid_patients]
        numeric_labels = [self.label_mapping[label] for _, label in valid_patients]

        # Stratified split into train (75%) and test (25%) sets
        train_ids, test_ids, _, _ = train_test_split(
            patient_ids, numeric_labels,
            test_size=self.test_size,  # 25%
            stratify=numeric_labels,
            random_state=self.random_state
        )

        split_info = {
            'train': set(train_ids),
            'test': set(test_ids)
        }

        print(f"Data split following stratified sampling (75:25 for 5-fold CV):")
        print(f"  Train: {len(train_ids)} patients ({len(train_ids) / len(patient_ids) * 100:.1f}%)")
        print(f"  Test: {len(test_ids)} patients ({len(test_ids) / len(patient_ids) * 100:.1f}%)")

        # Print class distribution in each split
        for split_name, patient_set in [('train', train_ids), ('test', test_ids)]:
            split_labels = [self.label_mapping[patient_data[pid]['murmur']] for pid in patient_set]
            print(f"  {split_name.capitalize()} class distribution:")
            for i, class_name in enumerate(self.class_names):
                count = split_labels.count(i)
                print(f"    {class_name}: {count} patients")

        return split_info

    def process_dataset(self):
        """Process the entire dataset following the reference paper methodology"""
        print("Starting CirCor DigiScope dataset processing...")
        print("Following: 'Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection'")

        # Load patient data and labels
        patient_data = self._load_patient_data()

        # Get audio files
        audio_files = self._get_audio_files()

        # Split data into train/test
        split_info = self._split_data(patient_data)

        # Process audio files
        processed_count = 0
        skipped_count = 0

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            patient_id, location = self._extract_patient_and_location(audio_file.name)

            # Skip if patient not in our dataset
            if patient_id not in patient_data:
                skipped_count += 1
                continue

            # Get patient information
            patient_info = patient_data[patient_id]
            murmur_label = patient_info['murmur']

            # Convert label to numeric
            if murmur_label not in self.label_mapping:
                print(f"Unknown label {murmur_label} for patient {patient_id}, skipping...")
                skipped_count += 1
                continue

            label = self.label_mapping[murmur_label]

            # Determine split (train/test)
            split = None
            for split_name, patient_set in split_info.items():
                if patient_id in patient_set:
                    split = split_name
                    break

            if split is None:
                skipped_count += 1
                continue

            # Process audio following paper methodology
            audio = self._preprocess_audio(str(audio_file))
            if audio is None:
                skipped_count += 1
                continue

            # Segment audio - use overlapping segments for training, non-overlapping for testing
            is_training = (split == 'train')
            segments = self._segment_audio(audio, is_training=is_training)

            # Process each segment
            for i, segment in enumerate(segments):
                # Extract log-mel spectrogram features
                features = self._extract_log_mel_features(segment)
                if features is None:
                    continue

                # Store data
                recording_id = f"{patient_id}_{location}_{i:03d}"
                self.data[split]['specs'].append(features)
                self.data[split]['labels'].append(label)
                self.data[split]['patient_ids'].append(patient_id)
                self.data[split]['recording_ids'].append(recording_id)

                # Store metadata
                self.metadata.append({
                    'patient_id': patient_id,
                    'recording_id': recording_id,
                    'location': location,
                    'segment_index': i,
                    'label': label,
                    'label_text': murmur_label,
                    'split': split,
                    'age': patient_info.get('age'),
                    'sex': patient_info.get('sex'),
                    'pregnancy_status': patient_info.get('pregnancy_status'),
                    'audio_file': str(audio_file),
                    'segment_start_time': i * (self.training_stride if is_training else self.segment_length),
                    'segment_duration': self.segment_length
                })

            processed_count += 1

        print(f"Processing complete: {processed_count} files processed, {skipped_count} files skipped")

        # Convert to numpy arrays with appropriate data types
        for split in ['train', 'test']:
            if self.data[split]['specs']:
                self.data[split]['specs'] = np.array(self.data[split]['specs'], dtype=np.float32)
                self.data[split]['labels'] = np.array(self.data[split]['labels'], dtype=np.int64)

                print(f"\n{split.capitalize()} data converted:")
                print(f"  Log-mel specs shape: {self.data[split]['specs'].shape}")
                print(f"  Labels shape: {self.data[split]['labels'].shape}")
                print(f"  Feature dtype: {self.data[split]['specs'].dtype}")
                print(f"  Labels dtype: {self.data[split]['labels'].dtype}")

        # Print comprehensive dataset statistics
        self._print_dataset_stats()

        # Save visualizations if requested
        if self.save_visualizations:
            self._save_sample_visualizations()

        # Save processed data
        self._save_processed_data()

        return self.data

    def _print_dataset_stats(self):
        """Print comprehensive dataset statistics"""
        print("\n" + "=" * 70)
        print("CirCor DigiScope Dataset Statistics")
        print("Following reference paper methodology")
        print("=" * 70)

        # Create DataFrame from metadata for analysis
        df = pd.DataFrame(self.metadata)

        if not df.empty:
            # Sample distribution by split and label
            print("\nSample distribution by split and label:")
            split_label_counts = df.groupby(['split', 'label_text']).size().unstack(fill_value=0)
            print(split_label_counts)

            # Total samples by split
            print("\nTotal samples by split:")
            split_counts = df.groupby('split').size()
            for split, count in split_counts.items():
                print(f"  {split}: {count} segments")

            # Patient distribution by split
            print("\nUnique patients by split:")
            patient_split_counts = df.groupby('split')['patient_id'].nunique()
            for split, count in patient_split_counts.items():
                print(f"  {split}: {count} patients")

            # Feature shape information
            for split in ['train', 'test']:
                if len(self.data[split]['specs']) > 0:
                    sample_shape = self.data[split]['specs'][0].shape
                    print(f"\n{split.capitalize()} log-mel spectrogram shape: {sample_shape}")
                    print(f"  - Mel bins (frequency): {sample_shape[0]}")
                    print(f"  - Time frames: {sample_shape[1]}")

                    # Calculate time duration represented
                    time_duration = sample_shape[1] * self.hop_length / self.target_sr
                    print(f"  - Time duration: ~{time_duration:.2f} seconds")
                    print(f"  - Sampling rate: {self.target_sr} Hz")
                    print(f"  - Segment length: {self.segment_length} seconds")

            # Segmentation statistics
            print(f"\nSegmentation strategy (as per reference paper):")
            print(f"  - Training segments: {self.segment_length}s with {self.training_stride}s stride (overlapping)")
            print(f"  - Testing segments: {self.segment_length}s with {self.segment_length}s stride (non-overlapping)")

        print("=" * 70)

    def _save_sample_visualizations(self, n_samples_per_class=2):
        """Save sample log-mel spectrogram visualizations"""
        print("Saving sample log-mel spectrogram visualizations...")

        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Visualize each dataset split
        for split in ['train', 'test']:
            if len(self.data[split]['specs']) == 0:
                continue

            specs = self.data[split]['specs']
            labels = self.data[split]['labels']

            # Select samples for each class
            unique_labels = np.unique(labels)
            selected_indices = []

            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                if len(label_indices) > 0:
                    # Select multiple samples per class
                    n_select = min(n_samples_per_class, len(label_indices))
                    selected = np.random.choice(label_indices, n_select, replace=False)
                    selected_indices.extend(selected)

            # Create visualization
            n_samples = len(selected_indices)
            fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
            if n_samples == 1:
                axes = [axes]

            for i, idx in enumerate(selected_indices):
                spec = specs[idx]
                label = labels[idx]
                label_text = self.class_names[label]

                # Plot log-mel spectrogram
                img = librosa.display.specshow(
                    spec,
                    sr=self.target_sr,
                    hop_length=self.hop_length,
                    x_axis='time',
                    y_axis='mel',
                    ax=axes[i],
                    cmap='viridis'
                )

                plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
                axes[i].set_title(f'{split.capitalize()} - {label_text} (Sample {idx})')
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel('Mel Frequency Bins')

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'log_mel_samples_{split}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved log-mel visualization for {split} split")

    def _save_processed_data(self):
        """Save processed data to NPY files and metadata to CSV"""
        print("Saving processed data...")

        # Save log-mel spectrogram arrays and labels
        for split in ['train', 'test']:
            if len(self.data[split]['specs']) > 0:
                # Save log-mel spectrograms and labels as NPY files
                specs_file = os.path.join(self.output_dir, f"{split}_specs.npy")
                labels_file = os.path.join(self.output_dir, f"{split}_labels.npy")

                np.save(specs_file, self.data[split]['specs'])
                np.save(labels_file, self.data[split]['labels'])

                print(f"Saved {split} data:")
                print(f"  Log-mel specs: {specs_file} - shape {self.data[split]['specs'].shape}")
                print(f"  Labels: {labels_file} - shape {self.data[split]['labels'].shape}")

        # Save metadata
        df = pd.DataFrame(self.metadata)
        if not df.empty:
            metadata_file = os.path.join(self.output_dir, "metadata.csv")
            df.to_csv(metadata_file, index=False)
            print(f"Saved metadata: {metadata_file} - {len(df)} entries")

        # Save comprehensive dataset information
        info = {
            'dataset': 'CirCor DigiScope Phonocardiogram Dataset',
            'task': 'Heart Murmur Detection (3-class classification)',
            'reference_paper': 'Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection',
            'authors': 'Niizumi et al.',
            'challenge': 'George B. Moody PhysioNet Challenge 2022',
            'classes': self.class_names,
            'label_mapping': self.label_mapping,
            'preprocessing': {
                'methodology': 'Following reference paper exactly',
                'target_sampling_rate_hz': self.target_sr,
                'segment_length_seconds': self.segment_length,
                'training_stride_seconds': self.training_stride,
                'testing_stride_seconds': self.segment_length,
                'normalization': 'Per-recording max normalization',
                'segmentation_strategy': {
                    'training': 'Overlapping 5s segments with 2.5s stride',
                    'testing': 'Non-overlapping 5s segments'
                }
            },
            'feature_extraction': {
                'type': 'Log-mel spectrogram',
                'n_mels': self.n_mels,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'fmin_hz': 0,
                'fmax_hz': self.target_sr // 2,
                'power': 2.0,
                'db_conversion': 'librosa.power_to_db with ref=np.max'
            },
            'data_split': {
                'method': 'Stratified patient-level split',
                'train_ratio': 1 - self.test_size,
                'test_ratio': self.test_size,
                'split_proportions': '75:25 (train:test) for 5-fold cross-validation',
                'random_state': self.random_state,
                'note': 'Use 5-fold cross-validation on training set for model validation'
            },
            'file_format': {
                'features': 'NumPy arrays (.npy) - float32',
                'labels': 'NumPy arrays (.npy) - int64',
                'metadata': 'CSV file with detailed information'
            },
            'usage_notes': {
                'training': 'Use overlapping segments for data augmentation',
                'testing': 'Use non-overlapping segments for fair evaluation',
                'model_input': 'Log-mel spectrograms with shape (n_mels, time_frames)',
                'recommended_validation': '5-fold cross-validation on training set'
            }
        }

        # Save as JSON
        info_file = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Saved dataset info: {info_file}")
        print(f"\nProcessing complete! All data saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  - train_specs.npy: Training log-mel spectrograms")
        print("  - train_labels.npy: Training labels")
        print("  - test_specs.npy: Test log-mel spectrograms")
        print("  - test_labels.npy: Test labels")
        print("  - metadata.csv: Detailed metadata for all samples")
        print("  - dataset_info.json: Complete dataset information")
        print("  - visualizations/: Sample log-mel spectrogram plots")


# Example usage following the reference paper specifications exactly
if __name__ == "__main__":
    # Set paths (update these to match your dataset location)
    data_dir = "/home/p2412918/Lung_sound_detection_C1/datasets/CirCor_DigiScope_Database/the-circor-digiscope-phonocardiogram-dataset-1.0.3"  # Path to CirCor DigiScope dataset
    output_dir = "/home/p2412918/Lung_sound_detection_C1/datasets/CirCor_DigiScope_2022_processed_data"  # Output directory

    # Create processor following the reference paper methodology exactly
    # "Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection"
    processor = CirCorDigiScopeProcessor(
        data_dir=data_dir,
        output_dir=output_dir,
        target_sr=16000,  # 16kHz - convert to match model input (paper requirement)
        segment_length=5.0,  # 5 seconds - fixed-length audio input (paper requirement)
        training_stride=2.5,  # 2.5s stride for training (paper requirement)
        feature_type='log-mel',  # Log-mel spectrogram (paper standard)
        n_mels=128,  # 128 mel filters (standard for audio)
        n_fft=1024,  # Standard FFT size for 16kHz
        hop_length=512,  # Standard hop length
        test_size=0.25,  # 25% test set (75:25 split for 5-fold CV)
        random_state=42,  # For reproducibility
        save_visualizations=True
    )

    # Process dataset following reference paper methodology
    data = processor.process_dataset()

    print("\n" + "=" * 80)
    print("CirCor DigiScope Dataset Processing Completed!")
    print("=" * 80)
    print("Reference paper methodology implemented:")
    print("- Convert all recordings to 16 kHz sampling rate")
    print("- Training: split every 5s with 2.5s stride (overlapping)")
    print("- Testing: split into 5s segments without overlap")
    print("- Extract log-mel spectrogram features")
    print("- Save as NumPy arrays (.npy format)")
    print("- Data split: 75:25 (train:test) with stratified sampling")
    print("- Use 5-fold cross-validation on training set for validation")
    print("=" * 80)
    # Fix the problematic lines by checking if the array exists and has elements
    print(f"Train samples: {len(data['train']['specs']) if data['train']['specs'] is not None and len(data['train']['specs']) > 0 else 0}")
    print(f"Test samples: {len(data['test']['specs']) if data['test']['specs'] is not None and len(data['test']['specs']) > 0 else 0}")
    print("=" * 80)
    print("Task: Heart Murmur Detection (3-class classification)")
    print("Classes: Present (存在), Absent (不存在), Unknown (未知)")
    print("=" * 80)
    print("Reference: 'Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection'")
    print("Authors: Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, Kunio Kashino")
    print("=" * 80)