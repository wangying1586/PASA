import os
import json
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import scipy.signal as signal
from pathlib import Path
import warnings
from typing import Dict, Tuple, List, Union, Optional

warnings.filterwarnings('ignore')


def get_class_names(task_type):
    """Get class names for each task type"""
    if task_type == 11:
        return ['Normal', 'Adventious']
    elif task_type == 12:
        return ['Normal', 'Rhonchi', 'Wheeze', 'Stridor',
                'Coarse Crackle', 'Fine Crackle', 'Wheeze+Crackle']
    elif task_type == 21:
        return ['Normal', 'Poor Quality', 'Adventious']
    elif task_type == 22:
        return ['Normal', 'Poor Quality', 'CAS', 'DAS', 'CAS & DAS']


class SPRSoundDatasetProcessor:
    """
    SPRSound dataset processor
    Processes dataset and saves as NPY and CSV files for easier loading later
    """

    def __init__(self,
                 wav_path: str,
                 json_path: str,
                 output_dir: str,
                 task_type: int,
                 year: Optional[int] = None,
                 valid_type: Optional[str] = None,
                 feature_type: str = 'log-mel',
                 frame_length: Optional[int] = None,
                 hop_ratio: int = 4,
                 save_visualizations: bool = True,
                 test_2022_wav_path: Optional[str] = None,
                 test_2022_json_path: Optional[str] = None,
                 test_2023_wav_path: Optional[str] = None,
                 test_2023_json_path: Optional[str] = None):
        """
        Initialize the dataset processor

        Args:
            wav_path: Path to wav files
            json_path: Path to json annotation files
            output_dir: Directory to save processed files
            task_type: Task type (11, 12, 21, 22)
            year: Year of dataset (2022, 2023)
            valid_type: Validation type ('intra', 'inter')
            feature_type: Feature type ('log-mel', 'MFCC', 'mel', 'STFT')
            frame_length: Frame length for feature extraction (if None, will be set automatically)
            hop_ratio: Hop ratio for feature extraction
            save_visualizations: Whether to save sample visualizations
            test_2022_wav_path: Path to 2022 test WAV files
            test_2022_json_path: Path to 2022 test JSON files
            test_2023_wav_path: Path to 2023 test WAV files
            test_2023_json_path: Path to 2023 test JSON files
        """
        self.wav_path = wav_path
        self.json_path = json_path
        self.output_dir = output_dir
        self.task_type = task_type
        self.year = year
        self.valid_type = valid_type
        self.feature_type = feature_type
        self.hop_ratio = hop_ratio
        self.save_visualizations = save_visualizations
        self.test_2022_wav_path = test_2022_wav_path
        self.test_2022_json_path = test_2022_json_path
        self.test_2023_wav_path = test_2023_wav_path
        self.test_2023_json_path = test_2023_json_path

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set sampling rate and desired length based on task_type
        if self.task_type in [11, 12]:
            self.sr = 16000
            self.desired_length = 2  # Changed from 7.2 to match SPRSound.py
            self.frame_length = 512 if frame_length is None else frame_length
            self.hop_length = 64  # Fixed hop_length to match SPRSound.py
        else:
            self.sr = 8000
            self.desired_length = 16
            self.frame_length = 128 if frame_length is None else frame_length
            self.hop_length = 256  # Fixed hop_length to match SPRSound.py

        # Initialize FFT window
        self.fft_window = torch.hann_window(self.frame_length).numpy()

        # Initialize storage
        self.data = {
            'train': {'specs': [], 'labels': [], 'file_names': [], 'event_ids': []},
            'test_2022_intra': {'specs': [], 'labels': [], 'file_names': [], 'event_ids': []},
            'test_2022_inter': {'specs': [], 'labels': [], 'file_names': [], 'event_ids': []},
            'test_2023': {'specs': [], 'labels': [], 'file_names': [], 'event_ids': []},
            'test': {'specs': [], 'labels': [], 'file_names': [], 'event_ids': []}
        }

        # Initialize metadata storage
        self.metadata = []

    def _get_task_dict(self):
        """Get the task label mapping dictionary"""
        if self.task_type == 11:
            return {'Normal': 0,
                    'Rhonchi': 1,
                    'Wheeze': 1,
                    'Stridor': 1,
                    'Coarse Crackle': 1,
                    'Fine Crackle': 1,
                    'Wheeze+Crackle': 1}
        elif self.task_type == 12:
            return {'Normal': 0,
                    'Rhonchi': 1,
                    'Wheeze': 2,
                    'Stridor': 3,
                    'Coarse Crackle': 4,
                    'Fine Crackle': 5,
                    'Wheeze+Crackle': 6}
        elif self.task_type == 21:
            return {'Normal': 0,
                    'Poor Quality': 1,
                    'CAS': 2,
                    'DAS': 2,
                    'CAS & DAS': 2}
        elif self.task_type == 22:
            return {'Normal': 0,
                    'Poor Quality': 1,
                    'CAS': 2,
                    'DAS': 3,
                    'CAS & DAS': 4}

    def _sparse_json(self, file_name, level):
        """Parse JSON label file"""
        with open(file_name, 'r') as f:
            data = json.load(f)

        if level == "recording":
            record_label = data["record_annotation"]
            return record_label
        elif level == "event":
            events = data["event_annotation"]
            if events:
                event_labels = []
                starts = []
                ends = []
                for event in events:
                    label = event["type"]
                    if int(event['end']) > int(event['start']):
                        event_labels.append(label)
                        starts.append(int(event['start']))
                        ends.append(int(event['end']))
                return event_labels, starts, ends
            else:
                return [], [], []

    def signal_preprocessing(self, sig, fs):
        """Preprocess signal with bandpass filtering and resampling"""
        # Bandpass filtering
        nyquist = 0.5 * fs
        low = 50 / nyquist
        high = 2500 / nyquist
        b, a = signal.butter(5, [low, high], btype='band')
        sig = signal.filtfilt(b, a, sig)

        # Resample to target sampling rate
        sig = librosa.resample(sig, orig_sr=fs, target_sr=self.sr)

        return sig

    def normalize_audio(self, audio_signal):
        """Normalize audio signal"""
        if not isinstance(audio_signal, np.ndarray):
            audio_signal = np.array(audio_signal)

        audio_signal = audio_signal.astype(np.float32)
        max_abs = np.max(np.abs(audio_signal))
        if max_abs > 0:  # Avoid division by zero
            audio_signal = audio_signal / max_abs

        return audio_signal

    def cut_pad_sample(self, data):
        """Cut or pad audio to fixed length"""
        # Ensure input data is numpy array and convert to float32
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = data.astype(np.float32)

        # Calculate target length
        target_length = int(self.sr * self.desired_length)
        current_length = len(data)

        # If current length exceeds target length, cut from the beginning
        if current_length > target_length:
            return data[:target_length]

        # If length is insufficient, use repeat padding
        elif current_length < target_length:
            # Calculate repeat times
            repeat_times = target_length // current_length + 1
            # Repeat original data
            repeated_data = np.tile(data, repeat_times)
            # Cut to needed length
            padded_data = repeated_data[:target_length]

            # Apply fade in/out at connection points to avoid discontinuities
            fade_length = min(int(self.sr * 0.01),
                              target_length // 10)  # Use 1% of sample rate or 1/10 of data length for fade

            # Apply cross-fade at connection points
            for i in range(1, repeat_times):
                pos = i * current_length
                if pos < target_length:
                    end_pos = min(pos + fade_length, target_length)
                    start_pos = max(pos - fade_length, 0)
                    cross_fade_length = end_pos - start_pos

                    # Create fade in/out weights
                    fade_out = np.linspace(1, 0, cross_fade_length, dtype=np.float32)
                    fade_in = np.linspace(0, 1, cross_fade_length, dtype=np.float32)

                    # Apply cross-fade
                    padded_data[start_pos:end_pos] = (
                            padded_data[start_pos:end_pos] * fade_out +
                            repeated_data[start_pos:end_pos] * fade_in
                    )

            return padded_data

        # If length is exactly right, return as is
        return data

    def feature_extraction(self, sig, fs):
        """Extract features from audio signal"""
        if self.feature_type == 'MFCC':
            resamulated_audio_signal = librosa.resample(sig, orig_sr=fs, target_sr=16000)
            mfcc = librosa.feature.mfcc(y=resamulated_audio_signal, sr=fs, n_mfcc=40)
            return mfcc

        elif self.feature_type == 'log-mel':
            if self.task_type in [11, 12]:
                n_mels = 128
            else:
                n_mels = 32

            mel_spec = librosa.feature.melspectrogram(
                y=sig,
                sr=fs,
                n_fft=self.frame_length,
                win_length=self.frame_length,
                hop_length=self.hop_length,
                window=self.fft_window if hasattr(self, 'fft_window') else 'hann',
                n_mels=n_mels
            )

            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec

        elif self.feature_type == 'mel':
            mel_spec = librosa.feature.melspectrogram(
                y=sig,
                sr=fs,
                n_fft=self.frame_length,
                win_length=self.frame_length,
                window=self.fft_window if hasattr(self, 'fft_window') else 'hann',
                n_mels=256
            )
            return mel_spec

        elif self.feature_type == 'STFT':
            stft_spec = librosa.stft(
                sig,
                n_fft=self.frame_length,
                win_length=self.frame_length,
                window=self.fft_window if hasattr(self, 'fft_window') else 'hann'
            )
            stft_spec = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max)
            return stft_spec

    def process_training_data(self):
        """Process training data"""
        print("Processing training data...")
        task_dict = self._get_task_dict()
        json_files = os.listdir(self.json_path)

        for i in tqdm(json_files):
            if self.task_type in [11, 12]:
                event_labels, starts, ends = self._sparse_json(os.path.join(self.json_path, i), level="event")
                if not event_labels:
                    continue

                file_name = i.rsplit(".json", 1)[0]
                try:
                    # Load audio using librosa to ensure consistency
                    audio, fs = librosa.load(os.path.join(self.wav_path, file_name + '.wav'), sr=None)

                    for j in range(len(event_labels)):
                        # Extract segment
                        segment_sig = audio[int(starts[j] * fs / 1000): int(ends[j] * fs / 1000)]

                        # Preprocess signal
                        processed_sig = self.signal_preprocessing(segment_sig, fs)

                        # Normalize audio
                        normalized_sig = self.normalize_audio(processed_sig)

                        # Cut/pad to fixed length
                        padded_sig = self.cut_pad_sample(normalized_sig)

                        # Extract features
                        features = self.feature_extraction(padded_sig, self.sr)

                        # Get label
                        label = task_dict[event_labels[j]]

                        # Add to data storage
                        self.data['train']['specs'].append(features)
                        self.data['train']['labels'].append(label)
                        self.data['train']['file_names'].append(file_name)
                        self.data['train']['event_ids'].append(j)

                        # Add to metadata
                        self.metadata.append({
                            'file_name': file_name,
                            'event_id': j,
                            'label': label,
                            'label_text': event_labels[j],
                            'split': 'train',
                            'start_time': starts[j],
                            'end_time': ends[j],
                            'duration': (ends[j] - starts[j]) / 1000.0,  # seconds
                        })

                except Exception as e:
                    print(f"Error loading file {file_name}: {str(e)}")
                    continue

            elif self.task_type in [21, 22]:
                label_text = self._sparse_json(os.path.join(self.json_path, i), level="recording")
                file_name = i.rsplit(".json", 1)[0]

                try:
                    # Load audio using librosa
                    audio, fs = librosa.load(os.path.join(self.wav_path, file_name + '.wav'), sr=None)

                    # Preprocess signal
                    processed_sig = self.signal_preprocessing(audio, fs)

                    # Normalize audio
                    normalized_sig = self.normalize_audio(processed_sig)

                    # Cut/pad to fixed length
                    padded_sig = self.cut_pad_sample(normalized_sig)

                    # Extract features
                    features = self.feature_extraction(padded_sig, self.sr)

                    # Get label
                    label = task_dict[label_text]

                    # Add to data storage
                    self.data['train']['specs'].append(features)
                    self.data['train']['labels'].append(label)
                    self.data['train']['file_names'].append(file_name)
                    self.data['train']['event_ids'].append(0)  # No event ID for recording-level tasks

                    # Add to metadata
                    self.metadata.append({
                        'file_name': file_name,
                        'event_id': 0,
                        'label': label,
                        'label_text': label_text,
                        'split': 'train',
                        'start_time': 0,
                        'end_time': len(audio) / fs * 1000,  # Convert to ms
                        'duration': len(audio) / fs,  # seconds
                    })

                except Exception as e:
                    print(f"Error loading file {file_name}: {str(e)}")
                    continue

    def process_2022_test_data(self):
        """Process 2022 test data (intra and inter)"""
        if not self.test_2022_json_path or not self.test_2022_wav_path:
            print("Skipping 2022 test data processing since paths are not specified")
            return

        print("\nProcessing 2022 test data...")
        task_dict = self._get_task_dict()

        # Check directory structure - look for 2022 subdirectory
        base_2022_path = os.path.join(self.test_2022_json_path, "2022")
        if os.path.exists(base_2022_path) and os.path.isdir(base_2022_path):
            # Using the 2022 subdirectory
            json_base_path = base_2022_path
        else:
            # Using the root path
            json_base_path = self.test_2022_json_path

        # Process intra test data
        intra_json_path = os.path.join(json_base_path, "intra_test_json")
        if not os.path.exists(intra_json_path):
            # Try alternate path structure
            intra_json_path = os.path.join(self.test_2022_json_path, "intra_test_json")

        if os.path.exists(intra_json_path):
            print(f"Processing 2022 intra test data from {intra_json_path}...")
            json_files = [f for f in os.listdir(intra_json_path) if f.endswith('.json')]

            for i in tqdm(json_files):
                if self.task_type in [11, 12]:
                    event_labels, starts, ends = self._sparse_json(os.path.join(intra_json_path, i), level="event")
                    if not event_labels:
                        continue

                    file_name = i.rsplit(".json", 1)[0]
                    try:
                        # Find wav file - try multiple possible locations
                        wav_path = None
                        possible_paths = [
                            os.path.join(self.test_2022_wav_path, file_name + '.wav'),
                            os.path.join(self.test_2022_wav_path, "2022", file_name + '.wav'),
                            os.path.join(self.test_2022_wav_path, "restore_valid_classification_wav",
                                         file_name + '.wav')
                        ]

                        for path in possible_paths:
                            if os.path.exists(path):
                                wav_path = path
                                break

                        if not wav_path:
                            print(f"Could not find WAV file for {file_name}")
                            continue

                        # Load audio
                        audio, fs = librosa.load(wav_path, sr=None)

                        for j in range(len(event_labels)):
                            # Extract segment
                            segment_sig = audio[int(starts[j] * fs / 1000): int(ends[j] * fs / 1000)]

                            # Preprocess
                            processed_sig = self.signal_preprocessing(segment_sig, fs)
                            normalized_sig = self.normalize_audio(processed_sig)
                            padded_sig = self.cut_pad_sample(normalized_sig)

                            # Extract features
                            features = self.feature_extraction(padded_sig, self.sr)

                            # Get label
                            label = task_dict[event_labels[j]]

                            # Add to data storage
                            self.data['test_2022_intra']['specs'].append(features)
                            self.data['test_2022_intra']['labels'].append(label)
                            self.data['test_2022_intra']['file_names'].append(file_name)
                            self.data['test_2022_intra']['event_ids'].append(j)

                            # Also add to the general test set
                            self.data['test']['specs'].append(features)
                            self.data['test']['labels'].append(label)
                            self.data['test']['file_names'].append(file_name)
                            self.data['test']['event_ids'].append(j)

                            # Add to metadata
                            self.metadata.append({
                                'file_name': file_name,
                                'event_id': j,
                                'label': label,
                                'label_text': event_labels[j],
                                'split': 'test_2022_intra',
                                'start_time': starts[j],
                                'end_time': ends[j],
                                'duration': (ends[j] - starts[j]) / 1000.0,  # seconds
                                'year': 2022,
                                'test_type': 'intra'
                            })

                    except Exception as e:
                        print(f"Error loading 2022 intra test file {file_name}: {str(e)}")
                        continue

                elif self.task_type in [21, 22]:
                    label_text = self._sparse_json(os.path.join(intra_json_path, i), level="recording")
                    file_name = i.rsplit(".json", 1)[0]

                    try:
                        # For recording-level tasks, store metadata for later processing
                        label = task_dict[label_text]

                        # Add to metadata
                        self.metadata.append({
                            'file_name': file_name,
                            'event_id': 0,
                            'label': label,
                            'label_text': label_text,
                            'split': 'test_2022_intra',
                            'year': 2022,
                            'test_type': 'intra'
                        })

                    except Exception as e:
                        print(f"Error processing 2022 intra test metadata for {file_name}: {str(e)}")
                        continue

        # Process inter test data
        inter_json_path = os.path.join(json_base_path, "inter_test_json")
        if not os.path.exists(inter_json_path):
            # Try alternate path structure
            inter_json_path = os.path.join(self.test_2022_json_path, "inter_test_json")

        if os.path.exists(inter_json_path):
            print(f"Processing 2022 inter test data from {inter_json_path}...")
            json_files = [f for f in os.listdir(inter_json_path) if f.endswith('.json')]

            for i in tqdm(json_files):
                if self.task_type in [11, 12]:
                    event_labels, starts, ends = self._sparse_json(os.path.join(inter_json_path, i), level="event")
                    if not event_labels:
                        continue

                    file_name = i.rsplit(".json", 1)[0]
                    try:
                        # Find wav file - try multiple possible locations
                        wav_path = None
                        possible_paths = [
                            os.path.join(self.test_2022_wav_path, file_name + '.wav'),
                            os.path.join(self.test_2022_wav_path, "2022", file_name + '.wav'),
                            os.path.join(self.test_2022_wav_path, "restore_valid_classification_wav",
                                         file_name + '.wav')
                        ]

                        for path in possible_paths:
                            if os.path.exists(path):
                                wav_path = path
                                break

                        if not wav_path:
                            print(f"Could not find WAV file for {file_name}")
                            continue

                        # Load audio
                        audio, fs = librosa.load(wav_path, sr=None)

                        for j in range(len(event_labels)):
                            # Extract segment
                            segment_sig = audio[int(starts[j] * fs / 1000): int(ends[j] * fs / 1000)]

                            # Preprocess
                            processed_sig = self.signal_preprocessing(segment_sig, fs)
                            normalized_sig = self.normalize_audio(processed_sig)
                            padded_sig = self.cut_pad_sample(normalized_sig)

                            # Extract features
                            features = self.feature_extraction(padded_sig, self.sr)

                            # Get label
                            label = task_dict[event_labels[j]]

                            # Add to data storage
                            self.data['test_2022_inter']['specs'].append(features)
                            self.data['test_2022_inter']['labels'].append(label)
                            self.data['test_2022_inter']['file_names'].append(file_name)
                            self.data['test_2022_inter']['event_ids'].append(j)

                            # Also add to the general test set
                            self.data['test']['specs'].append(features)
                            self.data['test']['labels'].append(label)
                            self.data['test']['file_names'].append(file_name)
                            self.data['test']['event_ids'].append(j)

                            # Add to metadata
                            self.metadata.append({
                                'file_name': file_name,
                                'event_id': j,
                                'label': label,
                                'label_text': event_labels[j],
                                'split': 'test_2022_inter',
                                'start_time': starts[j],
                                'end_time': ends[j],
                                'duration': (ends[j] - starts[j]) / 1000.0,  # seconds
                                'year': 2022,
                                'test_type': 'inter'
                            })

                    except Exception as e:
                        print(f"Error loading 2022 inter test file {file_name}: {str(e)}")
                        continue

                elif self.task_type in [21, 22]:
                    label_text = self._sparse_json(os.path.join(inter_json_path, i), level="recording")
                    file_name = i.rsplit(".json", 1)[0]

                    try:
                        # For recording-level tasks, store metadata for later processing
                        label = task_dict[label_text]

                        # Add to metadata
                        self.metadata.append({
                            'file_name': file_name,
                            'event_id': 0,
                            'label': label,
                            'label_text': label_text,
                            'split': 'test_2022_inter',
                            'year': 2022,
                            'test_type': 'inter'
                        })

                    except Exception as e:
                        print(f"Error processing 2022 inter test metadata for {file_name}: {str(e)}")
                        continue

        intra_count = len([m for m in self.metadata if m['split'] == 'test_2022_intra'])
        inter_count = len([m for m in self.metadata if m['split'] == 'test_2022_inter'])
        print(f"Processed {intra_count} 2022 intra test samples and {inter_count} 2022 inter test samples")

    def process_2023_test_data(self):
        """Process 2023 test data"""
        if not self.test_2023_json_path or not self.test_2023_wav_path:
            print("Skipping 2023 test data processing since paths are not specified")
            return

        print("\nProcessing 2023 test data...")
        task_dict = self._get_task_dict()

        # Find the 2023 directory for JSON files
        json_2023_dir = os.path.join(self.test_2023_json_path, "2023")
        if not os.path.exists(json_2023_dir) or not os.path.isdir(json_2023_dir):
            print(f"Cannot find 2023 JSON directory at {json_2023_dir}")
            return

        # Find the 2023 directory for WAV files
        wav_2023_dir = os.path.join(self.test_2023_wav_path, "2023")
        if not os.path.exists(wav_2023_dir) or not os.path.isdir(wav_2023_dir):
            print(f"Cannot find 2023 WAV directory at {wav_2023_dir}")
            return

        # Get all JSON files in the 2023 directory
        json_files = []
        for file in os.listdir(json_2023_dir):
            if file.endswith('.json') and not os.path.isdir(os.path.join(json_2023_dir, file)):
                json_files.append(file)

        if not json_files:
            print(f"No JSON files found in {json_2023_dir}")
            return

        print(f"Found {len(json_files)} JSON files for 2023 test data")

        for json_file in tqdm(json_files):
            if self.task_type in [11, 12]:
                try:
                    json_path = os.path.join(json_2023_dir, json_file)
                    event_labels, starts, ends = self._sparse_json(json_path, level="event")
                    if not event_labels:
                        continue

                    # Get file base name without extension
                    file_name = json_file.replace('.json', '')

                    # Find corresponding WAV file
                    wav_path = os.path.join(wav_2023_dir, file_name + '.wav')

                    if not os.path.exists(wav_path):
                        print(f"Could not find WAV file for {file_name} at {wav_path}")
                        continue

                    # Load audio
                    audio, fs = librosa.load(wav_path, sr=None)

                    for j in range(len(event_labels)):
                        # Extract segment
                        segment_sig = audio[int(starts[j] * fs / 1000): int(ends[j] * fs / 1000)]

                        # Preprocess
                        processed_sig = self.signal_preprocessing(segment_sig, fs)
                        normalized_sig = self.normalize_audio(processed_sig)
                        padded_sig = self.cut_pad_sample(normalized_sig)

                        # Extract features
                        features = self.feature_extraction(padded_sig, self.sr)

                        # Get label
                        label = task_dict[event_labels[j]]

                        # Add to data storage
                        self.data['test_2023']['specs'].append(features)
                        self.data['test_2023']['labels'].append(label)
                        self.data['test_2023']['file_names'].append(file_name)
                        self.data['test_2023']['event_ids'].append(j)

                        # Also add to the general test set
                        self.data['test']['specs'].append(features)
                        self.data['test']['labels'].append(label)
                        self.data['test']['file_names'].append(file_name)
                        self.data['test']['event_ids'].append(j)

                        # Add to metadata
                        self.metadata.append({
                            'file_name': file_name,
                            'event_id': j,
                            'label': label,
                            'label_text': event_labels[j],
                            'split': 'test_2023',
                            'start_time': starts[j],
                            'end_time': ends[j],
                            'duration': (ends[j] - starts[j]) / 1000.0,  # seconds
                            'year': 2023,
                            'test_type': 'test'
                        })

                except Exception as e:
                    print(f"Error processing file {json_file}: {str(e)}")
                    continue

            elif self.task_type in [21, 22]:
                try:
                    json_path = os.path.join(json_2023_dir, json_file)
                    label_text = self._sparse_json(json_path, level="recording")
                    file_name = json_file.replace('.json', '')

                    # For recording-level tasks, store metadata for later processing
                    label = task_dict[label_text]

                    # Add to metadata
                    self.metadata.append({
                        'file_name': file_name,
                        'event_id': 0,
                        'label': label,
                        'label_text': label_text,
                        'split': 'test_2023',
                        'year': 2023,
                        'test_type': 'test'
                    })

                except Exception as e:
                    print(f"Error processing metadata for {json_file}: {str(e)}")
                    continue

        # Count processed samples
        test_2023_count = len([m for m in self.metadata if m['split'] == 'test_2023'])
        print(f"Processed {test_2023_count} 2023 test samples")

    def process_test_data(self):
        """Process test data for basic test data processing if no specific test paths provided"""
        print("\nProcessing basic test data...")
        task_dict = self._get_task_dict()

        # Check if test directory exists in json_path (for cases where specific test paths aren't provided)
        test_json_path = os.path.join(os.path.dirname(self.json_path), 'test_json')
        test_wav_path = os.path.join(os.path.dirname(self.wav_path), 'test_wav')

        if not os.path.exists(test_json_path) or not os.path.exists(test_wav_path):
            print(f"Skipping basic test data processing - test paths not found: {test_json_path}, {test_wav_path}")
            return

        json_files = os.listdir(test_json_path)

        for i in tqdm(json_files):
            if self.task_type in [11, 12]:
                event_labels, starts, ends = self._sparse_json(os.path.join(test_json_path, i), level="event")
                if not event_labels:
                    continue

                file_name = i.rsplit(".json", 1)[0]
                try:
                    # Load audio
                    audio, fs = librosa.load(os.path.join(test_wav_path, file_name + '.wav'), sr=None)

                    for j in range(len(event_labels)):
                        # Extract segment
                        segment_sig = audio[int(starts[j] * fs / 1000): int(ends[j] * fs / 1000)]

                        # Preprocess
                        processed_sig = self.signal_preprocessing(segment_sig, fs)
                        normalized_sig = self.normalize_audio(processed_sig)
                        padded_sig = self.cut_pad_sample(normalized_sig)

                        # Extract features
                        features = self.feature_extraction(padded_sig, self.sr)

                        # Get label
                        label = task_dict[event_labels[j]]

                        # Add to data storage
                        self.data['test']['specs'].append(features)
                        self.data['test']['labels'].append(label)
                        self.data['test']['file_names'].append(file_name)
                        self.data['test']['event_ids'].append(j)

                        # Add to metadata
                        self.metadata.append({
                            'file_name': file_name,
                            'event_id': j,
                            'label': label,
                            'label_text': event_labels[j],
                            'split': 'test',
                            'start_time': starts[j],
                            'end_time': ends[j],
                            'duration': (ends[j] - starts[j]) / 1000.0,  # seconds
                        })

                except Exception as e:
                    print(f"Error loading basic test file {file_name}: {str(e)}")
                    continue

            elif self.task_type in [21, 22]:
                label_text = self._sparse_json(os.path.join(test_json_path, i), level="recording")
                file_name = i.rsplit(".json", 1)[0]

                try:
                    # For recording-level tasks, store metadata for later processing
                    label = task_dict[label_text]

                    # Add to metadata
                    self.metadata.append({
                        'file_name': file_name,
                        'event_id': 0,
                        'label': label,
                        'label_text': label_text,
                        'split': 'test',
                    })

                except Exception as e:
                    print(f"Error processing basic test metadata for {file_name}: {str(e)}")
                    continue

        # Count processed samples
        test_count = len([m for m in self.metadata if m['split'] == 'test' and 'test_type' not in m])
        print(f"Processed {test_count} basic test samples")

    def process_recording_level_tasks(self):
        """Process recording-level tasks (21, 22) with audio loading"""
        if not self.task_type in [21, 22]:
            return

        # Filter metadata for recording-level tasks
        recording_metadata = [m for m in self.metadata if m['event_id'] == 0]

        print(f"Processing {len(recording_metadata)} recording-level files...")

        for meta in tqdm(recording_metadata):
            split = meta['split']
            file_name = meta['file_name']

            # Determine the correct path for the audio file
            audio_path = None

            if split == 'train':
                audio_path = os.path.join(self.wav_path, file_name + '.wav')
            elif split == 'test':
                test_path = os.path.join(self.wav_path)
                audio_path = os.path.join(
                    test_path.replace('test_json', 'test_wav'),
                    file_name + '.wav')
            elif split == 'test_2022_intra' or split == 'test_2022_inter':
                # For 2022 test data, check in the 2022 subdirectory
                audio_path = os.path.join(self.test_2022_wav_path, "2022", file_name + '.wav')

                # If not found, try the root directory
                if not os.path.exists(audio_path):
                    audio_path = os.path.join(self.test_2022_wav_path, file_name + '.wav')

            elif split == 'test_2023':
                # For 2023 test data, check in the 2023 subdirectory
                audio_path = os.path.join(self.test_2023_wav_path, "2023", file_name + '.wav')

            if not audio_path or not os.path.exists(audio_path):
                print(f"Skipping {file_name} - cannot find audio file")
                continue

            try:
                # Load audio
                audio, fs = librosa.load(audio_path, sr=None)

                # Record duration in metadata
                meta['duration'] = len(audio) / fs
                meta['end_time'] = len(audio) / fs * 1000  # Convert to ms

                # Preprocess
                processed_sig = self.signal_preprocessing(audio, fs)
                normalized_sig = self.normalize_audio(processed_sig)
                padded_sig = self.cut_pad_sample(normalized_sig)

                # Extract features
                features = self.feature_extraction(padded_sig, self.sr)

                # Add to data storage
                self.data[split]['specs'].append(features)
                self.data[split]['labels'].append(meta['label'])
                self.data[split]['file_names'].append(file_name)
                self.data[split]['event_ids'].append(0)

                # Also add to the general test set if it's a test split
                if split.startswith('test_'):
                    self.data['test']['specs'].append(features)
                    self.data['test']['labels'].append(meta['label'])
                    self.data['test']['file_names'].append(file_name)
                    self.data['test']['event_ids'].append(0)

            except Exception as e:
                print(f"Error processing audio for {file_name}: {str(e)}")
                continue


    def create_test_split_from_train(self, split_ratio=0.2):
        """Create a test split from the training data if no test data is provided"""
        if len(self.data['train']['specs']) == 0:
            print("Cannot create test split: Training data is empty")
            return

        if len(self.data['test']['specs']) > 0:
            print("Test split already exists, skipping creation")
            return

        print(f"\nCreating test split from training data with ratio {split_ratio}...")

        # Get all training data indices
        indices = list(range(len(self.data['train']['specs'])))

        # Shuffle indices
        np.random.shuffle(indices)

        # Calculate split point
        split_idx = int(len(indices) * (1 - split_ratio))

        # Indices for training and test sets
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        print(f"Split training data: {len(train_indices)} train, {len(test_indices)} test")

        # Create new test data from selected training samples
        for i in test_indices:
            self.data['test']['specs'].append(self.data['train']['specs'][i])
            self.data['test']['labels'].append(self.data['train']['labels'][i])
            self.data['test']['file_names'].append(self.data['train']['file_names'][i])
            self.data['test']['event_ids'].append(self.data['train']['event_ids'][i])

            # Update metadata
            for m in self.metadata:
                if (m['file_name'] == self.data['train']['file_names'][i] and
                        m['event_id'] == self.data['train']['event_ids'][i] and
                        m['split'] == 'train'):
                    # Create a copy of the metadata with updated split
                    test_metadata = m.copy()
                    test_metadata['split'] = 'test'
                    self.metadata.append(test_metadata)
                    break

        # Remove the selected samples from training data (optional)
        train_specs = []
        train_labels = []
        train_file_names = []
        train_event_ids = []

        for i in train_indices:
            train_specs.append(self.data['train']['specs'][i])
            train_labels.append(self.data['train']['labels'][i])
            train_file_names.append(self.data['train']['file_names'][i])
            train_event_ids.append(self.data['train']['event_ids'][i])

        self.data['train']['specs'] = train_specs
        self.data['train']['labels'] = train_labels
        self.data['train']['file_names'] = train_file_names
        self.data['train']['event_ids'] = train_event_ids

        # Update metadata
        self.metadata = [m for m in self.metadata if m['split'] != 'train' or
                         any(m['file_name'] == self.data['train']['file_names'][i] and
                             m['event_id'] == self.data['train']['event_ids'][i]
                             for i in range(len(self.data['train']['file_names'])))]

        print(
            f"Created test split. New sizes: {len(self.data['train']['specs'])} train, {len(self.data['test']['specs'])} test")


    def visualize_class_distribution(self, output_dir):
        """Visualize class distribution across different splits"""
        # Create a DataFrame from metadata
        df = pd.DataFrame(self.metadata)

        if df.empty:
            print("No data available for visualization")
            return

        class_names = get_class_names(self.task_type)

        # Plot class distribution
        plt.figure(figsize=(12, 8))

        # Count samples by split and label
        split_label_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)

        # Create a bar plot
        ax = split_label_counts.plot(kind='bar', stacked=False)

        # Customize the plot
        plt.title(f'Class Distribution - Task Type {self.task_type}')
        plt.xlabel('Dataset Split')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)

        # Set legend with class names
        if len(class_names) == split_label_counts.shape[1]:
            ax.legend([class_names[i] for i in range(len(class_names))])
        else:
            ax.legend([f"Class {i}" for i in split_label_counts.columns])

        # Add count labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, label_type='edge')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_distribution_task{self.task_type}.png'))
        plt.close()

        # Plot ratio of classes for each split
        plt.figure(figsize=(10, 6))

        # Calculate percentage
        split_label_percent = split_label_counts.div(split_label_counts.sum(axis=1), axis=0) * 100

        # Create a stacked bar plot
        ax = split_label_percent.plot(kind='bar', stacked=True)

        # Customize the plot
        plt.title(f'Class Distribution (%) - Task Type {self.task_type}')
        plt.xlabel('Dataset Split')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)

        # Set legend with class names
        if len(class_names) == split_label_percent.shape[1]:
            ax.legend([class_names[i] for i in range(len(class_names))])
        else:
            ax.legend([f"Class {i}" for i in split_label_percent.columns])

        # Add percentage labels in the middle of stacked bars
        for i, container in enumerate(ax.containers):
            # Only add labels for bars with significant height
            for j, val in enumerate(container):
                width = val.get_width()
                height = val.get_height()
                x = val.get_x() + width / 2
                y = val.get_y() + height / 2

                # Only show label if height is significant
                if height > 5:  # Show only if percentage is greater than 5%
                    ax.text(x, y, f'{split_label_percent.iloc[j, i]:.1f}%',
                            ha='center', va='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_distribution_percent_task{self.task_type}.png'))
        plt.close()


    def print_dataset_stats(self):
        """Print dataset statistics"""
        print("\n===== Dataset Statistics =====")

        # Create a DataFrame from metadata
        df = pd.DataFrame(self.metadata)

        # Count samples by split and label
        if not df.empty:
            split_label_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
            print("\nSample distribution by split and label:")
            print(split_label_counts)

            # Total counts by split
            print("\nTotal samples by split:")
            split_counts = df.groupby('split').size()
            print(split_counts)

            # Classes
            class_names = get_class_names(self.task_type)
            print(f"\nClasses for task type {self.task_type}:")
            for i, name in enumerate(class_names):
                print(f"  {i}: {name}")

            # Feature shape information
            for split in ['train', 'test', 'test_2022_intra', 'test_2022_inter', 'test_2023']:
                if self.data[split]['specs']:
                    print(f"\n{split} feature shape: {self.data[split]['specs'][0].shape}")
                    break

            # Additional stats for event-level tasks
            if self.task_type in [11, 12]:
                print("\nEvent duration statistics (seconds):")
                print(df['duration'].describe())
        else:
            print("No metadata available for statistics.")


    def save_sample_visualizations(self, n_samples=3):
        """Save sample visualizations"""
        if not self.save_visualizations:
            return

        print("Saving sample visualizations...")

        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, f"task{self.task_type}", "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Visualize each dataset split
        for split in ['train', 'test', 'test_2022_intra', 'test_2022_inter', 'test_2023']:
            if len(self.data[split]['specs']) == 0:
                continue

            # Select samples for each class
            specs = self.data[split]['specs']
            labels = self.data[split]['labels']

            # Try to get at least one sample per class
            unique_labels = np.unique(labels)
            indices = []

            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                if len(label_indices) > 0:
                    # Select one sample from each class
                    idx = np.random.choice(label_indices)
                    indices.append(idx)

            # If samples are fewer than n_samples, add more random samples
            while len(indices) < n_samples and len(specs) > len(indices):
                idx = np.random.randint(0, len(specs))
                if idx not in indices:
                    indices.append(idx)

            # Create visualization
            fig, axes = plt.subplots(len(indices), 1, figsize=(10, 4 * len(indices)))
            if len(indices) == 1:
                axes = [axes]  # Make sure axes is a list

            class_names = get_class_names(self.task_type)

            for i, idx in enumerate(indices):
                spec = specs[idx]
                label = labels[idx]
                label_text = class_names[label] if label < len(class_names) else f"Label {label}"

                # Plot log-mel spectrogram
                img = librosa.display.specshow(
                    spec,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    x_axis='time',
                    y_axis='mel',
                    ax=axes[i],
                    cmap='jet'
                )

                plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
                axes[i].set_title(f"{split} - {label_text}")

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'sample_specs_{split}.png'))
            plt.close()
            print(f"Saved visualization for {split} split")

        # Create class distribution visualization
        self.visualize_class_distribution(vis_dir)

    def save_processed_data(self):
        """Save processed data to NPY files and metadata to CSV"""
        print("Saving processed data...")

        # Create task-specific output directory
        task_output_dir = os.path.join(self.output_dir, f"task{self.task_type}")
        os.makedirs(task_output_dir, exist_ok=True)

        # For tasks 21 and 22, we need special handling to ensure exactly 1949 training samples
        if self.task_type in [21, 22]:
            # Count unique training files in metadata
            train_files = set()
            for item in self.metadata:
                if item['split'] == 'train':
                    train_files.add(item['file_name'])

            expected_train_count = len(train_files)
            print(f"Expected training count based on unique files: {expected_train_count}")

            # Create new training arrays with exactly the right number of samples
            # Group by filename to remove duplicates
            unique_train_specs = []
            unique_train_labels = []
            unique_train_filenames = []

            # Track seen filenames to avoid duplicates
            seen_filenames = set()

            for i in range(len(self.data['train']['file_names'])):
                filename = self.data['train']['file_names'][i]
                # Only include each filename once (first occurrence)
                if filename not in seen_filenames:
                    seen_filenames.add(filename)
                    unique_train_specs.append(self.data['train']['specs'][i])
                    unique_train_labels.append(self.data['train']['labels'][i])
                    unique_train_filenames.append(filename)

                    # Stop once we've reached the expected count
                    if len(unique_train_specs) >= expected_train_count:
                        break

            # Convert to numpy arrays
            train_specs = np.array(unique_train_specs)
            train_labels = np.array(unique_train_labels)

            # Save to NPY files
            np.save(os.path.join(task_output_dir, "train_specs.npy"), train_specs)
            np.save(os.path.join(task_output_dir, "train_labels.npy"), train_labels)

            print(f"Saved train data: {train_specs.shape} specs, {train_labels.shape} labels")
            print(
                f"Removed {len(self.data['train']['specs']) - len(train_specs)} duplicate/test samples from training data")

        # For tasks 11 and 12, use standard approach
        else:
            # Save train data
            if len(self.data['train']['specs']) > 0:
                # Convert to numpy arrays
                specs = np.array(self.data['train']['specs'])
                labels = np.array(self.data['train']['labels'])

                # Save to NPY files
                np.save(os.path.join(task_output_dir, "train_specs.npy"), specs)
                np.save(os.path.join(task_output_dir, "train_labels.npy"), labels)

                print(f"Saved train data: {specs.shape} specs, {labels.shape} labels")

        # Save test data (unchanged)
        if len(self.data['test']['specs']) > 0:
            # Convert to numpy arrays
            specs = np.array(self.data['test']['specs'])
            labels = np.array(self.data['test']['labels'])

            # Save to NPY files
            np.save(os.path.join(task_output_dir, "test_specs.npy"), specs)
            np.save(os.path.join(task_output_dir, "test_labels.npy"), labels)

            print(f"Saved test data: {specs.shape} specs, {labels.shape} labels")

        # Save specific test datasets
        for test_set in ['test_2022_intra', 'test_2022_inter', 'test_2023']:
            if len(self.data[test_set]['specs']) > 0:
                # Convert to numpy arrays
                specs = np.array(self.data[test_set]['specs'])
                labels = np.array(self.data[test_set]['labels'])

                # Save to NPY files
                np.save(os.path.join(task_output_dir, f"{test_set}_specs.npy"), specs)
                np.save(os.path.join(task_output_dir, f"{test_set}_labels.npy"), labels)

                print(f"Saved {test_set} data: {specs.shape} specs, {labels.shape} labels")

        # Save metadata to CSV
        df = pd.DataFrame(self.metadata)
        if not df.empty:
            df.to_csv(os.path.join(task_output_dir, "metadata.csv"), index=False)
            print(f"Saved metadata with {len(df)} entries")

        # Save dataset info
        info = {
            'task_type': self.task_type,
            'feature_type': self.feature_type,
            'sample_rate': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'desired_length': self.desired_length,
            'class_names': get_class_names(self.task_type)
        }

        # Save as JSON
        with open(os.path.join(task_output_dir, "dataset_info.json"), 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Processing complete. Data saved to {task_output_dir}")


    def process_dataset(self, create_test_split=False, split_ratio=0.2):
        """Process the entire dataset"""
        # Process training data
        self.process_training_data()

        # Process test data from all available sources
        if self.test_2022_json_path and self.test_2022_wav_path:
            self.process_2022_test_data()

        if self.test_2023_json_path and self.test_2023_wav_path:
            self.process_2023_test_data()

        # Fallback to basic test data processing if no specific test paths provided
        if (not self.test_2022_json_path or not self.test_2022_wav_path) and \
                (not self.test_2023_json_path or not self.test_2023_wav_path):
            self.process_test_data()

        # Process recording-level tasks if applicable
        self.process_recording_level_tasks()

        # Create test split if requested and no test data exists
        if create_test_split and len(self.data['test']['specs']) == 0:
            self.create_test_split_from_train(split_ratio)

        # Print statistics
        self.print_dataset_stats()

        # Save sample visualizations
        self.save_sample_visualizations()

        # Save processed data
        self.save_processed_data()

        return self.data


# Example usage
if __name__ == "__main__":
    # Task 11: Binary event classification
    # processor_11 = SPRSoundDatasetProcessor(
    #     wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_train_classification_wav",
    #     json_path="/home/p2412918/Lung_sound_detection/Classification/train_classification_json",
    #     test_2022_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
    #     test_2022_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
    #     test_2023_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
    #     test_2023_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
    #     output_dir="datasets/SPRSound_2022+2023_processed_data",
    #     task_type=11,
    #     feature_type="log-mel"
    # )
    # processor_11.process_dataset()
    #
    # # Task 12: Multiclass event classification
    # processor_12 = SPRSoundDatasetProcessor(
    #     wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_train_classification_wav",
    #     json_path="/home/p2412918/Lung_sound_detection/Classification/train_classification_json",
    #     test_2022_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
    #     test_2022_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
    #     test_2023_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
    #     test_2023_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
    #     output_dir="datasets/SPRSound_2022+2023_processed_data",
    #     task_type=12,
    #     feature_type="log-mel"
    # )
    # processor_12.process_dataset()

    # Task 21: Binary recording classification
    processor_21 = SPRSoundDatasetProcessor(
        wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_train_classification_wav",
        json_path="/home/p2412918/Lung_sound_detection/Classification/train_classification_json",
        test_2022_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
        test_2022_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
        test_2023_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
        test_2023_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
        output_dir="datasets/SPRSound_2022+2023_processed_data",
        task_type=21,
        feature_type="log-mel"
    )
    processor_21.process_dataset()

    # Task 22: Multiclass recording classification
    processor_22 = SPRSoundDatasetProcessor(
        wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_train_classification_wav",
        json_path="/home/p2412918/Lung_sound_detection/Classification/train_classification_json",
        test_2022_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
        test_2022_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
        test_2023_wav_path="/home/p2412918/Lung_sound_detection/Classification/restore_valid_classification_wav",
        test_2023_json_path="/home/p2412918/Lung_sound_detection/Classification/valid_classification_json",
        output_dir="datasets/SPRSound_2022+2023_processed_data",
        task_type=22,
        feature_type="log-mel"
    )
    processor_22.process_dataset()