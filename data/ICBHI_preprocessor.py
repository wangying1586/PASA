import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ICBHIDatasetProcessor:
    """
    ICBHI 2017 肺音数据集处理类
    实现数据集加载、预处理和频谱特征提取
    """

    def __init__(self,
                 data_dir,
                 split_file,
                 sample_rate=4000,
                 desired_length=8,
                 n_mels=64,
                 n_fft=256,
                 hop_length=128,
                 f_max=2000,
                 task_type='binary'):  # 'binary' 或 'multiclass'
        """
        初始化数据处理器

        参数:
            data_dir: 存放音频文件的目录
            split_file: 训练/测试集划分文件路径
            sample_rate: 采样率
            desired_length: 所需的音频长度(秒)
            n_mels: 梅尔频谱图的梅尔滤波器数
            n_fft: FFT窗口大小
            hop_length: 帧移
            f_max: 最大频率
            task_type: 分类任务类型, 'binary'表示二分类, 'multiclass'表示四分类
        """
        self.data_dir = data_dir
        self.split_file = split_file
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_max = f_max
        self.task_type = task_type

        # 读取训练/测试集划分
        self.train_files, self.test_files = self._read_split_file()

        # 初始化存储数据集信息
        self.cycles_info = []  # 存储所有信息
        self.data = {
            'train': {'specs': [], 'labels': [], 'patient_ids': [], 'cycle_ids': []},
            'test': {'specs': [], 'labels': [], 'patient_ids': [], 'cycle_ids': []}
        }

    def _read_split_file(self):
        """读取训练/测试集划分文件并验证文件存在性"""
        train_files = []
        test_files = []

        with open(self.split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    file_name, split = parts

                    # 检查音频文件是否存在
                    audio_file = os.path.join(self.data_dir, f"{file_name}.wav")
                    if not os.path.exists(audio_file):
                        # print(f"警告: 音频文件不存在，跳过: {file_name}")
                        continue

                    if split.lower() == 'train':
                        train_files.append(file_name)
                    elif split.lower() == 'test':
                        test_files.append(file_name)

        print(f"读取到 {len(train_files)} 个训练文件和 {len(test_files)} 个测试文件")
        return train_files, test_files

    def _read_annotation_file(self, file_name):
        """读取注释文件"""
        base_name = file_name.split('.')[0]
        annotation_file = os.path.join(self.data_dir, f"{base_name}.txt")

        cycles = []

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            # 获取起止时间和标签
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            crackles = '1' in parts[2]
                            wheezes = '1' in parts[3] if len(parts) > 3 else False

                            # 确定标签
                            if self.task_type == 'binary':
                                # 二分类: 0-正常, 1-异常(有爆裂音或哮鸣音)
                                label = 1 if crackles or wheezes else 0
                            else:
                                # 四分类: 0-正常, 1-爆裂音, 2-哮鸣音, 3-两者都有
                                if crackles and wheezes:
                                    label = 3  # 既有爆裂音又有哮鸣音
                                elif crackles:
                                    label = 1  # 只有爆裂音
                                elif wheezes:
                                    label = 2  # 只有哮鸣音
                                else:
                                    label = 0  # 正常

                            cycles.append({
                                'start': start_time,
                                'end': end_time,
                                'label': label
                            })
        except FileNotFoundError:
            print(f"注释文件未找到: {annotation_file}")
            return []

        return cycles

    def _load_and_preprocess_audio(self, file_path, start_time, end_time):
        """加载和预处理单个音频片段，使用复制填充并确保正确记录填充信息"""
        try:
            # 加载完整音频文件
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # 提取所需的呼吸周期
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            if start_sample >= len(audio) or end_sample > len(audio):
                print(f"警告: 时间范围超出音频长度 - {file_path}, {start_time}-{end_time}")
                return None

            cycle_audio = audio[start_sample:end_sample]

            # 标准化为固定长度
            desired_samples = self.desired_length * sr

            # 记录是否进行了填充
            self.padding_applied = False

            if len(cycle_audio) < desired_samples:
                # 使用复制填充而不是零填充
                self.padding_applied = True
                padding_factor = desired_samples / len(cycle_audio)

                # 策略1: 直接复制重复直到达到所需长度
                # repetitions = int(np.ceil(desired_samples / len(cycle_audio)))
                # padded_audio = np.tile(cycle_audio, repetitions)[:desired_samples]

                # 策略2: 使用镜像反射填充，避免硬边界
                padded_audio = np.zeros(desired_samples)
                curr_pos = 0

                while curr_pos < desired_samples:
                    # 如果是偶数次填充，正向复制
                    if (curr_pos // len(cycle_audio)) % 2 == 0:
                        seg = cycle_audio
                    # 如果是奇数次填充，反向复制以避免硬边界
                    else:
                        seg = cycle_audio[::-1]

                    # 计算可以复制的样本数
                    copy_len = min(len(seg), desired_samples - curr_pos)
                    padded_audio[curr_pos:curr_pos + copy_len] = seg[:copy_len]
                    curr_pos += copy_len

                cycle_audio = padded_audio

                # 记录填充信息，用于后续分析或可视化
                self.padding_ratio = padding_factor

            elif len(cycle_audio) > desired_samples:
                # 如果音频片段太长，截取中间部分而不是开始部分
                # 这样可以避免截断可能包含重要信息的结尾部分
                extra = len(cycle_audio) - desired_samples
                start_idx = extra // 2  # 从中间开始截取
                cycle_audio = cycle_audio[start_idx:start_idx + desired_samples]

            return cycle_audio

        except Exception as e:
            print(f"处理音频时出错 {file_path}: {e}")
            return None

    def _extract_mel_spectrogram(self, audio):
        """提取梅尔频谱图特征，并确保正确返回log-mel频谱"""
        try:
            # 计算梅尔频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmax=self.f_max
            )

            # 转换为对数刻度 (这一步创建log-mel频谱)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # 保存原始形状，用于后续检查填充
            original_shape = log_mel_spec.shape

            # 归一化到[0,1]范围，但保留动态范围特征
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

            return log_mel_spec

        except Exception as e:
            print(f"提取梅尔频谱图时出错: {e}")
            return None

    def process_dataset(self, save_samples=True):
        """处理整个数据集，并可选择保存原始和增强的样本可视化"""
        print("开始处理ICBHI数据集...")

        # 处理训练集
        print("处理训练集...")
        for file_name in tqdm(self.train_files):
            self._process_file(file_name, 'train')

        # 处理测试集
        print("处理测试集...")
        for file_name in tqdm(self.test_files):
            self._process_file(file_name, 'test')

        # 转换为numpy数组
        for split in ['train', 'test']:
            if self.data[split]['specs']:
                self.data[split]['specs'] = np.array(self.data[split]['specs'])
                self.data[split]['labels'] = np.array(self.data[split]['labels'])

        # 打印数据集统计信息
        self._print_dataset_stats()

        # 保存样本可视化
        if save_samples:
            try:
                self._save_sample_visualizations()
            except Exception as e:
                print(f"样本可视化保存失败: {e}")

        return self.data

    def _save_sample_visualizations(self, n_samples=3):
        """保存原始音频和频谱图的可视化比较"""
        for split in ['train', 'test']:
            if len(self.data[split]['specs']) == 0:
                continue

            # 为每个类别选择样本
            specs = self.data[split]['specs']
            labels = self.data[split]['labels']

            # 每个类别选择至少一个样本
            unique_labels = np.unique(labels)
            indices = []

            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                if len(label_indices) > 0:
                    # 从每个类别随机选择一个样本
                    idx = np.random.choice(label_indices)
                    indices.append(idx)

            # 如果样本数量不足n_samples，随机添加更多
            while len(indices) < n_samples and len(indices) < len(specs):
                idx = np.random.randint(0, len(specs))
                if idx not in indices:
                    indices.append(idx)

            # 创建可视化
            fig, axes = plt.subplots(len(indices), 1, figsize=(12, 4 * len(indices)))
            if len(indices) == 1:
                axes = [axes]  # 确保axes是列表

            for i, idx in enumerate(indices):
                spec = specs[idx]
                label = labels[idx]

                # 确定标签文本
                if self.task_type == 'binary':
                    label_text = "正常" if label == 0 else "异常"
                else:
                    label_texts = ["正常", "爆裂音", "哮鸣音", "爆裂音和哮鸣音"]
                    label_text = label_texts[label]

                # 绘制log-mel频谱图
                img = librosa.display.specshow(
                    spec,
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    x_axis='time',
                    y_axis='mel',
                    fmax=self.f_max,
                    ax=axes[i]
                )

                # 添加颜色条
                plt.colorbar(img, ax=axes[i], format='%+2.0f dB')

                # 检查是否有明显的零填充区域
                # 计算频谱图右侧区域的平均能量
                right_energy = np.mean(spec[:, -int(spec.shape[1] / 5):])
                left_energy = np.mean(spec[:, :int(spec.shape[1] / 5)])
                has_zeros = right_energy < 0.05 * left_energy  # 如果右侧能量远低于左侧，可能是零填充

                padding_info = " (疑似零填充)" if has_zeros else ""
                axes[i].set_title(f"{split} - {label_text}{padding_info}")

            plt.tight_layout()
            plt.savefig(f"log_mel_samples_{split}_{self.task_type}.png")
            plt.close()

    def _process_file(self, file_name, split):
        """处理单个音频文件"""
        # 检查音频文件是否存在
        file_path = os.path.join(self.data_dir, f"{file_name}.wav")
        if not os.path.exists(file_path):
            print(f"跳过文件 {file_name} - 音频文件不存在")
            return

        # 读取注释
        cycles = self._read_annotation_file(file_name)

        if not cycles:
            print(f"跳过文件 {file_name} - 没有找到有效的注释")
            return

        # 提取患者ID
        patient_id = file_name.split('_')[0]

        # 处理每个呼吸周期
        for i, cycle in enumerate(cycles):
            # 加载和预处理音频
            audio = self._load_and_preprocess_audio(
                file_path,
                cycle['start'],
                cycle['end']
            )

            if audio is None:
                continue

            # 提取梅尔频谱图
            mel_spec = self._extract_mel_spectrogram(audio)

            if mel_spec is None:
                continue

            # 添加到数据集
            self.data[split]['specs'].append(mel_spec)
            self.data[split]['labels'].append(cycle['label'])
            self.data[split]['patient_ids'].append(patient_id)
            self.data[split]['cycle_ids'].append(i)

            # 添加到循环信息
            self.cycles_info.append({
                'file_name': file_name,
                'cycle_id': i,
                'patient_id': patient_id,
                'label': cycle['label'],
                'split': split,
                'start_time': cycle['start'],
                'end_time': cycle['end']
            })

    def _print_dataset_stats(self):
        """打印数据集统计信息"""
        print("\n===== 数据集统计信息 =====")

        # 创建一个DataFrame
        df = pd.DataFrame(self.cycles_info)

        # 按split和label统计样本数
        print("\n各集合中的样本分布:")
        split_label_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
        print(split_label_counts)

        # 计算每个集合的总数
        print("\n总样本数:")
        split_counts = df.groupby('split').size()
        print(split_counts)

        # 患者统计
        print("\n患者统计:")
        patient_split_counts = df.groupby(['split'])['patient_id'].nunique()
        print(f"训练集中的患者数量: {patient_split_counts.get('train', 0)}")
        print(f"测试集中的患者数量: {patient_split_counts.get('test', 0)}")

        # 如果是二分类任务，添加正常/异常的统计
        if self.task_type == 'binary':
            print("\n二分类标签分布:")
            binary_label_names = {0: "正常", 1: "异常"}
            binary_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
            for split in binary_counts.index:
                normal = binary_counts.loc[split, 0] if 0 in binary_counts.columns else 0
                abnormal = binary_counts.loc[split, 1] if 1 in binary_counts.columns else 0
                print(f"{split}: 正常 = {normal}, 异常 = {abnormal}, 比例 = {abnormal / (normal + abnormal):.2f}")

        # 如果是四分类任务，添加类别统计
        else:
            print("\n四分类标签分布:")
            multiclass_label_names = {0: "正常", 1: "爆裂音", 2: "哮鸣音", 3: "两者都有"}
            multiclass_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
            for split in multiclass_counts.index:
                for label in range(4):
                    count = multiclass_counts.loc[split, label] if label in multiclass_counts.columns else 0
                    print(f"{split} - {multiclass_label_names[label]}: {count}")

    def save_processed_data(self, output_dir):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存处理后的数据
        for split in ['train', 'test']:
            np.save(os.path.join(output_dir, f"{split}_specs.npy"), self.data[split]['specs'])
            np.save(os.path.join(output_dir, f"{split}_labels.npy"), self.data[split]['labels'])

        # 保存循环信息
        df = pd.DataFrame(self.cycles_info)
        df.to_csv(os.path.join(output_dir, "cycles_info.csv"), index=False)

        print(f"处理后的数据已保存到 {output_dir}")

    def visualize_samples(self, n_samples=2, random_seed=42):
        """可视化数据集中的样本，确保正确显示log-mel频谱特性"""
        np.random.seed(random_seed)

        fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))

        for i, split in enumerate(['train', 'test']):
            if len(self.data[split]['specs']) == 0:
                print(f"没有{split}集的样本可供可视化")
                continue

            # 随机选择样本
            n_available = len(self.data[split]['specs'])
            indices = np.random.choice(n_available, min(n_samples, n_available), replace=False)

            for j, idx in enumerate(indices):
                spec = self.data[split]['specs'][idx]
                label = self.data[split]['labels'][idx]

                # 二分类或四分类的标签文本
                if self.task_type == 'binary':
                    label_text = "Normal" if label == 0 else "Abnormal"
                else:
                    label_texts = ["Normal", "Crackle", "Wheeze", "Wheeze + Crackle"]
                    label_text = label_texts[label]

                # 绘制log-mel频谱图，确保正确设置参数
                img = librosa.display.specshow(
                    spec,
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    x_axis='time',
                    y_axis='mel',  # 确保使用'mel'选项
                    fmax=self.f_max,
                    cmap='viridis',  # 使用更适合log-mel的颜色映射
                    ax=axes[i, j]
                )

                # 添加颜色条显示动态范围
                plt.colorbar(img, ax=axes[i, j], format='%+2.0f dB')

                # 检查是否有明显的零填充区域
                has_zeros = np.any(spec[:, -5:] < 0.01)  # 检查最后几列是否全为接近0的值
                padding_info = " (有零填充)" if has_zeros else ""

                axes[i, j].set_title(f"{split} - {label_text}{padding_info}")

        plt.tight_layout()

        # 保存并返回图像
        return fig


# 数据加载示例
def load_dataset_example():
    """使用示例"""
    # 设置路径
    data_dir = "/path/to/ICBHI_dataset"  # 包含.wav和.txt文件的目录
    split_file = "/path/to/ICBHI_challenge_train_test.txt"  # 提供的训练/测试分割文件
    output_dir = "/path/to/processed_data"  # 处理后数据的保存目录

    # 创建处理器实例进行数据集完整性检查
    checker = ICBHIDatasetProcessor(
        data_dir=data_dir,
        split_file=split_file,
        task_type='binary'
    )

    # 二分类任务
    processor_binary = ICBHIDatasetProcessor(
        data_dir=data_dir,
        split_file=split_file,
        task_type='binary'
    )

    # 处理数据集
    data_binary = processor_binary.process_dataset()

    # 保存处理后的数据
    processor_binary.save_processed_data(os.path.join(output_dir, 'binary'))

    # 可视化一些样本 - 添加异常处理
    try:
        fig = processor_binary.visualize_samples(n_samples=3)
        plt.savefig(os.path.join(output_dir, 'binary_samples.png'))
    except Exception as e:
        print(f"可视化样本时出错: {e}")

    # 四分类任务
    processor_multiclass = ICBHIDatasetProcessor(
        data_dir=data_dir,
        split_file=split_file,
        task_type='multiclass'
    )

    # 处理数据集
    data_multiclass = processor_multiclass.process_dataset()

    # 保存处理后的数据
    processor_multiclass.save_processed_data(os.path.join(output_dir, 'multiclass'))

    return data_binary, data_multiclass


# 使用处理好的数据集进行训练的示例
def training_example():
    """基于处理好的数据训练一个简单模型的示例"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # 加载预处理好的数据
    processed_data_dir = "/path/to/processed_data/binary"  # 处理后的数据目录

    # 加载数据
    X_train = np.load(os.path.join(processed_data_dir, "train_specs.npy"))
    y_train = np.load(os.path.join(processed_data_dir, "train_labels.npy"))
    X_test = np.load(os.path.join(processed_data_dir, "test_specs.npy"))
    y_test = np.load(os.path.join(processed_data_dir, "test_labels.npy"))

    # 增加通道维度 (N, H, W) -> (N, 1, H, W)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义一个简单的CNN模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=2):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

            # 计算全连接层的输入特征数
            # 假设输入是 (1, 64, 313)，经过两次池化后变为 (32, 16, 78)
            self.fc_input_size = 32 * 16 * 78

            self.fc1 = nn.Linear(self.fc_input_size, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, self.fc_input_size)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # 实例化模型
    model = SimpleCNN(num_classes=2)  # 二分类

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    # 设置路径
    data_dir = "/home/p2412918/Benchmark_Codebase_Construction/datasets/datasets/ICBHI_final_database"  # ICBHI数据集目录
    split_file = "/home/p2412918/Benchmark_Codebase_Construction/datasets/datasets/ICBHI_final_database/ICBHI_challenge_train_test.txt"  # 训练测试集划分文件
    output_dir = "datasets/ICBHI2017_processed_data"  # 输出目录

    # 创建处理器实例进行数据集完整性检查
    checker = ICBHIDatasetProcessor(
        data_dir=data_dir,
        split_file=split_file,
        task_type='binary'
    )

    # 二分类任务
    processor_binary = ICBHIDatasetProcessor(
        data_dir=data_dir,
        split_file=split_file,
        task_type='binary'
    )

    # 处理数据集
    data_binary = processor_binary.process_dataset()

    # 保存处理后的数据
    processor_binary.save_processed_data(os.path.join(output_dir, 'binary'))

    # 四分类任务
    processor_multiclass = ICBHIDatasetProcessor(
        data_dir=data_dir,
        split_file=split_file,
        task_type='multiclass'
    )

    # 处理数据集
    data_multiclass = processor_multiclass.process_dataset()

    # 保存处理后的数据
    processor_multiclass.save_processed_data(os.path.join(output_dir, 'multiclass'))