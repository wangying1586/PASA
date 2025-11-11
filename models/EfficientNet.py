from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from feature_extractor.HarmonicBridge import HarmonicBridge
import torch.nn.init as init
from efficientnet_pytorch.utils import load_pretrained_weights
from efficientnet_pytorch.model import EfficientNet

class CustomEfficientNet(EfficientNet):
    def __init__(self, blocks_args, global_params, num_classes=None):
        super(CustomEfficientNet, self).__init__(blocks_args, global_params)
        # 初始化自定义的WTConv2d模块，这里假设输入通道数根据你的实际输入数据为1，输出通道数设为与原EfficientNet第一层卷积输入通道数匹配（比如3，具体需确认）
        self.custom_conv = HarmonicBridge(in_channels=1, out_channels=1)
        # 重新初始化custom_conv中的权重（调用已有的初始化权重方法，你可以根据实际需求进一步完善这个方法里的逻辑）
        self._reinitialize_wtconv2d_weights()
        self.num_classes = num_classes

        # 获取模型的第一层卷积层（对于EfficientNet-b0来说，_conv_stem是第一层卷积层，不同版本可能有不同命名，需确认）
        self.original_conv_stem = self._conv_stem
        # 获取当前第一层卷积层的权重数据，它的形状通常是 [out_channels, in_channels, kernel_size, kernel_size]
        weight_data =  self.original_conv_stem.weight.data
        # 原第一层卷积层的输入通道数
        original_in_channels = weight_data.shape[1]
        # 你的数据现在的通道数（通道数为1）
        new_in_channels = 1
        # 如果输入通道数发生了变化，需要调整第一层卷积层的相关参数
        # 情况一：如果新的通道数小于原通道数，可以选择保留部分通道对应的权重（这里简单示例只取前new_in_channels个通道对应的权重）
        if new_in_channels < original_in_channels:
            weight_data = weight_data[:, :new_in_channels, :, :]
        # 情况二：如果新的通道数大于原通道数，需要扩充权重数据（比如通过复制、随机初始化等方式扩充，这里简单示例用随机初始化扩充权重）
        elif new_in_channels > original_in_channels:
            additional_channels_weight = torch.randn(weight_data.shape[0], new_in_channels - original_in_channels,
                                                     weight_data.shape[2], weight_data.shape[3])
            weight_data = torch.cat([weight_data, additional_channels_weight], dim=1)
        # 更新第一层卷积层的权重数据
        self.original_conv_stem.weight.data = weight_data

        # 同时需要更新偏置项（如果有），这里假设原偏置项维度和原输入通道数有关，要根据新通道数进行调整
        if  self.original_conv_stem.bias is not None:
            bias_data =  self.original_conv_stem.bias.data
            if new_in_channels < original_in_channels:
                bias_data = bias_data[:new_in_channels]
            elif new_in_channels > original_in_channels:
                additional_bias = torch.zeros(new_in_channels - original_in_channels)
                bias_data = torch.cat([bias_data, additional_bias], dim=0)
            self.original_conv_stem.bias.data = bias_data

        # 用一个Sequential容器将custom_conv和原_conv_stem按顺序组合起来，形成新的第一层
        self._conv_stem = nn.Sequential(
            self.custom_conv,
            self.original_conv_stem
        )

        if num_classes is not None:
            num_ftrs = self._fc.in_features
            # 重新初始化全连接层权重，不使用预训练权重里的_fc层权重
            self._fc = nn.Linear(num_ftrs, num_classes)
            init.kaiming_normal_(self._fc.weight, mode='fan_out', nonlinearity='relu')
            if self._fc.bias is not None:
                self._fc.bias.data.zero_()

    def _reinitialize_wtconv2d_weights(self):
        """
        方法功能：重新初始化WTConv2d模块（self.custom_conv）里的所有可学习权重参数
        """
        # 初始化 _conv_stem 或 custom_conv 中的 wt_filter 和 iwt_filter（这里它们是固定参数，一般不需要额外初始化，仅赋值）
        if hasattr(self, '_conv_stem') and hasattr(self._conv_stem, 'wt_filter'):
            self._conv_stem.wt_filter = nn.Parameter(self._conv_stem.wt_filter, requires_grad=False)
            self._conv_stem.iwt_filter = nn.Parameter(self._conv_stem.iwt_filter, requires_grad=False)
        if hasattr(self, 'custom_conv') and hasattr(self.custom_conv, 'wt_filter'):
            self.custom_conv.wt_filter = nn.Parameter(self.custom_conv.wt_filter, requires_grad=False)
            self.custom_conv.iwt_filter = nn.Parameter(self.custom_conv.iwt_filter, requires_grad=False)

        # 初始化 _conv_stem 或 custom_conv 中的基础卷积层权重
        if hasattr(self, '_conv_stem') and hasattr(self._conv_stem, 'base_conv'):
            init.kaiming_normal_(self._conv_stem.base_conv.weight, mode='fan_out', nonlinearity='relu')
            if self._conv_stem.base_conv.bias is not None:
                self._conv_stem.base_conv.bias.data.zero_()
        if hasattr(self, 'custom_conv') and hasattr(self.custom_conv, 'base_conv'):
            init.kaiming_normal_(self.custom_conv.base_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.custom_conv.base_conv.bias is not None:
                self.custom_conv.base_conv.bias.data.zero_()

        # 初始化 _conv_stem 或 custom_conv 中的基础缩放模块权重
        if hasattr(self, '_conv_stem') and hasattr(self._conv_stem, 'base_scale'):
            init.ones_(self._conv_stem.base_scale.weight)
        if hasattr(self, 'custom_conv') and hasattr(self.custom_conv, 'base_scale'):
            init.ones_(self.custom_conv.base_scale.weight)

        # 初始化 _conv_stem 或 custom_conv 中的小波域卷积层权重
        if hasattr(self, '_conv_stem') and hasattr(self._conv_stem, 'wavelet_convs'):
            for wavelet_conv in self._conv_stem.wavelet_convs:
                init.kaiming_normal_(wavelet_conv.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(self, 'custom_conv') and hasattr(self.custom_conv, 'wavelet_convs'):
            for wavelet_conv in self.custom_conv.wavelet_convs:
                init.kaiming_normal_(wavelet_conv.weight, mode='fan_out', nonlinearity='relu')

        # 初始化 _conv_stem 或 custom_conv 中的小波域缩放模块权重
        if hasattr(self, '_conv_stem') and hasattr(self._conv_stem, 'wavelet_scale'):
            for wavelet_scale in self._conv_stem.wavelet_scale:
                init.ones_(wavelet_scale.weight)
        if hasattr(self, 'custom_conv') and hasattr(self.custom_conv, 'wavelet_scale'):
            for wavelet_scale in self.custom_conv.wavelet_scale:
                init.ones_(wavelet_scale.weight)

        # 初始化 _conv_stem 或 custom_conv 中的归一化层权重
        if hasattr(self, '_conv_stem') and hasattr(self._conv_stem, 'gn'):
            if isinstance(self._conv_stem.gn, nn.GroupNorm):
                pass  # nn.GroupNorm默认初始化方式已满足需求，无需额外操作
            else:
                init.ones_(self._conv_stem.gn.weight)
                init.zeros_(self._conv_stem.gn.bias)
        if hasattr(self, 'custom_conv') and hasattr(self.custom_conv, 'gn'):
            if isinstance(self.custom_conv.gn, nn.GroupNorm):
                pass  # nn.GroupNorm默认初始化方式已满足需求，无需额外操作
            else:
                init.ones_(self.custom_conv.gn.weight)
                init.zeros_(self.custom_conv.gn.bias)

        # 初始化全连接层权重（如果存在）
        if hasattr(self, '_fc'):
            init.kaiming_normal_(self._fc.weight, mode='fan_out', nonlinearity='relu')
            if self._fc.bias is not None:
                self._fc.bias.data.zero_()

    def __repr__(self):
        """
        重写__repr__方法，定制模型打印输出内容，排除custom_conv和original_conv_stem相关信息
        """
        main_modules = []
        for name, module in self.named_children():
            if name not in ["custom_conv", "original_conv_stem"]:
                main_modules.append(f"{name}: {module}")
        return f"CustomEfficientNet(\n    " + ",\n    ".join(main_modules) + "\n)"

    def forward(self, x):
        x = super().forward(x)
        return x

class CustomEfficientNetWithLoad(CustomEfficientNet):
    @classmethod
    def from_pretrained(cls, model_name, num_classes=None, **override_params):
        # 先按照原始方式创建模型实例，但不加载权重
        model = super().from_name(model_name, num_classes=num_classes, **override_params)

        # 手动处理缺失键情况，模拟strict=False的效果
        weights_path = f'/home/p2412918/Lung_sound_detection/feature_extractor/efficientnet-b4-6ed6700e.pth'
        pretrained_dict = torch.load(weights_path)
        model_dict = model.state_dict()
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and not k.startswith('_fc'):  # 这里修改，只添加不以'_fc'开头的键值对，彻底排除全连接层相关键
                updated_pretrained_dict[k] = v

        # 加载处理后的权重字典到模型
        model.load_state_dict(updated_pretrained_dict, strict=False)

        return model

def load_efficientnet_model(num_classes=None):

    model = EfficientNet.from_pretrained('efficientnet-b4')

    # 获取模型的第一层卷积层（对于EfficientNet-b0来说，_conv_stem是第一层卷积层，不同版本可能有不同命名，需确认）
    first_conv_layer = model._conv_stem
    # 获取当前第一层卷积层的权重数据，它的形状通常是 [out_channels, in_channels, kernel_size, kernel_size]
    weight_data = first_conv_layer.weight.data
    # 原第一层卷积层的输入通道数（这里假设原输入通道数是3，对于常见图像输入情况，需要根据实际情况确认）
    original_in_channels = weight_data.shape[1]
    # 你的数据现在的通道数（这里根据你前面提到的经过处理后的数据维度是[32, 1, 20, 40]，通道数为1）
    new_in_channels = 1
    # 如果输入通道数发生了变化，需要调整第一层卷积层的相关参数
    # 情况一：如果新的通道数小于原通道数，可以选择保留部分通道对应的权重（这里简单示例只取前new_in_channels个通道对应的权重）
    if new_in_channels < original_in_channels:
        weight_data = weight_data[:, :new_in_channels, :, :]
    # 情况二：如果新的通道数大于原通道数，需要扩充权重数据（比如通过复制、随机初始化等方式扩充，这里简单示例用随机初始化扩充权重）
    elif new_in_channels > original_in_channels:
        additional_channels_weight = torch.randn(weight_data.shape[0], new_in_channels - original_in_channels,
                                                 weight_data.shape[2], weight_data.shape[3])
        weight_data = torch.cat([weight_data, additional_channels_weight], dim=1)
    # 更新第一层卷积层的权重数据
    first_conv_layer.weight.data = weight_data

    # 同时需要更新偏置项（如果有），这里假设原偏置项维度和原输入通道数有关，要根据新通道数进行调整
    if first_conv_layer.bias is not None:
        bias_data = first_conv_layer.bias.data
        if new_in_channels < original_in_channels:
            bias_data = bias_data[:new_in_channels]
        elif new_in_channels > original_in_channels:
            additional_bias = torch.zeros(new_in_channels - original_in_channels)
            bias_data = torch.cat([bias_data, additional_bias], dim=0)
        first_conv_layer.bias.data = bias_data

    # 调整全连接层（根据你原来代码中修改全连接层的逻辑，这里假设保持不变，适配任务类别数量）
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)
    return model

if __name__ == '__main__':
    # 假设的分类任务类别数
    num_classes = [2, 7, 3, 5]  # 例如：2, 7, 3, 5 分类任务
    for num_classe in num_classes:
        # 加载预训练的EfficientNet-b4模型，并调整第一层卷积层以适应新的输入通道数
        model = CustomEfficientNetWithLoad.from_pretrained('efficientnet-b4', num_classes=num_classe)
        model_state_dict_keys = model.state_dict().keys()
        # print("Model state dict keys:", model_state_dict_keys)
        # print("After loading weights, _conv_stem.weight shape:", model._conv_stem[1].weight.shape)
        print(model)

        # 设置模型为评估模式
        model.eval()

        # 随机生成一批音频数据作为输入
        # 假设音频数据经过预处理后的形状为 [batch_size, channels, height, width]
        # 这里我们使用随机数据模拟，batch_size=4, channels=1, height=20, width=40
        input_data = torch.randn(2, 1, 128, 1001)  # 模拟单通道音频数据

        # 添加一个批次维度并确保数据在正确的设备上（CPU或GPU）
        # input_data = input_data.unsqueeze(0)  # 添加批次维度

        # 前向传播
        with torch.no_grad():  # 关闭梯度计算
            outputs = model(input_data)

        # 打印输出的形状
        print("Output shape:", outputs.shape)