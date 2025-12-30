"""
================================================================================
Corruption 数据集模块
================================================================================

包含: CorruptionDataset, CORRUPTIONS 常量
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config.core import Config

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..preloaded import DATASET_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 全局常量定义                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 全局常量定义                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 完整 19 种 Corruption，分为 4 大类
# 参考: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
CORRUPTION_CATEGORIES = {
    "noise": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",  # Extra
    ],
    "blur": [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",  # Extra
    ],
    "weather": [
        "snow",
        "frost",
        "fog",
        "brightness",
        "spatter",  # Extra
    ],
    "digital": [
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "saturate",  # Extra
    ],
}

# 扁平化列表，方便遍历
CORRUPTIONS = [c for cat in CORRUPTION_CATEGORIES.values() for c in cat]

# 3种严重程度 (或者完整 1-5，这里保持 1, 3, 5 以节省空间，或者改为 range(1, 6))
# 既然用户想要更标准的评估，这里扩展为完整的 1-5
SEVERITIES = [1, 3, 5]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Corruption数据集                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CorruptionDataset:
    """Corruption 评估数据集 (仅支持预生成模式)

    从预生成的 .npy 文件加载 corruption 数据。
    使用 `python -m ensemble.datasets.robustness.generate` 预生成数据。

    支持加载单一类型 (如 'gaussian_noise') 或 整个大类 (如 'noise')。

    使用示例:
        >>> dataset = CorruptionDataset.from_name("cifar10", "./data")
        >>> loader = dataset.get_loader("noise", severity=3, config=config)
    """

    # 引用模块级常量
    CORRUPTIONS = CORRUPTIONS
    CATEGORIES = CORRUPTION_CATEGORIES
    SEVERITIES = SEVERITIES

    def __init__(self, dataset_name: str, root: str = "./data"):
        """Corruption 数据集构造函数

        Args:
            dataset_name: 数据集名称 (如 "cifar10", "eurosat")
            root: 数据根目录
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        DatasetClass = DATASET_REGISTRY[dataset_name]
        self.data_dir = Path(root) / f"{DatasetClass.NAME}-C"

        labels_path = self.data_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"未找到预生成数据: {labels_path}\n"
                f"请先运行: python -m ensemble.datasets.robustness.generate --type corruption --dataset {dataset_name}"
            )

        self.labels = torch.from_numpy(np.load(str(labels_path))).long()
        self.mean = torch.tensor(DatasetClass.MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(DatasetClass.STD).view(1, 3, 1, 1)

    def get_loader(
        self,
        corruption_type: str,
        severity: int,
        config: "Config",
    ) -> DataLoader:
        """获取特定损坏类型和严重程度的数据加载器

        Args:
            corruption_type: 可以是具体的损坏名 (如 'fog')，也可以是大类名 (如 'weather')
            severity: 严重程度 (1-5)
            config: 全局配置对象
        """
        # 1. 确定要加载哪些具体类型
        if corruption_type in self.CATEGORIES:
            # 如果是大类，加载该类下所有子类型
            target_types = self.CATEGORIES[corruption_type]
        elif corruption_type in self.CORRUPTIONS:
            # 如果是具体类型
            target_types = [corruption_type]
        else:
            raise ValueError(
                f"未知 Corruption 类型或类别: {corruption_type}. "
                f"可用类别: {list(self.CATEGORIES.keys())}, "
                f"可用具体类型: {self.CORRUPTIONS}"
            )

        # 2. 加载并合并数据
        all_data = []
        all_labels = []

        # 注意：如果加载多个类型，标签需要重复多次
        # 预生成的 labels 对应原始测试集的一次完整拷贝 (或采样后的子集)
        # 每个 corruption type 的数据文件都与 self.labels 一一对应

        for c_type in target_types:
            data = self._load_corruption(c_type, severity)
            # 处理 labels 切片: 如果 data 比 labels 短 (samples_per_category case)
            current_n = data.size(0)
            if current_n < len(self.labels):
                labels = self.labels[:current_n]
            else:
                labels = self.labels

            all_data.append(data)
            all_labels.append(labels)

        # 合并 (N_types * N_samples, C, H, W)
        combined_data = torch.cat(all_data, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)

        dataset = TensorDataset(combined_data, combined_labels)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,  # 测试集通常不需要 shuffle，但在混合模式下 shuffle 可能有帮助？(对于BN评估) 一般 evaluation 不 shuffle
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    def _load_corruption(self, corruption_type: str, severity: int) -> torch.Tensor:
        """从预生成文件加载单个 corruption 类型的数据"""
        if severity not in self.SEVERITIES:
            raise ValueError(
                f"Severity must be one of {self.SEVERITIES}, got {severity}"
            )

        file_path = self.data_dir / f"{corruption_type}.npy"
        if not file_path.exists():
            raise FileNotFoundError(
                f"未找到 corruption 文件: {file_path}\n"
                f"请尝试重新生成数据: python -m ensemble.datasets.robustness.generate --type corruption"
            )

        # 加载数据: shape = (N_severities * N_samples, H, W, 3)
        # 注意：现在的生成脚本可能会支持只生成特定 severity 或所有 severity。
        # 假设生成脚本仍然生成所有 severity 并在第0维堆叠。
        try:
            data = np.load(str(file_path))
        except Exception as e:
            raise RuntimeError(f"无法读取文件 {file_path}: {e}")

        # 计算样本数 (总数 / severity 数量)
        total_records = data.shape[0]
        n_severities = len(self.SEVERITIES)
        if total_records % n_severities != 0:
            # 简单的完整性检查
            raise ValueError(
                f"数据文件 {file_path} 损坏或格式不符。总行数 {total_records} 不能被 {n_severities} 整除。"
            )

        n_samples = total_records // n_severities

        # 校验 n_samples 是否与 labels 长度一致
        # (如果生成时使用了抽样，labels 长度也就是抽样后的长度)
        # 如果 n_samples < len(self.labels)，说明是 variable sampling (samples_per_category)，
        # 我们假设使用的是前 n_samples 个标签 (因为 generate.py 也是取 [:current_n])
        if n_samples > len(self.labels):
            raise ValueError(
                f"数据不一致: {corruption_type} 包含 {n_samples} 个样本 (per severity), "
                f"但 labels 文件仅包含 {len(self.labels)} 个样本。"
            )

        # 如果 n_samples < len(self.labels), 我们只返回部分数据
        # 注意: 这里只返回了 data,labels 的切片将在 get_loader 中处理?
        # 不, _load_corruption 返回 data tensor。
        # get_loader 组装 TensorDataset(data, labels)。
        # 如果 data 只有 250 个, labels 有 1000 个 -> Mismatch!
        # 所以 get_loader 需要知道 data 的长度并切片 labels。

        # 定位数据片段
        # 数据通常按 severity 升序排列: [sev1_data, sev2_data, ...]
        severity_idx = self.SEVERITIES.index(severity)
        start_idx = severity_idx * n_samples
        end_idx = (severity_idx + 1) * n_samples

        # 读取实际数据 (这一步会触发磁盘IO并加载进内存)
        images = data[start_idx:end_idx]

        # 转换为 Tensor 并归一化
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        return (images_tensor - self.mean) / self.std

    # 需要修改 get_loader 来适配变长 labels
