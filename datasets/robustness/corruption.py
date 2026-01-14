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

from ...utils import get_logger
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
    """Corruption 评估数据集 (Full Coverage 模式)

    从预生成的 .npy 文件加载 corruption 数据。
    使用 `python -m ensemble.datasets.robustness.generate` 预生成数据。

    本类依赖于 `metadata.json` 文件，该文件由生成脚本自动创建。
    """

    # 引用模块级常量 (保留兼容性)
    CORRUPTIONS = CORRUPTIONS
    CATEGORIES = CORRUPTION_CATEGORIES
    SEVERITIES = SEVERITIES

    def __init__(self, dataset_name: str, root: str = "./data"):
        """Corruption 数据集构造函数"""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        self.name = dataset_name
        DatasetClass = DATASET_REGISTRY[dataset_name]
        self.data_dir = Path(root) / f"{DatasetClass.NAME}-C"

        # 1. 基础初始化
        get_logger().info(f"📥 准备加载 Corruption 数据: {self.data_dir}...")
        self._verify_installation()
        self._init_statistics(DatasetClass)
        self._load_labels()

        get_logger().info("✅ Corruption 数据集准备就绪 (Full Coverage)")

    def _verify_installation(self):
        """确保预生成数据包已安装"""
        labels_path = self.data_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"未找到预生成数据: {labels_path}")

    def _init_statistics(self, DatasetClass):
        """初始化数据集统计信息"""
        self.mean = torch.tensor(DatasetClass.MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(DatasetClass.STD).view(1, 3, 1, 1)

    def _load_labels(self):
        """加载标签文件"""
        labels_path = self.data_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        self.labels = torch.from_numpy(np.load(str(labels_path))).long()

    def get_loader(
        self, corruption_type: str, severity: int, config: "Config"
    ) -> DataLoader:
        """获取特定损坏类型和严重程度的数据加载器"""

        # 1. 解析目标类型
        target_types = self._resolve_types(corruption_type)

        # 2. 收集数据批次
        all_data, all_labels = [], []

        # 全量模式下，每个 corruption 对应的都是完整的 labels
        current_labels = self.labels

        for c_type in target_types:
            # 加载图像数据
            data = self._load_corruption(c_type, severity)

            # 校验数据长度
            if len(data) != len(current_labels):
                raise ValueError(
                    f"Data length mismatch for {c_type}: data={len(data)}, labels={len(current_labels)}"
                )

            all_data.append(data)
            all_labels.append(current_labels)

        # 3. 组装 DataLoader
        return self._prepare_dataloader(
            torch.cat(all_data, dim=0), torch.cat(all_labels, dim=0), config
        )

    def _resolve_types(self, corruption_type: str) -> list:
        """解析输入的类型名称(单类或具体类型)"""
        if corruption_type in self.CATEGORIES:
            return self.CATEGORIES[corruption_type]
        if corruption_type in self.CORRUPTIONS:
            return [corruption_type]
        raise ValueError(f"未知 Corruption 类型: {corruption_type}")

    def _prepare_dataloader(self, data, labels, config) -> DataLoader:
        """整理并创建 DataLoader"""
        dataset = TensorDataset(data, labels)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    def _load_corruption(self, corruption_type: str, severity: int) -> torch.Tensor:
        """从预生成文件加载单个 corruption 类型的数据"""
        # 1. 构造文件名
        filename = f"{corruption_type}.npy"

        # 2. 读取对应严重程度的数据切片
        images_np = self._read_npy_slice(filename, severity)

        # 3. 转换为标准化的 Tensor
        return self._postprocess_tensor(images_np)

    def _read_npy_slice(self, filename: str, severity: int) -> np.ndarray:
        """执行具体的二进制文件读取和切片计算"""
        if severity not in self.SEVERITIES:
            raise ValueError(f"Severity 必须在 {self.SEVERITIES} 中, 得到 {severity}")

        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"未找到数据文件: {file_path}")

        # 使用 mmap 模式读取，避免一次性加载过大文件 (视情况而定，这里先 load 进来)
        # 注意: 如果文件巨大，建议改用 mmap_mode='r' 并小心处理
        data = np.load(str(file_path))

        # 计算切片索引 (假设数据按 severity 排序堆叠)
        total_records = data.shape[0]
        n_sev = len(self.SEVERITIES)
        n_samples = total_records // n_sev

        if total_records % n_sev != 0:
            raise ValueError(f"数据格式错误: {total_records} 无法被 {n_sev} 整除")

        # 定位并切片
        sev_idx = self.SEVERITIES.index(severity)
        return data[sev_idx * n_samples : (sev_idx + 1) * n_samples]

    def _postprocess_tensor(self, images_np: np.ndarray) -> torch.Tensor:
        """将 numpy 图像阵列转换为标准化 PyTorch 张量"""
        # [N, H, W, 3] -> [N, 3, H, W]
        images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
        return (images_tensor - self.mean) / self.std
