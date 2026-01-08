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
        """Corruption 数据集构造函数"""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        self.name = dataset_name  # 保存名称供评估使用
        DatasetClass = DATASET_REGISTRY[dataset_name]
        self.data_dir = Path(root) / f"{DatasetClass.NAME}-C"

        # 1. 基础初始化
        self._verify_installation(dataset_name)
        self._init_statistics(DatasetClass)
        self._load_labels()

    def _verify_installation(self, dataset_name):
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
        self.labels = torch.from_numpy(np.load(str(labels_path))).long()

    def get_loader(
        self, corruption_type: str, severity: int, config: "Config"
    ) -> DataLoader:
        """获取特定损坏类型和严重程度的数据加载器"""

        # 1. 解析目标类型
        target_types = self._resolve_types(corruption_type)

        # 2. 收集数据批次
        all_data, all_labels = [], []
        total_samples = len(self.labels)

        for c_type in target_types:
            # 加载图像数据
            data = self._load_corruption(c_type, severity)

            # 判断是否为完整数据 - 如果数据大小等于 labels 大小，说明是 Full 模式
            # 如果数据大小小于 labels，说明是 Balanced Slicing 模式，需要切分 labels
            if len(data) == total_samples:
                # Full Mode
                current_labels = self.labels
            else:
                # Balanced Mode: 计算该 corruption 对应的 slice
                slice_obj = self._get_slice_indices(c_type, total_samples)
                current_labels = self.labels[slice_obj]

                # 简单校验
                if len(data) != len(current_labels):
                    # 容错: 如果切片计算有细微误差 (e.g. 尾部处理), 尝试以数据长度为准
                    # 这通常不应该发生，除非 generate.py 逻辑变更
                    get_logger().warning(
                        f"Data/Label Mismatch for {c_type}: Data={len(data)}, LabelSlice={len(current_labels)}. "
                        "Using data length."
                    )
                    # 重新对齐: 如果数据少，就再切 label；如果数据多(不可能)，就报错
                    current_labels = current_labels[: len(data)]

            all_data.append(data)
            all_labels.append(current_labels)

        # 3. 组装 DataLoader
        return self._prepare_dataloader(
            torch.cat(all_data, dim=0), torch.cat(all_labels, dim=0), config
        )

    def _get_slice_indices(self, corruption_type: str, total_samples: int) -> slice:
        """根据 Corruption 类型和总样本数，计算其在 Balanced 模式下的数据切片"""
        import math

        # 找到该 corruption 所属的 category 及其索引
        found_cat, found_list = None, None
        for cat, c_list in self.CATEGORIES.items():
            if corruption_type in c_list:
                found_cat = cat
                found_list = c_list
                break

        if not found_cat:
            # Should not happen if _resolve_types works
            return slice(0, total_samples)

        num_types = len(found_list)
        chunk_size = math.ceil(total_samples / num_types)
        idx_in_cat = found_list.index(corruption_type)

        start_idx = idx_in_cat * chunk_size
        end_idx = min((idx_in_cat + 1) * chunk_size, total_samples)

        # 边界修正 (同 generate.py)
        if start_idx >= total_samples:
            start_idx = total_samples
            end_idx = total_samples

        return slice(start_idx, end_idx)

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
        # 1. 读取对应严重程度的数据切片
        images_np = self._read_npy_slice(corruption_type, severity)

        # 2. 转换为标准化的 Tensor
        return self._postprocess_tensor(images_np)

    def _read_npy_slice(self, corruption_type: str, severity: int) -> np.ndarray:
        """执行具体的二进制文件读取和切片计算"""
        if severity not in self.SEVERITIES:
            raise ValueError(f"Severity 必须在 {self.SEVERITIES} 中, 得到 {severity}")

        file_path = self.data_dir / f"{corruption_type}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到 corruption 文件: {file_path}")

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
