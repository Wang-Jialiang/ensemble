"""
================================================================================
类别校准追踪器
================================================================================

追踪每个类别的置信度与准确率偏差，用于类别自适应数据增强 (CADA)。

基于论文: "Improving Calibration of BatchEnsemble with Data Augmentation"
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class CalibrationTracker:
    """类别级校准追踪器。

    追踪每个类别的:
        - 平均置信度 (Confidence): 模型对该类别预测的平均最大 softmax 概率
        - 准确率 (Accuracy): 该类别的实际正确率
        - 置信度偏差 (Δc = Confidence - Accuracy): 正值表示过自信，负值表示欠自信

    Attributes:
        num_classes: 类别数量
        _class_confidence: 每类累计置信度
        _class_correct: 每类正确预测数
        _class_total: 每类样本总数
        _calibration_bias: 最近一次计算的置信度偏差
    """

    def __init__(self, num_classes: int):
        """初始化校准追踪器。

        Args:
            num_classes: 数据集类别数量
        """
        self.num_classes = num_classes
        self._reset_accumulators()
        self._calibration_bias: Optional[torch.Tensor] = None

    def _reset_accumulators(self):
        """重置累加器。"""
        self._class_confidence = torch.zeros(self.num_classes)
        self._class_correct = torch.zeros(self.num_classes)
        self._class_total = torch.zeros(self.num_classes)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """更新类别统计（单批次）。

        Args:
            logits: 模型输出 logits，形状 [B, num_classes]
            targets: 真实标签，形状 [B]
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)

            # 移到 CPU 进行累加
            preds = preds.cpu()
            targets = targets.cpu()
            max_probs = max_probs.cpu()

            for c in range(self.num_classes):
                mask = targets == c
                count = mask.sum().item()
                if count > 0:
                    self._class_total[c] += count
                    self._class_correct[c] += ((preds == targets) & mask).sum().item()
                    self._class_confidence[c] += max_probs[mask].sum().item()

    def compute_bias(self) -> torch.Tensor:
        """计算每个类别的置信度偏差。

        Returns:
            torch.Tensor: 形状 [num_classes] 的置信度偏差
                - 正值: 过自信 (Confidence > Accuracy)，需要增强
                - 负值: 欠自信 (Confidence < Accuracy)，需要保护
                - 0: 校准良好
        """
        # 避免除零
        valid_mask = self._class_total > 0

        # 计算平均置信度和准确率
        avg_confidence = torch.zeros(self.num_classes)
        accuracy = torch.zeros(self.num_classes)

        avg_confidence[valid_mask] = (
            self._class_confidence[valid_mask] / self._class_total[valid_mask]
        )
        accuracy[valid_mask] = (
            self._class_correct[valid_mask] / self._class_total[valid_mask]
        )

        # 置信度偏差 = Confidence - Accuracy
        self._calibration_bias = avg_confidence - accuracy

        # 无样本的类别偏差设为 0（中性）
        self._calibration_bias[~valid_mask] = 0.0

        return self._calibration_bias

    def get_adaptive_prob(
        self,
        class_idx: int,
        base_prob: float = 0.8,
        prob_range: Tuple[float, float] = (0.0, 0.8),
        sensitivity: float = 2.0,
    ) -> float:
        """获取特定类别的自适应增强概率。

        根据置信度偏差动态调整:
            - 过自信类: 提高概率
            - 欠自信类: 降低概率

        Args:
            class_idx: 类别索引
            base_prob: 基础触发概率
            prob_range: 概率范围 (min, max)
            sensitivity: 偏差敏感度 (越大响应越强)

        Returns:
            float: 调整后的概率
        """
        if self._calibration_bias is None:
            return base_prob

        bias = self._calibration_bias[class_idx].item()

        # 使用 tanh 映射到 [-1, 1] 范围，sensitivity 控制响应曲线陡峭程度
        adjustment = torch.tanh(torch.tensor(bias * sensitivity)).item()

        # 计算自适应概率
        # adjustment > 0 (过自信): 增大 prob
        # adjustment < 0 (欠自信): 减小 prob
        prob_min, prob_max = prob_range

        if adjustment >= 0:
            adapted_prob = base_prob + adjustment * (prob_max - base_prob)
        else:
            adapted_prob = base_prob + adjustment * (base_prob - prob_min)

        # 限制在有效范围内
        return max(prob_min, min(prob_max, adapted_prob))

    def get_all_adaptive_probs(
        self,
        base_prob: float = 0.8,
        prob_range: Tuple[float, float] = (0.0, 0.8),
        sensitivity: float = 2.0,
    ) -> List[float]:
        """获取所有类别的自适应概率。

        Returns:
            List[float]: 每个类别的触发概率
        """
        return [
            self.get_adaptive_prob(c, base_prob, prob_range, sensitivity)
            for c in range(self.num_classes)
        ]

    def reset(self):
        """重置追踪器状态（每个 epoch 开始时调用）。"""
        self._reset_accumulators()

    def get_summary(self) -> Dict[str, torch.Tensor]:
        """获取校准摘要统计。

        Returns:
            Dict: 包含 avg_confidence, accuracy, bias 的字典
        """
        valid_mask = self._class_total > 0

        avg_confidence = torch.zeros(self.num_classes)
        accuracy = torch.zeros(self.num_classes)

        avg_confidence[valid_mask] = (
            self._class_confidence[valid_mask] / self._class_total[valid_mask]
        )
        accuracy[valid_mask] = (
            self._class_correct[valid_mask] / self._class_total[valid_mask]
        )

        return {
            "avg_confidence": avg_confidence,
            "accuracy": accuracy,
            "bias": self._calibration_bias
            if self._calibration_bias is not None
            else torch.zeros(self.num_classes),
            "sample_count": self._class_total,
        }

    def log_status(self, logger=None, top_k: int = 5):
        """记录校准状态日志。

        Args:
            logger: 日志记录器 (可选，默认 print)
            top_k: 显示最过自信/欠自信的 K 个类别
        """
        if self._calibration_bias is None:
            return

        log_fn = logger.info if logger else print
        summary = self.get_summary()

        # 找出最过自信和最欠自信的类别
        bias = summary["bias"]
        sorted_indices = torch.argsort(bias, descending=True)

        log_fn("=" * 60)
        log_fn("类别校准状态")
        log_fn("=" * 60)
        log_fn(f"{'类别':^8} | {'置信度':^10} | {'准确率':^10} | {'偏差':^10}")
        log_fn("-" * 60)

        # 最过自信的类别
        log_fn("【过自信 Top-K】")
        for i in range(min(top_k, len(sorted_indices))):
            c = sorted_indices[i].item()
            if summary["sample_count"][c] > 0:
                log_fn(
                    f"{c:^8} | {summary['avg_confidence'][c]:^10.4f} | "
                    f"{summary['accuracy'][c]:^10.4f} | {bias[c]:^+10.4f}"
                )

        # 最欠自信的类别
        log_fn("【欠自信 Top-K】")
        for i in range(min(top_k, len(sorted_indices))):
            c = sorted_indices[-(i + 1)].item()
            if summary["sample_count"][c] > 0:
                log_fn(
                    f"{c:^8} | {summary['avg_confidence'][c]:^10.4f} | "
                    f"{summary['accuracy'][c]:^10.4f} | {bias[c]:^+10.4f}"
                )

        log_fn("=" * 60)
