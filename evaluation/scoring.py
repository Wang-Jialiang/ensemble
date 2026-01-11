"""
================================================================================
评分系统模块
================================================================================

包含: ScoreCalculator - 多维度评分计算器
"""

import math
from typing import Any, Dict, List, Tuple

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 维度配置                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 各维度权重 (总和 = 1.0)
DIMENSION_WEIGHTS = {
    "accuracy": 0.20,  # 准确度
    "calibration": 0.10,  # 校准性
    "diversity": 0.15,  # 多样性 (含 distance_matrix)
    "fairness": 0.10,  # 公平性
    "corruption": 0.15,  # Corruption 鲁棒性
    "ood": 0.10,  # OOD 检测
    "adversarial": 0.10,  # 对抗鲁棒性
    "interpretability": 0.10,  # 可解释性 (GradCAM)
}

# 评分等级阈值
GRADE_THRESHOLDS = [
    (90, "S"),
    (80, "A"),
    (70, "B"),
    (60, "C"),
    (0, "D"),
]

# 各维度的指标配置: (指标名, 是否越高越好, 参考范围 [min, max])
DIMENSION_METRICS = {
    "accuracy": [
        ("ensemble_acc", True, (0, 100)),
        ("oracle_acc", True, (0, 100)),
        ("balanced_acc", True, (0, 100)),
    ],
    "calibration": [
        ("ece", False, (0, 0.5)),  # ECE 越低越好, 典型范围 0-0.5
        ("nll", False, (0, 5)),  # NLL 越低越好, 典型范围 0-5
    ],
    "diversity": [
        ("disagreement", True, (0, 50)),  # 分歧度，高更好
        ("cka_diversity", True, (0, 1)),  # CKA 多样性，高更好
        ("avg_distance", True, (0, 500)),  # 平均模型距离，高更好
        ("std_distance", True, (0, 100)),  # 距离标准差，高=有离群模型
        ("direction_diversity", True, (0, 1)),  # 方向多样性，高=探索方向分散
    ],
    "fairness": [
        ("fairness_score", True, (0, 100)),
        ("acc_gini_coef", False, (0, 1)),  # 越低越公平
        ("bottom_3_class_acc", True, (0, 100)),
        ("bottom_5_class_acc", True, (0, 100)),
    ],
    "corruption": [
        ("corruption_overall", True, (0, 100)),
        ("corruption_sev_1", True, (0, 100)),
        ("corruption_sev_3", True, (0, 100)),
        ("corruption_sev_5", True, (0, 100)),
    ],
    "ood": [
        ("ood_auroc_msp", True, (50, 100)),
        ("ood_auroc_entropy", True, (50, 100)),
        ("ood_fpr95_msp", False, (0, 100)),  # 越低越好
        ("ood_fpr95_entropy", False, (0, 100)),
    ],
    "adversarial": [
        ("clean_acc", True, (0, 100)),
        ("fgsm_acc", True, (0, 100)),
        ("pgd_acc", True, (0, 100)),
    ],
    "interpretability": [
        ("avg_cam_entropy", True, (0, 10)),  # 熵越高，关注越分散，可能更合理
        ("avg_cam_similarity", False, (0, 1)),  # 越低越多样
        ("avg_cam_overlap", True, (0, 1)),  # 重叠度，适中即可
    ],
}

# 维度显示配置
DIMENSION_DISPLAY = {
    "accuracy": ("🎯", "准确度", "Accuracy"),
    "calibration": ("📊", "校准性", "Calibration"),
    "diversity": ("🔀", "多样性", "Diversity"),
    "fairness": ("⚖️", "公平性", "Fairness"),
    "corruption": ("🌪️", "Corruption", "Corruption"),
    "ood": ("🔮", "OOD检测", "OOD Detection"),
    "adversarial": ("⚔️", "对抗鲁棒", "Adversarial"),
    "interpretability": ("🔍", "可解释性", "Interpretability"),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 评分计算器                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ScoreCalculator:
    """多维度评分计算器"""

    @staticmethod
    def _normalize_score(
        value: float, higher_is_better: bool, ref_range: Tuple[float, float]
    ) -> float:
        """将指标值归一化到 0-100 分"""
        min_val, max_val = ref_range
        if max_val == min_val:
            return 50.0

        # 线性归一化
        if higher_is_better:
            score = (value - min_val) / (max_val - min_val) * 100
        else:
            score = (max_val - value) / (max_val - min_val) * 100

        return max(0, min(100, score))

    @staticmethod
    def _extract_flat_metrics(result: Dict[str, Any]) -> Dict[str, float]:
        """将嵌套的结果字典扁平化为单层字典"""
        flat = {}

        # Standard metrics
        std = result.get("standard_metrics", {})
        for k, v in std.items():
            if isinstance(v, (int, float)):
                flat[k] = float(v)

        # Corruption results
        corr = result.get("corruption_results") or {}
        if corr:
            flat["corruption_overall"] = corr.get("overall_avg", 0)
            by_sev = corr.get("by_severity", {})
            for sev, val in by_sev.items():
                flat[f"corruption_sev_{sev}"] = val

        # OOD results
        ood = result.get("ood_results") or {}
        for k, v in ood.items():
            flat[k] = float(v) if isinstance(v, (int, float)) else 0

        # Adversarial results
        adv = result.get("adversarial_results") or {}
        for k, v in adv.items():
            if isinstance(v, (int, float)):
                flat[k] = float(v)

        # GradCAM metrics
        cam = result.get("gradcam_metrics") or {}
        for k, v in cam.items():
            if isinstance(v, (int, float)):
                flat[k] = float(v)

        # Distance matrix -> avg_distance, std_distance, direction_diversity
        dist_matrix = result.get("distance_matrix")
        # 支持 list 和 numpy.ndarray，需用 is not None 避免 numpy 数组布尔判断歧义
        if dist_matrix is not None and hasattr(dist_matrix, "__len__"):
            n = len(dist_matrix)
            if n > 1:
                # 提取上三角距离
                distances = [
                    dist_matrix[i][j] for i in range(n) for j in range(i + 1, n)
                ]
                count = len(distances)

                # 平均距离
                avg_dist = sum(distances) / count if count > 0 else 0
                flat["avg_distance"] = avg_dist

                # 距离标准差 (识别离群模型)
                if count > 1:
                    variance = sum((d - avg_dist) ** 2 for d in distances) / count
                    flat["std_distance"] = math.sqrt(variance)
                else:
                    flat["std_distance"] = 0

                # 方向多样性: 使用变异系数 (CV) 归一化
                # CV = std / mean，值越高表示距离分布越分散
                # 转换为 0-1 范围: tanh(CV) 或 min(CV, 1)
                if avg_dist > 0:
                    cv = flat["std_distance"] / avg_dist
                    flat["direction_diversity"] = min(cv, 1.0)
                else:
                    flat["direction_diversity"] = 0

        return flat

    @classmethod
    def calculate_dimension_score(
        cls, flat_metrics: Dict[str, float], dimension: str
    ) -> Tuple[float, Dict[str, float]]:
        """计算单个维度的综合得分

        Returns:
            (dimension_score, {metric_name: individual_score})
        """
        metrics_config = DIMENSION_METRICS.get(dimension, [])
        if not metrics_config:
            return 0.0, {}

        scores = {}
        valid_scores = []

        for metric_name, higher_is_better, ref_range in metrics_config:
            value = flat_metrics.get(metric_name)
            if value is not None:
                score = cls._normalize_score(value, higher_is_better, ref_range)
                scores[metric_name] = score
                valid_scores.append(score)

        dim_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        return dim_score, scores

    @classmethod
    def calculate_all_scores(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """计算所有维度的分数

        Returns:
            {
                "total_score": float,
                "grade": str,
                "dimensions": {
                    "accuracy": {"score": float, "metrics": {...}},
                    ...
                }
            }
        """
        flat = cls._extract_flat_metrics(result)
        dimensions = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_name, weight in DIMENSION_WEIGHTS.items():
            dim_score, metric_scores = cls.calculate_dimension_score(flat, dim_name)

            # 只有有有效分数的维度才计入
            if metric_scores:
                dimensions[dim_name] = {
                    "score": dim_score,
                    "weight": weight,
                    "metrics": metric_scores,
                }
                weighted_sum += dim_score * weight
                total_weight += weight

        total_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        grade = cls.get_grade(total_score)

        return {
            "total_score": total_score,
            "grade": grade,
            "dimensions": dimensions,
        }

    @staticmethod
    def get_grade(score: float) -> str:
        """根据分数返回等级"""
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "D"

    @staticmethod
    def get_medal(rank: int, total: int) -> str:
        """返回金银铜牌或排名"""
        if total < 2:
            return ""
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        return medals.get(rank, f"#{rank + 1}")

    @classmethod
    def rank_experiments(
        cls, results: Dict[str, Dict], key: str = "total_score"
    ) -> List[Tuple[str, float, str]]:
        """对实验按指定指标排名

        Returns:
            [(exp_name, score, medal), ...]
        """
        # 计算所有实验的分数
        exp_scores = []
        for name, result in results.items():
            score_data = cls.calculate_all_scores(result)
            exp_scores.append((name, score_data[key]))

        # 排序
        exp_scores.sort(key=lambda x: x[1], reverse=True)
        total = len(exp_scores)

        # 添加奖牌
        return [
            (name, score, cls.get_medal(i, total))
            for i, (name, score) in enumerate(exp_scores)
        ]
