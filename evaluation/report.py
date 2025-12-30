"""
================================================================================
报告生成器模块
================================================================================

包含: ReportGenerator - 实验评估与报告生成
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..datasets.robustness.corruption import CorruptionDataset
    from ..datasets.robustness.domain import DomainShiftDataset
    from ..datasets.robustness.ood import OODDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..utils import ensure_dir, format_duration, get_logger
from .adversarial import evaluate_adversarial
from .checkpoint import CheckpointLoader
from .corruption_robustness import evaluate_corruption
from .domain_robustness import evaluate_domain_shift
from .gradcam import GradCAMAnalyzer, ModelListWrapper
from .inference import get_all_models_logits, get_models_from_source
from .landscape import LossLandscapeVisualizer
from .metrics import MetricsCalculator
from .ood import evaluate_ood
from .saver import ResultsSaver
from .strategies import get_ensemble_fn
from .visualizer import ReportVisualizer

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 报告生成器 (评估 + 报告)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ReportGenerator:
    """实验评估与报告生成器

    两种主要使用方式:
        1. 从内存评估 (训练后立即评估):
           ReportGenerator.evaluate_trainers(trainers=[...], ...)

        2. 从磁盘评估 (加载 checkpoint):
           ReportGenerator.evaluate_checkpoints(checkpoint_paths=[...], ...)
    """

    @staticmethod
    def _get_rank_marker(
        value: float, all_values: List[float], higher_is_better: bool
    ) -> str:
        """获取排名标记 (仅多实验时显示)"""
        if len(all_values) <= 1:
            return ""
        sorted_values = sorted(all_values, reverse=higher_is_better)
        if value == sorted_values[0]:
            return " 🥇"
        elif value == sorted_values[1]:
            return " 🥈"
        return ""

    @staticmethod
    def _evaluate_models(models, exp_name, test_loader, cfg, device, training_time=0.0, **datasets) -> Dict[str, Any]:
        """通用模型评估方法 - 生命周期钩子模式"""
        get_logger().info(f"\n📊 Evaluating: {exp_name}")
        res = {"experiment_name": exp_name, "training_time_seconds": training_time}

        # 1. 标准标准指标 (Acc, ECE, NLL)
        res["standard_metrics"] = ReportGenerator._run_standard_eval(models, test_loader, cfg, device)
        
        # 2. 鲁棒性套件 (Corruption, OOD, Domain)
        res.update(ReportGenerator._run_robustness_eval(models, cfg, test_loader, **datasets))
        
        # 3. 对抗性与可解释性分析
        res.update(ReportGenerator._run_analysis_eval(models, cfg, test_loader, **datasets))
        
        return res

    @staticmethod
    def _run_standard_eval(models, loader, cfg, device):
        get_logger().info("   🔍 Standard evaluation...")
        all_l, all_t = get_all_models_logits(models, loader, device)
        m = MetricsCalculator(cfg.num_classes, cfg.ece_n_bins).calculate_all_metrics(all_l, all_t, get_ensemble_fn(cfg))
        get_logger().info(f"   Ensemble Acc: {m['ensemble_acc']:.2f}% | ECE: {m['ece']:.4f}")
        return m

    @staticmethod
    def _run_robustness_eval(models, cfg, loader, **ds):
        r = {"corruption_results": None, "ood_results": None, "domain_results": None}
        
        if ds.get("corruption_dataset"):
            get_logger().info("   🔍 Corruption evaluation...")
            r["corruption_results"] = evaluate_corruption(models, ds["corruption_dataset"], cfg)
            
        if ds.get("ood_dataset"):
            get_logger().info("   🔍 OOD detection evaluation...")
            r["ood_results"] = evaluate_ood(models, loader, ds["ood_dataset"].get_loader(cfg), ds["ood_dataset"].name)
            
        if ds.get("domain_dataset"):
            get_logger().info("   🔍 Domain shift evaluation...")
            r["domain_results"] = ReportGenerator._evaluate_domain_suite(models, ds["domain_dataset"], cfg)
            
        return r

    @staticmethod
    def _run_analysis_eval(models, cfg, loader, **ds):
        a = {"adversarial_results": None, "gradcam_metrics": None}
        if ds.get("run_adversarial", True):
            get_logger().info("   🔍 Adversarial evaluation...")
            a["adversarial_results"] = evaluate_adversarial(models, loader, cfg.adv_eps, cfg.adv_alpha, cfg.adv_pgd_steps, cfg.dataset_name)
        
        if ds.get("run_gradcam", False):
            get_logger().info("   🔍 Grad-CAM analysis...")
            a["gradcam_metrics"] = GradCAMAnalyzer(cfg).analyze_ensemble_quality([ModelListWrapper(models)], loader, 50, cfg.image_size)
        return a

    @staticmethod
    def _evaluate_domain_suite(models, dataset, cfg):
        """执行全风格组合的 Domain Shift 评估"""
        res = {"by_style_strength": {}, "overall_avg": 0.0}
        accs = []
        for s in dataset.STYLES:
            for st in dataset.STRENGTHS:
                try:
                    loader = dataset.get_loader(s, st, cfg)
                    m = evaluate_domain_shift(models, loader, f"{s}_{st}", cfg.num_classes)
                    res["by_style_strength"][f"{s}_{st}"] = m
                    accs.append(m["domain_acc"])
                except FileNotFoundError: continue
        if accs: res["overall_avg"] = sum(accs) / len(accs)
        return res

    @staticmethod
    def _evaluate_trainer(
        trainer: Any,
        test_loader: DataLoader,
        cfg: Config,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ) -> Dict[str, Any]:
        """评估单个 trainer 并返回结果字典"""
        models, device = get_models_from_source(trainer)
        return ReportGenerator._evaluate_models(
            models=models,
            exp_name=trainer.name,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            training_time=getattr(trainer, "total_training_time", 0.0),
            corruption_dataset=corruption_dataset,
            run_gradcam=run_gradcam,
            run_adversarial=run_adversarial,
        )

    @classmethod
    def _generate_report(cls, results: Dict[str, Any]) -> str:
        """生成文本报告 (大纲化渲染)"""
        lines = []
        exps = list(results.keys())
        
        # 1. 绘制 Header
        lines.append("="*115)
        lines.append(f"📊 EXPERIMENT COMPARISON" if len(exps)>1 else f"📊 RESULTS: {exps[0]}")
        lines.append("="*115)

        # 2. 核心性能对比表
        lines.extend(cls._format_perf_table(results, exps))
        
        # 3. 详细子系统报告
        for name in exps: 
            lines.extend(cls._format_detailed_exp(results[name]))
            
        # 4. 鲁棒性专门板块
        lines.extend(cls._format_robustness_sections(results, exps))
        
        return "\n".join(lines)

    @classmethod
    def _format_perf_table(cls, results, names):
        t = ["\n🎯 Performance Metrics", "-"*115, 
             f"{'Experiment':<25} | {'EnsAcc↑':<10} | {'AvgInd↑':<10} | {'Oracle↑':<10} | {'ECE↓':<10} | {'NLL↓':<10} | {'Time':<12}",
             "-"*115]
        accs = [results[n].get("standard_metrics", {}).get("ensemble_acc", 0) for n in names]
        for n in names:
            m = results[n].get("standard_metrics", {})
            tm = format_duration(results[n].get("training_time_seconds", 0))
            mark = cls._get_rank_marker(m.get("ensemble_acc", 0), accs, True)
            t.append(f"{n:<25} | {m.get('ensemble_acc', 0):<7.2f}{mark:<3} | {m.get('avg_individual_acc', 0):<10.2f} | "
                     f"{m.get('oracle_acc', 0):<10.2f} | {m.get('ece', 0):<10.4f} | {m.get('nll', 0):<10.4f} | {tm:<12}")
        t.append("-" * 115)
        return t

    @classmethod
    def _format_detailed_exp(cls, r):
        m = r.get("standard_metrics", {})
        return ["\n📋 Detailed Metrics", "="*115, f"\n🔹 {r['experiment_name']}", "-"*40,
                f"   🔀 Div: Dis={m.get('disagreement', 0):.2f}% | JS={m.get('js_divergence', 0):.4f} | Spearman={m.get('spearman_correlation', 1.0):.4f}",
                f"   ⚖️ Fair: BalAcc={m.get('balanced_acc', 0):.2f}% | Gini={m.get('acc_gini_coef', 0):.4f} | Score={m.get('fairness_score', 0):.2f}",
                "-" * 40]

    @classmethod
    def _format_robustness_sections(cls, results, names):
        s = ["\n🧪 Robustness Summary"]
        for n in names:
            r = results[n]
            corr = r.get("corruption_results", {}).get("overall_avg", 0)
            pgd = r.get("adversarial_results", {}).get("pgd_acc", 0)
            ood = r.get("ood_results", {}).get("auc_roc", 0)
            s.append(f"   {n:<25} | Corr: {corr:2.2f}% | PGD: {pgd:2.2f}% | OOD AUC: {ood:.4f}")
        return s

    @classmethod
    def _save_and_print(cls, results: Dict[str, Dict], save_dir: str):
        """保存并打印报告"""
        saver = ResultsSaver(save_dir)
        report_content = cls._generate_report(results)

        # 保存结果 (统一格式)
        saver.save_comparison(results, "comprehensive_results")

        # 保存报告到文件 (不打印到控制台)
        report_path = Path(save_dir) / "detailed_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        get_logger().info(f"\n✅ Detailed report saved to: {report_path}")
        get_logger().info(f"✅ All results saved to: {save_dir}")

    @classmethod
    def evaluate_trainers(
        cls,
        trainers: List,  # List of StagedEnsembleTrainer instances
        test_loader: DataLoader,
        cfg: Config,
        save_dir: str,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ):
        """
        从内存评估多个 trainer 并生成报告

        适用场景: 训练刚完成，模型还在内存中
        """
        get_logger().info(
            f"\n{'=' * 80}\n📊 EVALUATION MODE | Models: {len(trainers)}\n{'=' * 80}"
        )

        # 评估所有 trainers
        results = {}
        for idx, trainer in enumerate(trainers, 1):
            get_logger().info(f"\n[{idx}/{len(trainers)}] {trainer.name}")
            result = cls._evaluate_trainer(
                trainer,
                test_loader,
                cfg,
                corruption_dataset,
                run_gradcam,
                run_adversarial,
            )
            results[trainer.name] = result

        # 生成可视化图表
        get_logger().info("\n📊 Generating visualizations...")
        visualizer = ReportVisualizer(save_dir, dpi=cfg.plot_dpi)
        visualizer.generate_all(results)

        # 生成并保存报告
        cls._save_and_print(results, save_dir)

    @classmethod
    def evaluate_checkpoints(
        cls,
        checkpoint_paths: List[str],
        test_loader: DataLoader,
        cfg: Config,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        ood_dataset: Optional["OODDataset"] = None,
        domain_dataset: Optional["DomainShiftDataset"] = None,
        run_gradcam: bool = False,
        run_loss_landscape: bool = False,
        run_adversarial: bool = True,
    ):
        """
        从磁盘加载 checkpoint 并评估

        适用场景: 评估已保存的模型，与训练解耦
        这是 evaluation 模块的主入口，完全独立于 training 模块。
        """
        get_logger().info(f"\n{'=' * 80}")
        get_logger().info(
            f"📊 EVALUATION FROM CHECKPOINTS | Count: {len(checkpoint_paths)}"
        )
        get_logger().info(f"{'=' * 80}")

        output_dir = cfg.save_dir
        ensure_dir(output_dir)
        results = {}
        all_models = {}  # 收集所有实验的模型用于 Loss Landscape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for idx, ckpt_path in enumerate(checkpoint_paths, 1):
            get_logger().info(f"\n[{idx}/{len(checkpoint_paths)}] Loading: {ckpt_path}")

            # 加载模型
            ctx = CheckpointLoader.load(ckpt_path, cfg)
            exp_name = ctx["name"]
            models = [m.to(device) for m in ctx["models"]]
            all_models[exp_name] = models  # 保存用于后续分析

            # 使用通用评估方法
            result = cls._evaluate_models(
                models=models,
                exp_name=exp_name,
                test_loader=test_loader,
                cfg=cfg,
                device=device,
                training_time=ctx["training_time"],
                corruption_dataset=corruption_dataset,
                ood_dataset=ood_dataset,
                domain_dataset=domain_dataset,
                run_gradcam=run_gradcam,
                run_adversarial=run_adversarial,
            )
            result["train_config"] = ctx["config"]
            results[exp_name] = result

        # 生成可视化图表
        get_logger().info("\n📊 Generating visualizations...")
        visualizer = ReportVisualizer(output_dir, dpi=cfg.plot_dpi)
        visualizer.generate_all(results)

        # Loss Landscape 分析
        if run_loss_landscape and all_models:
            get_logger().info("\n🏔️ Generating Loss Landscape visualizations...")
            landscape_viz = LossLandscapeVisualizer(output_dir, dpi=cfg.plot_dpi)

            for exp_name, models in all_models.items():
                # 模型参数距离热力图 (无需 loss-landscapes 依赖)
                landscape_viz.plot_model_distance_heatmap(
                    models, filename=f"{exp_name}_model_distances.png"
                )

                # Loss Landscape 插值 (需要 loss-landscapes)
                landscape_viz.plot_ensemble_interpolations(
                    models,
                    test_loader,
                    device,
                    filename=f"{exp_name}_loss_landscape.png",
                )

                # 2D/3D 表面图 - 为第一个模型生成 (计算量较大)
                if len(models) > 0:
                    landscape_viz.plot_2d_plane(
                        models[0],
                        test_loader,
                        device,
                        distance=1.0,
                        steps=20,  # 减少步数以加快计算
                        filename=f"{exp_name}_landscape_surface.png",
                        model_name=f"{exp_name}_M1",
                    )

        # 生成并保存文本报告
        cls._save_and_print(results, output_dir)

        get_logger().info(f"\n✅ Complete! All reports saved to: {output_dir}")
        return results
