"""
================================================================================
主入口模块 - CLI解析和程序入口
================================================================================
"""

import argparse
from pathlib import Path
from typing import List

from .config import Config, Experiment
from .datasets import load_dataset
from .evaluation import ReportGenerator
from .training import train_experiment
from .utils import get_logger, set_seed

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def run_train_mode(
    base_cfg: Config, experiments: List[Experiment], train_loader, val_loader
):
    """训练模式"""
    get_logger().info(f"🚂 训练模式 | 实验数: {len(experiments)}")

    for idx, exp in enumerate(experiments, 1):
        get_logger().info(f"\n🧪 [{idx}/{len(experiments)}] {exp.name} - {exp.desc}")

        cfg = base_cfg.apply_quick_test() if base_cfg.quick_test else base_cfg
        cfg = cfg.copy(experiment_name=exp.name, **exp.get_overrides())

        train_experiment(
            experiment_name=exp.name,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentation_method=exp.augmentation,
            use_curriculum=exp.use_curriculum,
            fixed_ratio=exp.fixed_ratio,
            fixed_prob=exp.fixed_prob,
            resume=base_cfg.resume_from if idx == 1 else None,
        )
    get_logger().info(f"\n✅ 完成 | Checkpoints -> {base_cfg.save_dir}/checkpoints/")


def run_eval_mode(
    base_cfg: Config,
    eval_checkpoints: list,
    test_loader,
    corruption_dataset,
    run_loss_landscape: bool = False,
):
    """评估模式 - 加载模型并生成报告"""
    checkpoint_paths = [ckpt["path"] for ckpt in eval_checkpoints]

    ReportGenerator.evaluate_checkpoints(
        checkpoint_paths=checkpoint_paths,
        test_loader=test_loader,
        cfg=base_cfg,
        output_dir=base_cfg.save_dir,
        corruption_dataset=corruption_dataset,
        run_gradcam=True,
        run_loss_landscape=run_loss_landscape,
    )


def main():
    parser = argparse.ArgumentParser(description="NDE 训练系统")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--eval", action="store_true", help="进入评估模式")
    parser.add_argument(
        "--quick-test", action="store_true", help="快速测试模式 (4 epoch, 1 model)"
    )
    parser.add_argument(
        "--loss-landscape", action="store_true", help="生成 Loss Landscape 可视化"
    )
    args = parser.parse_args()

    # 加载配置
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent / "configs" / args.config

    base_cfg, experiments, eval_ckpts = Config.load_yaml(str(cfg_path))

    # 命令行优先级高于 YAML
    if args.quick_test:
        base_cfg.quick_test = True

    set_seed(base_cfg.seed)

    get_logger().info(
        f"🚀 NDE System | Model: {base_cfg.model_name} | Total Models: {base_cfg.total_models}"
    )

    if args.eval:
        if not eval_ckpts:
            return get_logger().error("❌ 请在 config.yaml 中指定 eval_checkpoints")
        _, _, test_loader, corruption_dataset = load_dataset(base_cfg)
        run_eval_mode(
            base_cfg,
            eval_ckpts,
            test_loader,
            corruption_dataset,
            run_loss_landscape=args.loss_landscape,
        )
    else:
        train_loader, val_loader, _, _ = load_dataset(base_cfg)
        run_train_mode(base_cfg, experiments, train_loader, val_loader)


if __name__ == "__main__":
    main()
