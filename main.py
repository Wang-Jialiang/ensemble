"""
================================================================================
主入口模块 - CLI解析和程序入口
================================================================================
"""

import sys

sys.dont_write_bytecode = True  # 禁用 __pycache__ 生成

import argparse
from pathlib import Path
from typing import List

from .config import Config, Experiment
from .datasets import load_dataset
from .evaluation import ReportGenerator
from .training import train_experiment
from .utils import get_logger, set_seed

def main():
    """NDE 系统主入口: 解析 -> 初始化 -> 分发"""
    args = _parse_args()
    cfg = _init_config(args)
    set_seed(cfg.seed)
    
    log = get_logger()
    log.info(f"🚀 NDE System | Model: {cfg.model_name} | Models: {cfg.total_models}")

    if args.eval:
        _run_evaluation(cfg, args)
    else:
        _run_training(cfg)

def _parse_args():
    p = argparse.ArgumentParser(description="NDE 训练系统")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--landscape", action="store_true")
    return p.parse_args()

def _init_config(args):
    path = Path(args.config)
    if not path.exists(): path = Path(__file__).parent / args.config
    
    base_cfg, experiments, eval_ckpts = Config.load_yaml(str(path))
    if args.quick_test: base_cfg.quick_test = True
    
    # 将实验列表挂载到配置对象上便于后续传递 (临时)
    base_cfg._experiments = experiments
    base_cfg._eval_ckpts = eval_ckpts
    return base_cfg

def _run_training(cfg):
    log = get_logger()
    log.info(f"🚂 Training Mode | Experiments: {len(cfg._experiments)}")
    
    train_loader, val_loader, _, _, _, _ = load_dataset(cfg)
    
    for exp in cfg._experiments:
        log.info(f"\n🧪 Running: {exp.name}")
        c = cfg.apply_quick_test() if cfg.quick_test else cfg
        c = c.copy(experiment_name=exp.name, **exp.get_config_overrides())
        train_experiment(cfg=c, train_loader=train_loader, val_loader=val_loader)

def _run_evaluation(cfg, args):
    if not cfg._eval_ckpts:
        get_logger().error("❌ No checkpoints for evaluation"); return

    _, _, test_loader, c_ds, o_ds, d_ds = load_dataset(cfg)
    ckpts = [c["path"] for c in cfg._eval_ckpts]
    
    ReportGenerator.evaluate_checkpoints(
        checkpoint_paths=ckpts, test_loader=test_loader, cfg=cfg,
        corruption_dataset=c_ds, ood_dataset=o_ds, domain_dataset=d_ds,
        run_gradcam=args.gradcam, run_loss_landscape=args.landscape
    )

if __name__ == "__main__":
    main()
