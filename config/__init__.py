"""
================================================================================
配置模块
================================================================================

配置管理包，包含：
- core: Config 和 Experiment 数据类
- default.yaml: 默认配置文件

使用方式:
    from ensemble.config import Config, Experiment
"""

from .core import Config, Experiment, GenerationConfig

__all__ = [
    "Config",
    "Experiment",
    "GenerationConfig",
]
