"""
================================================================================
工具模块
================================================================================
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Union

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 日志系统                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_logger(level: int = logging.INFO, file: str = None) -> logging.Logger:
    """获取单例 NDE Logger"""
    log = logging.getLogger("NDE")
    if log.handlers: return log

    log.setLevel(level)
    fmt = logging.Formatter("%(message)s")
    
    # 终端输出
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); log.addHandler(ch)

    # 文件输出 (可选)
    if file:
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        log.addHandler(fh)
    return log


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 工具函数                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def set_seed(seed: int):
    """设置所有相关的随机种子"""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """将秒数格式化为 h:m:s 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
