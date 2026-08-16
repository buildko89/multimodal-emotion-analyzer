"""
logger.py
ロギングユーティリティ
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(config: dict, name: str = "emotion_detector") -> logging.Logger:
    """
    ロガーをセットアップ
    
    Args:
        config: 設定辞書
        name: ロガー名
        
    Returns:
        ロガーインスタンス
    """
    logger = logging.getLogger(name)
    
    # ログが有効でない場合
    if not config["logging"]["enabled"]:
        logger.addHandler(logging.NullHandler())
        return logger
    
    # ログレベル設定
    level_str = config["logging"]["level"]
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 既存のハンドラをクリア
    logger.handlers.clear()
    
    # フォーマッター
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # コンソール出力
    if config["logging"].get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # ファイル出力
    if "output_dir" in config["logging"]:
        output_dir = Path(config["logging"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名生成
        import time
        filename_format = config["logging"].get(
            "filename_format", 
            "emotion_log_%Y%m%d_%H%M%S.log"
        )
        filename = time.strftime(filename_format)
        log_path = output_dir / filename
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
