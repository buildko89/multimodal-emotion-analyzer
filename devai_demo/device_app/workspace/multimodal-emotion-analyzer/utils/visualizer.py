"""
visualizer.py
検出結果の可視化ユーティリティ
"""

from typing import List, Dict, Tuple

import cv2
import numpy as np


class Visualizer:
    """検出結果を画像に描画するクラス"""
    
    def __init__(self, config: dict):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.display_config = config["display"]
        
        # 感情別カラー設定 (BGR)
        self.emotion_colors = {
            emotion.lower(): tuple(color)
            for emotion, color in self.display_config.get("emotion_colors", {}).items()
        }
        
        # デフォルトカラー
        default_colors = {
            "angry": (0, 0, 255),       # 赤
            "contempt": (128, 0, 128),  # 紫
            "disgust": (0, 128, 0),     # 緑
            "fear": (128, 128, 0),      # シアン
            "happy": (0, 255, 255),     # 黄色
            "neutral": (128, 128, 128), # グレー
            "sad": (255, 0, 0),         # 青
            "sadness": (255, 255, 0),   # 水色
            "surprise": (0, 128, 255),  # オレンジ
        }
        
        # デフォルトカラーをマージ
        for emotion, color in default_colors.items():
            if emotion not in self.emotion_colors:
                self.emotion_colors[emotion] = color
    
    def _get_color_for_emotion(self, emotion: str, confidence: float) -> Tuple[int, int, int]:
        """
        感情に応じた色を取得
        
        Args:
            emotion: 感情ラベル
            confidence: 信頼度
            
        Returns:
            BGR色
        """
        color_mode = self.display_config["bbox"]["color_mode"]
        
        if color_mode == "emotion":
            # 感情別カラー
            key = emotion.lower()
            if key not in self.emotion_colors:
                print(f"Warning: Unknown emotion label '{emotion}' detected. Using default white color.")
                return (255, 255, 255)
            return self.emotion_colors[key]
        
        elif color_mode == "confidence":
            # 信頼度別カラー (緑→黄→赤)
            if confidence > 0.7:
                return (0, 255, 0)    # 緑
            elif confidence > 0.4:
                return (0, 255, 255)  # 黄色
            else:
                return (0, 0, 255)    # 赤
        
        else:  # fixed
            # 固定色
            return tuple(self.display_config["bbox"]["fixed_color"])
    
    def draw(self, image: np.ndarray, results: List[Dict], fps: float = 0.0) -> np.ndarray:
        """
        検出結果を描画
        
        Args:
            image: 入力画像 (BGR)
            results: 検出結果のリスト
            fps: FPS
            
        Returns:
            描画済み画像
        """
        output = image.copy()
        
        # フォント設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.display_config["font"]["scale"]
        font_thickness = self.display_config["font"]["thickness"]
        font_color = tuple(self.display_config["font"]["color"])
        
        # バウンディングボックス設定
        bbox_thickness = self.display_config["bbox"]["thickness"]
        
        # 各検出結果を描画
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            emotion = result["emotion"]
            confidence = result["confidence"]
            
            # バウンディングボックスの色
            bbox_color = self._get_color_for_emotion(emotion, confidence)
            
            # バウンディングボックス描画
            if self.display_config["show_bbox"]:
                cv2.rectangle(output, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
            
            # ラベル作成
            if self.display_config["show_confidence"]:
                label = f"{emotion}: {confidence:.2f}"
            else:
                label = emotion
            
            # ラベル背景
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            
            cv2.rectangle(
                output,
                (x1, label_y - label_height - baseline),
                (x1 + label_width, label_y + baseline),
                bbox_color,
                -1  # 塗りつぶし
            )
            
            # ラベルテキスト
            cv2.putText(
                output,
                label,
                (x1, label_y - baseline),
                font,
                font_scale,
                font_color,
                font_thickness,
                cv2.LINE_AA
            )
        
        # FPS表示
        if self.display_config["show_fps"] and fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                output,
                fps_text,
                (10, 30),
                font,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA
            )
        
        # 検出数表示
        detection_count = len(results)
        count_text = f"Faces: {detection_count}"
        cv2.putText(
            output,
            count_text,
            (10, 60),
            font,
            font_scale,
            (0, 255, 0),
            font_thickness,
            cv2.LINE_AA
        )
        
        return output
