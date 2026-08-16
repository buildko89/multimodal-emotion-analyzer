"""
メインプログラム: emotion_detector_mediapipe.py
MediaPipe専用の表情認識アプリケーション (エラー対策強化版)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import yaml

# 既存のモジュールを再利用
from models.detector import EmotionDetector
from utils.visualizer import Visualizer
from utils.logger import setup_logger

# === MediaPipe専用の検出器クラス ===
class MediaPipeEmotionDetector(EmotionDetector):
    """
    EmotionDetectorを継承し、顔検出部分のみMediaPipe専用にオーバーライドしたクラス
    """
    
    def _init_face_detector(self):
        """顔検出器を初期化 (MediaPipe強制・堅牢版)"""
        print("MediaPipe顔検出器を初期化します (専用モード)")
        
        # 設定を強制的に書き換え (config.yamlの内容を無視)
        self.config["face_detection"]["model"] = "mediapipe"
        if "mediapipe" not in self.config["face_detection"]:
            self.config["face_detection"]["mediapipe"] = {
                "min_detection_confidence": 0.5,
                "model_selection": 0
            }
        
        try:
            # MediaPipe 0.10以降は tasks APIを使用
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request
            import os
            
            config = self.config["face_detection"]["mediapipe"]
            
            # モデルファイルのパスとURL
            model_dir = Path("trained_models/mediapipe")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "blaze_face_short_range.tflite"
            
            # モデルファイルが存在しない場合はダウンロード
            if not model_path.exists():
                print(f"MediaPipeモデルをダウンロードしています... ({model_path})")
                model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print("モデルのダウンロードが完了しました")
                except Exception as e:
                    print(f"モデルのダウンロードに失敗しました: {e}")
                    raise
            
            # Face Detector用のオプション設定
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=config.get("min_detection_confidence", 0.5)
            )
            
            # Face Detectorの作成
            self.face_model = vision.FaceDetector.create_from_options(options)
            self.use_new_mediapipe = True
            print("MediaPipe顔検出器の初期化に成功しました (tasks API)")
            
        except Exception as e:
            print(f"\n[Error] MediaPipeの初期化に失敗しました。\n")
            print(f"詳細エラー: {e}")
            print("--------------------------------------------------")
            print("ヒント: 以下のコマンドで再インストールを試してください")
            print("pip uninstall mediapipe")
            print("pip install mediapipe")
            print("--------------------------------------------------\n")
            raise e

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """MediaPipeで顔検出 (tasks API対応)"""
        import mediapipe as mp
        
        # 画像をRGBに変換してMediaPipe Image形式に
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 推論実行
        detection_result = self.face_model.detect(mp_image)
        
        faces = []
        if detection_result.detections:
            h, w = image.shape[:2]
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                
                # バウンディングボックスの座標を取得
                x1 = bbox.origin_x
                y1 = bbox.origin_y
                x2 = x1 + bbox.width
                y2 = y1 + bbox.height
                
                # 画面外にはみ出した場合の補正
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                faces.append((x1, y1, x2, y2))
        
        return faces

    # detectメソッド内で親クラスが _detect_faces_mediapipe を呼ぶように
    # 実際には親クラスの detect メソッド内の分岐で呼び出されるが、
    # 念のため detect メソッド自体も、強制的に mediapipe を使うように書き換えることも可能。
    # ここでは親クラスの実装を利用する。


# === アプリケーションクラス (Detectorを差し替え) ===
class EmotionDetectorAppMP:
    """MediaPipe専用アプリケーションクラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # ロガー設定
        self.logger = setup_logger(self.config)
        self.logger.info("Emotion Detector (MediaPipe Custom) 起動")
        
        # ★ここで専用のDetectorを使用
        self.detector = MediaPipeEmotionDetector(self.config)
        
        # 以下はオリジナルと同じ初期化処理
        self.visualizer = Visualizer(self.config)
        self.frame_count = 0
        self.fps = 0.0
        self.prev_time = time.time()
        self.output_dir = Path(self.config["output"]["output_dir"])
        if self.config["output"]["save_results"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = None
        if self.config["output"]["save_csv"]:
            self._setup_csv_writer()
        self.video_writer = None

    # 以下、EmotionDetectorAppからメソッドをコピー・再利用
    # (継承を使っても良いが、構造を明確にするため必要なメソッドを定義)
    
    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception:
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        # デフォルト設定 (MediaPipe優先)
        return {
            "face_detection": {
                "model": "mediapipe",
                "mediapipe": {"min_detection_confidence": 0.5, "model_selection": 0}
            },
            "emotion_recognition": {
                "model": "original",
                "original": {"model_path": "trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5", "input_shape": [64, 64]},
                "smoothing": {"enabled": True, "window_size": 6, "min_confidence": 0.3}
            },
            "camera": {"device_id": 0, "width": 640, "height": 480, "fps": 30},
            "input": {"source": "camera", "file_path": None},
            "output": {"save_results": False, "output_dir": "output", "save_video": False, "save_csv": False},
            "display": {
                "show_window": True, "window_title": "Emotion Detection MP",
                "show_fps": True, "show_confidence": True, "show_bbox": True,
                "font": {"scale": 0.6, "thickness": 2},
                "bbox": {"thickness": 2, "color_mode": "emotion"}
            },
            "logging": {"enabled": True, "level": "INFO"},
            "performance": {"use_gpu": False, "frame_skip": 0}
        }

    def _setup_csv_writer(self):
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"emotions_mp_{timestamp}.csv"
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "timestamp", "face_id", "emotion", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])

    def _setup_video_source(self) -> cv2.VideoCapture:
        input_config = self.config["input"]
        if input_config["source"] == "camera":
            cap = cv2.VideoCapture(self.config["camera"]["device_id"])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
            cap.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
        else:
            cap = cv2.VideoCapture(str(input_config["file_path"]))
        if not cap.isOpened():
            raise RuntimeError("ビデオソースを開けませんでした")
        return cap

    def _setup_video_writer(self, frame_shape):
        if not self.config["output"]["save_video"]: return
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config["output"]["video_codec"])
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"output_mp_{timestamp}.mp4"
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, self.config["output"]["video_fps"], (w, h))

    def _process_frame(self, frame):
        results = self.detector.detect(frame)
        if self.csv_file:
            ts = time.time()
            for i, r in enumerate(results):
                b = r["bbox"]
                self.csv_writer.writerow([self.frame_count, ts, i, r["emotion"], r["confidence"], b[0], b[1], b[2], b[3]])
        return self.visualizer.draw(frame, results, self.fps)

    def run(self, camera_id=None, input_file=None):
        if camera_id is not None:
            self.config["camera"]["device_id"] = camera_id
            self.config["input"]["source"] = "camera"
        if input_file is not None:
            self.config["input"]["file_path"] = input_file
            self.config["input"]["source"] = "image" if input_file.lower().endswith((".jpg",".png",".bmp")) else "video"

        cap = self._setup_video_source()
        try:
            if self.config["input"]["source"] == "image":
                ret, frame = cap.read()
                if ret:
                    out = self._process_frame(frame)
                    if self.config["display"]["show_window"]:
                        cv2.imshow(self.config["display"]["window_title"], out)
                        cv2.waitKey(0)
            else:
                frame_skip = self.config["performance"]["frame_skip"]
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_skip > 0 and self.frame_count % (frame_skip + 1) != 0:
                        self.frame_count += 1
                        continue
                    if self.frame_count == 0 and self.video_writer is None:
                        self._setup_video_writer(frame.shape)
                    
                    self.fps = 1.0 / (time.time() - self.prev_time) if time.time() - self.prev_time > 0 else 0
                    self.prev_time = time.time()
                    
                    out = self._process_frame(frame)
                    if self.video_writer: self.video_writer.write(out)
                    
                    if self.config["display"]["show_window"]:
                        cv2.imshow(self.config["display"]["window_title"], out)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                    
                    self.frame_count += 1
                    if self.frame_count % 100 == 0:
                        self.logger.info(f"Frame: {self.frame_count}, FPS: {self.fps:.1f}")
        finally:
            cap.release()
            if self.video_writer: self.video_writer.release()
            if self.csv_file: self.csv_file.close()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="MediaPipe専用 表情認識アプリ")
    parser.add_argument("--config", default="config.yaml", help="設定ファイル")
    parser.add_argument("--camera", type=int, help="カメラID")
    parser.add_argument("--input", type=str, help="入力ファイル")
    args = parser.parse_args()
    
    EmotionDetectorAppMP(args.config).run(args.camera, args.input)

if __name__ == "__main__":
    main()