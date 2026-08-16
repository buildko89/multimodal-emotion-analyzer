"""
メインプログラム: emotion_detector.py
Windows/Linux対応の表情認識アプリケーション
GPIO/LED制御を削除し、画面表示のみに対応
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from models.detector import EmotionDetector
from utils.visualizer import Visualizer
from utils.logger import setup_logger


class EmotionDetectorApp:
    """表情認識アプリケーションのメインクラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # ロガー設定
        self.logger = setup_logger(self.config)
        self.logger.info("Emotion Detector アプリケーションを起動します")
        
        # 検出器初期化
        self.detector = EmotionDetector(self.config)
        
        # 可視化ツール初期化
        self.visualizer = Visualizer(self.config)
        
        # 統計情報
        self.frame_count = 0
        self.fps = 0.0
        self.prev_time = time.time()
        
        # 出力設定
        self.output_dir = Path(self.config["output"]["output_dir"])
        if self.config["output"]["save_results"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSVライター
        self.csv_file = None
        if self.config["output"]["save_csv"]:
            self._setup_csv_writer()
        
        # ビデオライター
        self.video_writer = None
    
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {config_path}")
            print("デフォルト設定を使用します")
            return self._get_default_config()
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            print("デフォルト設定を使用します")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """デフォルト設定を返す"""
        return {
            "face_detection": {
                "model": "yolov8",
                "yolov8": {
                    "model_size": "n",
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45,
                    "max_detections": 10
                }
            },
            "emotion_recognition": {
                "model": "hsemotion",
                "hsemotion": {"model_name": "enet_b0_8_best_afew"},
                "smoothing": {
                    "enabled": True,
                    "window_size": 6,
                    "min_confidence": 0.3
                }
            },
            "camera": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "input": {"source": "camera", "file_path": None},
            "output": {
                "save_results": False,
                "output_dir": "output",
                "save_video": False,
                "save_csv": False
            },
            "display": {
                "show_window": True,
                "window_title": "Emotion Detection",
                "show_fps": True,
                "show_confidence": True,
                "show_bbox": True,
                "font": {"scale": 0.6, "thickness": 2},
                "bbox": {"thickness": 2, "color_mode": "emotion"}
            },
            "logging": {"enabled": True, "level": "INFO"},
            "performance": {"use_gpu": False, "frame_skip": 0}
        }
    
    def _setup_csv_writer(self):
        """CSVライターをセットアップ"""
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"emotions_{timestamp}.csv"
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame", "timestamp", "face_id", 
            "emotion", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
        ])
        self.logger.info(f"CSV出力: {csv_path}")
    
    def _setup_video_source(self) -> cv2.VideoCapture:
        """ビデオソースをセットアップ"""
        input_config = self.config["input"]
        
        if input_config["source"] == "camera":
            # カメラ入力
            camera_config = self.config["camera"]
            cap = cv2.VideoCapture(camera_config["device_id"])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
            cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])
            self.logger.info(f"カメラ {camera_config['device_id']} を開きました")
        else:
            # ファイル入力
            file_path = input_config["file_path"]
            if file_path is None:
                raise ValueError("ファイル入力の場合は file_path を指定してください")
            cap = cv2.VideoCapture(str(file_path))
            self.logger.info(f"ファイル {file_path} を開きました")
        
        if not cap.isOpened():
            raise RuntimeError("ビデオソースを開けませんでした")
        
        return cap
    
    def _setup_video_writer(self, frame_shape: Tuple[int, int, int]):
        """ビデオライターをセットアップ"""
        if not self.config["output"]["save_video"]:
            return
        
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config["output"]["video_codec"])
        fps = self.config["output"]["video_fps"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"output_{timestamp}.mp4"
        
        self.video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )
        self.logger.info(f"動画出力: {output_path}")
    
    def _update_fps(self):
        """FPS計算"""
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        self.prev_time = current_time
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームを処理
        
        Args:
            frame: 入力フレーム
            
        Returns:
            処理済みフレーム
        """
        # 顔検出 + 表情認識
        results = self.detector.detect(frame)
        
        # CSV出力
        if self.csv_file is not None:
            timestamp = time.time()
            for i, result in enumerate(results):
                bbox = result["bbox"]
                self.csv_writer.writerow([
                    self.frame_count,
                    timestamp,
                    i,
                    result["emotion"],
                    result["confidence"],
                    bbox[0], bbox[1], bbox[2], bbox[3]
                ])
        
        # 可視化
        output_frame = self.visualizer.draw(frame, results, self.fps)
        
        return output_frame
    
    def run(self, camera_id: Optional[int] = None, 
            input_file: Optional[str] = None):
        """
        アプリケーション実行
        
        Args:
            camera_id: カメラID (指定時は設定を上書き)
            input_file: 入力ファイルパス (指定時は設定を上書き)
        """
        # 引数で設定を上書き
        if camera_id is not None:
            self.config["camera"]["device_id"] = camera_id
            self.config["input"]["source"] = "camera"
        if input_file is not None:
            self.config["input"]["file_path"] = input_file
            if input_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.config["input"]["source"] = "image"
            else:
                self.config["input"]["source"] = "video"
        
        # ビデオソース開始
        cap = self._setup_video_source()
        
        try:
            # 画像入力の場合
            if self.config["input"]["source"] == "image":
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("画像を読み込めませんでした")
                
                output_frame = self._process_frame(frame)
                
                if self.config["display"]["show_window"]:
                    cv2.imshow(self.config["display"]["window_title"], output_frame)
                    self.logger.info("何かキーを押すと終了します")
                    cv2.waitKey(0)
                
                if self.config["output"]["save_results"]:
                    output_path = self.output_dir / "result.jpg"
                    cv2.imwrite(str(output_path), output_frame)
                    self.logger.info(f"結果を保存しました: {output_path}")
            
            # 動画/カメラ入力の場合
            else:
                frame_skip = self.config["performance"]["frame_skip"]
                
                while True:
                    # フレーム読み込み
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.info("ビデオストリーム終了")
                        break
                    
                    # フレームスキップ
                    if frame_skip > 0 and self.frame_count % (frame_skip + 1) != 0:
                        self.frame_count += 1
                        continue
                    
                    # 初回のみビデオライター設定
                    if self.frame_count == 0 and self.video_writer is None:
                        self._setup_video_writer(frame.shape)
                    
                    # フレーム処理
                    output_frame = self._process_frame(frame)
                    
                    # FPS更新
                    self._update_fps()
                    
                    # 動画出力
                    if self.video_writer is not None:
                        self.video_writer.write(output_frame)
                    
                    # 画面表示
                    if self.config["display"]["show_window"]:
                        cv2.imshow(
                            self.config["display"]["window_title"], 
                            output_frame
                        )
                        
                        # キー入力処理
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            self.logger.info("ユーザーが終了しました")
                            break
                    
                    self.frame_count += 1
                    
                    # ログ出力 (100フレームごと)
                    if self.frame_count % 100 == 0:
                        self.logger.info(
                            f"処理フレーム数: {self.frame_count}, FPS: {self.fps:.1f}"
                        )
        
        finally:
            # クリーンアップ
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            if self.csv_file is not None:
                self.csv_file.close()
            cv2.destroyAllWindows()
            
            self.logger.info(f"総処理フレーム数: {self.frame_count}")
            self.logger.info("アプリケーションを終了しました")


def main():
    """エントリーポイント"""
    parser = argparse.ArgumentParser(
        description="表情認識アプリケーション (Windows/Linux対応)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="設定ファイルのパス (デフォルト: config.yaml)"
    )
    parser.add_argument(
        "--camera", 
        type=int,
        help="カメラID (指定時は設定ファイルを上書き)"
    )
    parser.add_argument(
        "--input", 
        type=str,
        help="入力ファイルパス (動画または画像)"
    )
    
    args = parser.parse_args()
    
    try:
        # アプリケーション起動
        app = EmotionDetectorApp(config_path=args.config)
        app.run(camera_id=args.camera, input_file=args.input)
    
    except KeyboardInterrupt:
        print("\n中断されました")
        sys.exit(0)
    
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
