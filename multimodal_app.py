"""
マルチモーダル感情分析統合アプリケーション
表情認識 × 音声認識のリアルタイム統合分析
"""

import argparse
import csv
from datetime import datetime
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# 既存モジュールをインポート
sys.path.insert(0, str(Path(__file__).parent))
from emotion_detector_mediapipe import MediaPipeEmotionDetector
from modules.whisper_recognizer import WhisperRecognizer
from modules.multimodal_analyzer import MultiModalAnalyzer
from utils.visualizer import Visualizer
from utils.logger import setup_logger


class MultiModalEmotionApp:
    """マルチモーダル感情分析アプリケーション"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # ロガー設定
        self.logger = setup_logger(self.config)
        self.logger.info("マルチモーダル感情分析アプリケーション 起動")
        
        # 各モジュール初期化
        print("\n=== モジュール初期化 ===")
        
        # 1. 表情認識
        print("1. 表情認識モジュールを初期化中...")
        self.emotion_detector = MediaPipeEmotionDetector(self.config)
        
        # 2. 音声認識
        print("2. 音声認識モジュールを初期化中...")
        whisper_config = self.config.get("whisper", {})
        self.whisper = WhisperRecognizer(
            model_name=whisper_config.get("model", "base"),
            device=whisper_config.get("device", "cpu"),
            language=whisper_config.get("language", "ja")
        )
        
        # 3. マルチモーダル統合分析
        print("3. マルチモーダル分析モジュールを初期化中...")
        self.multimodal_analyzer = MultiModalAnalyzer()
        
        # 4. 可視化
        self.visualizer = Visualizer(self.config)
        
        # 統計情報
        self.frame_count = 0
        self.fps = 0.0
        self.prev_time = time.time()
        
        # 出力設定
        self.output_dir = Path(self.config["output"]["output_dir"])
        if self.config["output"]["save_results"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 音声処理用
        self.audio_file = None
        self.speech_result = None
        self.speech_features = None
        
        # 感情変遷ログ用
        self.prev_emotion_state = {
            "emotion": None,
            "stress_score": None,
            "positive_score": None
        }
        self.emotion_log_file = self.output_dir / "emotion_timeline.csv"
        self._init_emotion_log()
        
        print("=== 初期化完了 ===\n")
    
    def _init_emotion_log(self):
        """感情変遷ログファイルを初期化"""
        if not self.config["output"]["save_results"]:
            return
        
        # ファイルが存在しない場合はヘッダーを書き込む
        if not self.emotion_log_file.exists():
            with open(self.emotion_log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "DateTime",
                    "Emotion",
                    "Confidence",
                    "Stress_Score",
                    "Positive_Score",
                    "State",
                    "Has_Speech"
                ])
            print(f"感情変遷ログファイルを作成: {self.emotion_log_file}")
    
    def _log_emotion_change(self, emotion_data: dict, overall_state: dict, has_speech: bool):
        """
        感情変化をログに記録
        
        Args:
            emotion_data: 表情データ
            overall_state: 総合状態
            has_speech: 音声データの有無
        """
        if not self.config["output"]["save_results"]:
            return
        
        if not emotion_data:
            return
        
        current_emotion = emotion_data.get("emotion", "unknown")
        current_confidence = emotion_data.get("confidence", 0.0)
        current_stress = overall_state.get("stress_score", 0)
        current_positive = overall_state.get("positive_score", 50)
        current_state = overall_state.get("state", "")
        
        # 値が変化したかチェック
        emotion_changed = self.prev_emotion_state["emotion"] != current_emotion
        stress_changed = (self.prev_emotion_state["stress_score"] is None or 
                         abs(self.prev_emotion_state["stress_score"] - current_stress) >= 5)
        positive_changed = (self.prev_emotion_state["positive_score"] is None or 
                           abs(self.prev_emotion_state["positive_score"] - current_positive) >= 5)
        
        # いずれかが変化した場合のみログに記録
        if emotion_changed or stress_changed or positive_changed:
            timestamp = time.time()
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.emotion_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{timestamp:.3f}",
                    datetime_str,
                    current_emotion,
                    f"{current_confidence:.2f}",
                    f"{current_stress:.1f}",
                    f"{current_positive:.1f}",
                    current_state,
                    "Yes" if has_speech else "No"
                ])
            
            # 前回の値を更新
            self.prev_emotion_state["emotion"] = current_emotion
            self.prev_emotion_state["stress_score"] = current_stress
            self.prev_emotion_state["positive_score"] = current_positive
    
    def _load_config(self, config_path: str) -> dict:
        """設定を読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """デフォルト設定"""
        return {
            "whisper": {
                "model": "base",
                "device": "cpu",
                "language": "ja"
            },
            "face_detection": {
                "model": "mediapipe",
                "mediapipe": {
                    "min_detection_confidence": 0.5,
                    "model_selection": 0
                }
            },
            "emotion_recognition": {
                "model": "original",
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
            "input": {
                "source": "camera"
            },
            "output": {
                "save_results": True,
                "output_dir": "output",
                "save_video": False,
                "save_csv": True
            },
            "display": {
                "show_window": True,
                "window_title": "MultiModal Emotion Analysis",
                "show_fps": True
            },
            "logging": {
                "enabled": True,
                "level": "INFO"
            }
        }
    
    def process_with_audio_file(
        self,
        video_source,
        audio_file: str
    ):
        """
        動画 + 音声ファイルで処理
        
        Args:
            video_source: 動画ソース
            audio_file: 音声ファイルパス
        """
        print(f"\n音声ファイルを認識中: {audio_file}")
        
        # 音声を事前に認識
        self.speech_result = self.whisper.transcribe_file(
            audio_file,
            word_timestamps=True
        )
        self.speech_features = self.whisper.analyze_speech_features(self.speech_result)
        
        print(f"認識テキスト: {self.speech_result['text']}")
        print(f"発話速度: {self.speech_features['speech_rate']:.2f} 単語/秒")
        print(f"総単語数: {self.speech_features['total_words']}\n")
        
        # 動画処理
        self._process_video(video_source, with_audio=True)
    
    def _process_video(self, video_source, with_audio: bool = False):
        """動画処理メインループ"""
        
        cap = cv2.VideoCapture(video_source) if isinstance(video_source, (str, int)) else video_source
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # FPS計算
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time
                
                # 表情認識
                emotion_results = self.emotion_detector.detect(frame)
                
                # 統合分析（音声データがある場合）
                if with_audio and emotion_results and self.speech_result:
                    # 最初の顔の表情データを取得
                    emotion_data = emotion_results[0] if emotion_results else None
                    
                    # マルチモーダル分析
                    analysis = self.multimodal_analyzer.analyze(
                        emotion_data=emotion_data,
                        speech_data={"text": self.speech_result['text']},
                        timestamp=current_time
                    )
                    
                    # 感情変遷をログに記録
                    self._log_emotion_change(
                        emotion_data,
                        analysis.get("overall_state", {}),
                        has_speech=True
                    )
                    
                    # 分析結果を画面に描画
                    frame = self._draw_multimodal_results(frame, analysis)
                else:
                    # 表情のみ
                    frame = self.visualizer.draw(frame, emotion_results, self.fps)
                    
                    # 表情のみの場合もログに記録
                    if emotion_results:
                        emotion_data = emotion_results[0]
                        # 表情のみの状態評価を取得
                        emotion_only_state = self.multimodal_analyzer._emotion_only_state(emotion_data)
                        self._log_emotion_change(
                            emotion_data,
                            emotion_only_state,
                            has_speech=False
                        )
                
                # 画面表示
                if self.config["display"]["show_window"]:
                    cv2.imshow(self.config["display"]["window_title"], frame)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                
                self.frame_count += 1
                
                if self.frame_count % 100 == 0:
                    self.logger.info(f"Frame: {self.frame_count}, FPS: {self.fps:.1f}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 統計表示
            if with_audio:
                self._print_final_statistics()
    
    def _draw_multimodal_results(self, frame, analysis: dict) -> np.ndarray:
        """マルチモーダル分析結果を描画"""
        
        # 基本的な表情描画
        emotion_data = analysis.get("emotion_data")
        if emotion_data:
            frame = self.visualizer.draw(frame, [emotion_data], self.fps)
        
        # 統合分析結果をオーバーレイ
        overall_state = analysis.get("overall_state", {})
        contradictions = analysis.get("contradictions", [])
        trust_score = analysis.get("trust_score", 0)
        
        h, w = frame.shape[:2]
        
        # 半透明のオーバーレイ
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 180), (w - 10, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        y_offset = h - 160
        
        # 総合状態
        state_text = f"Overall: {overall_state.get('state', 'N/A')}"
        cv2.putText(frame, state_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # ストレススコア
        stress = overall_state.get('stress_score', 0)
        stress_color = (0, 255, 0) if stress < 30 else (0, 165, 255) if stress < 60 else (0, 0, 255)
        cv2.putText(frame, f"Stress: {stress:.1f}/100", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_color, 2)
        y_offset += 25
        
        # ポジティブ度
        positive = overall_state.get('positive_score', 50)
        cv2.putText(frame, f"Positive: {positive:.1f}/100", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # 信頼度
        trust_color = (0, 255, 0) if trust_score > 0.7 else (0, 165, 255) if trust_score > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Trust: {trust_score:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, trust_color, 2)
        y_offset += 25
        
        # 矛盾数
        if contradictions:
            cv2.putText(frame, f"Contradictions: {len(contradictions)}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            y_offset += 25
            
            # 最初の矛盾のタイプを表示
            if len(contradictions) > 0:
                contr_type = contradictions[0]['type']
                cv2.putText(frame, f"Type: {contr_type}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        return frame
    
    def _print_final_statistics(self):
        """最終統計を表示"""
        print("\n=== 統合分析統計 ===")
        summary = self.multimodal_analyzer.get_history_summary()
        
        print(f"総分析回数: {summary['count']}")
        print(f"平均ストレススコア: {summary['avg_stress']:.1f}")
        print(f"最大ストレススコア: {summary['max_stress']:.1f}")
        print(f"総矛盾検出数: {summary['total_contradictions']}")
        print(f"平均矛盾数: {summary['avg_contradictions']:.1f}")
        
        if self.speech_features:
            print(f"\n発話速度: {self.speech_features['speech_rate']:.2f} 単語/秒")
            print(f"ポーズ数: {self.speech_features['num_pauses']}")
            print(f"平均ポーズ時間: {self.speech_features['avg_pause_duration']:.2f}秒")
    
    def _process_video_with_realtime_audio(self, video_source):
        """動画処理とリアルタイムマイク入力を統合"""
        
        # マイク録音開始
        chunk_duration = self.config.get("whisper", {}).get("chunk_duration", 5.0)
        self.whisper.start_realtime_recording(chunk_duration=chunk_duration)
        
        cap = cv2.VideoCapture(video_source) if isinstance(video_source, (str, int)) else video_source
        
        try:
            last_speech_text = ""
            speech_timestamp = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # FPS計算
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time
                
                # 新しい音声認識結果をチェック
                if self.whisper.has_new_audio():
                    speech_result = self.whisper.get_latest_transcription()
                    if speech_result and speech_result.get("text", "").strip():
                        last_speech_text = speech_result["text"]
                        speech_timestamp = current_time
                        print(f"\n[音声認識] {last_speech_text}")
                
                # 表情認識
                emotion_results = self.emotion_detector.detect(frame)
                
                # 統合分析（最近の音声データがある場合）
                if emotion_results and last_speech_text and (current_time - speech_timestamp < 10.0):
                    # 最初の顔の表情データを取得
                    emotion_data = emotion_results[0] if emotion_results else None
                    
                    # マルチモーダル分析
                    analysis = self.multimodal_analyzer.analyze(
                        emotion_data=emotion_data,
                        speech_data={"text": last_speech_text},
                        timestamp=current_time
                    )
                    
                    # 感情変遷をログに記録
                    self._log_emotion_change(
                        emotion_data,
                        analysis.get("overall_state", {}),
                        has_speech=True
                    )
                    
                    # 分析結果を画面に描画
                    frame = self._draw_multimodal_results(frame, analysis)
                else:
                    # 表情のみ
                    frame = self.visualizer.draw(frame, emotion_results, self.fps)
                    
                    # 表情のみの場合もログに記録
                    if emotion_results:
                        emotion_data = emotion_results[0]
                        # 表情のみの状態評価を取得
                        emotion_only_state = self.multimodal_analyzer._emotion_only_state(emotion_data)
                        self._log_emotion_change(
                            emotion_data,
                            emotion_only_state,
                            has_speech=False
                        )
                
                # 音声テキストをオーバーレイ表示
                if last_speech_text and (current_time - speech_timestamp < 10.0):
                    self._draw_speech_text(frame, last_speech_text)
                
                # 画面表示
                if self.config["display"]["show_window"]:
                    cv2.imshow(self.config["display"]["window_title"], frame)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                
                self.frame_count += 1
                
                if self.frame_count % 100 == 0:
                    self.logger.info(f"Frame: {self.frame_count}, FPS: {self.fps:.1f}")
        
        finally:
            # マイク録音停止
            self.whisper.stop_realtime_recording()
            cap.release()
            cv2.destroyAllWindows()
            
            # 統計表示
            self._print_final_statistics()
    
    def _draw_speech_text(self, frame, text: str):
        """音声テキストをフレームに描画"""
        h, w = frame.shape[:2]
        
        # 半透明の背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)[:]
        
        # テキストを表示（長い場合は折り返し）
        max_chars = 60
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        cv2.putText(frame, f"Speech: {text}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(
        self,
        camera_id: Optional[int] = None,
        video_file: Optional[str] = None,
        audio_file: Optional[str] = None,
        realtime_audio: bool = False
    ):
        """
        アプリケーション実行
        
        Args:
            camera_id: カメラID
            video_file: 動画ファイル
            audio_file: 音声ファイル
            realtime_audio: リアルタイムマイク入力を使用するか
        """
        # 入力ソース決定
        if camera_id is not None:
            source = camera_id
            print(f"\nカメラ {camera_id} から入力")
        elif video_file:
            source = video_file
            print(f"\n動画ファイル: {video_file}")
        else:
            source = self.config["camera"]["device_id"]
            print(f"\nデフォルトカメラ {source} から入力")
        
        # 音声モード決定
        if realtime_audio:
            print("\nリアルタイムマイク入力モード")
            self._process_video_with_realtime_audio(source)
        elif audio_file:
            self.process_with_audio_file(source, audio_file)
        else:
            print("\n表情認識のみモード")
            self._process_video(source, with_audio=False)


def main():
    parser = argparse.ArgumentParser(
        description="マルチモーダル感情分析アプリケーション（表情 × 音声）"
    )
    parser.add_argument("--config", default="config.yaml", help="設定ファイル")
    parser.add_argument("--camera", type=int, help="カメラID")
    parser.add_argument("--video", type=str, help="動画ファイル")
    parser.add_argument("--audio", type=str, help="音声ファイル（統合分析用）")
    parser.add_argument("--realtime-audio", action="store_true", help="リアルタイムマイク音声入力を使用")
    
    args = parser.parse_args()
    
    app = MultiModalEmotionApp(args.config)
    app.run(args.camera, args.video, args.audio, args.realtime_audio)


if __name__ == "__main__":
    main()
