"""
マルチモーダル感情分析統合アプリケーション
表情認識 × 音声認識のリアルタイム統合分析
"""

import argparse
import csv
import json
from datetime import datetime
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np
import yaml

# 既存モジュールをインポート
sys.path.insert(0, str(Path(__file__).parent))
from models.factory import create_emotion_detector
from modules.temporal_fusion import speech_windows, WindowJoiner
from modules.multimodal_analyzer import MultiModalAnalyzer
from utils.visualizer import Visualizer
from utils.logger import setup_logger


def _json_value(value):
    """Preserve raw detector arrays in JSON without stringifying measurements."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported observation value: {type(value).__name__}")

if TYPE_CHECKING:
    from modules.whisper_recognizer import WhisperRecognizer


class MultiModalEmotionApp:
    """マルチモーダル感情分析アプリケーション"""

    def __init__(self, config_path: str = "config.yaml", *, emotion_detector=None, observation_clients=None):
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
        self.emotion_detector = emotion_detector if emotion_detector is not None else create_emotion_detector(self.config)
        self.observation_clients = observation_clients

        # 2. 音声認識
        print("2. 音声認識モジュールは音声使用時に初期化します")
        self.whisper = None

        # 3. マルチモーダル統合分析
        print("3. マルチモーダル分析モジュールを初期化中...")
        self.multimodal_analyzer = MultiModalAnalyzer(min_confidence=float(
            self.config.get("emotion_recognition", {}).get("smoothing", {}).get("min_confidence", 0.3)))

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
        self.transcript_path = None
        self.observation_session = None
        if self.config.get("observation_integration", {}).get("enabled", False):
            if not self.config["output"]["save_results"]:
                raise ValueError("Observation integration requires saved results and a new output directory")
            self.config["output"]["save_frame_observations"] = True

        # 感情変遷ログ用
        self.prev_emotion_state = None
        self.emotion_log_file = self.output_dir / "observation_timeline_v2.csv"
        self._init_emotion_log()

        print("=== 初期化完了 ===\n")

    def _ensure_whisper(self) -> "WhisperRecognizer":
        """音声を使う経路だけで大きなASRモデルをロードする。"""
        if self.whisper is None:
            from modules.whisper_recognizer import WhisperRecognizer

            whisper_config = self.config.get("whisper", {})
            self.whisper = WhisperRecognizer(
                model_name=whisper_config.get("model", "base"),
                model_path=whisper_config.get("model_path"),
                device=whisper_config.get("device", "cpu"),
                language=whisper_config.get("language", "ja"),
            )
        return self.whisper

    def _init_emotion_log(self):
        if not self.config["output"]["save_results"]:
            return
        if not self.emotion_log_file.exists():
            with self.emotion_log_file.open('w', newline='', encoding='utf-8') as stream:
                csv.writer(stream).writerow([
                    "Schema_Version", "Timestamp", "DateTime", "Monotonic_NS", "Track_ID",
                    "Face_Label", "Model_Confidence", "Input_Quality_Status", "Status",
                    "Uncertainty", "Has_Transcript", "Model_ID",
                ])

    def _log_emotion_change(self, emotion_data, observation, has_speech):
        if not self.config["output"]["save_results"] or not emotion_data:
            return
        current = (observation["face_emotion"], observation["status"],
                   tuple(observation["uncertainty"]), has_speech)
        if self.prev_emotion_state == current:
            return
        with self.emotion_log_file.open('a', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerow([
                2, format(time.time(), '.3f'), datetime.now().isoformat(),
                emotion_data.get("monotonic_ns"), emotion_data.get("face_id"),
                observation["face_emotion"], observation["model_confidence"],
                observation["input_quality_status"], observation["status"],
                '|'.join(observation["uncertainty"]), has_speech, observation["model_id"],
            ])
        self.prev_emotion_state = current

    def _load_config(self, config_path: str) -> dict:
        """設定を読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as error:
            raise ValueError(f"Invalid configuration: {config_path}") from error

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
        integration = self.config.get("observation_integration", {})
        asr_backend = integration.get("asr_backend", "whisper") if integration.get("enabled") else "whisper"
        if asr_backend not in {"whisper", "sensevoice"}:
            raise ValueError("Supported ASR backends: whisper, sensevoice")
        overlap = integration.get('overlap_main_video', False)
        if overlap and not (integration.get('enabled') and integration.get('parallel_backends')
                            and asr_backend == 'sensevoice'
                            and self.config['output'].get('save_frame_observations')):
            raise ValueError('overlap_main_video requires parallel SenseVoice and saved frame observations')
        self.processing_timing = {'mode': 'overlap' if overlap else 'asr_then_video',
                                  'started_ns': time.monotonic_ns()}
        if integration.get("enabled"):
            from modules.observation_integration import ObservationIntegrationSession
            if not isinstance(video_source, str):
                raise ValueError("Observation integration currently requires recorded video and actual file audio")
            self.observation_session = ObservationIntegrationSession(self.config, self.output_dir, video_source, audio_file,
                                                                    clients=getattr(self, 'observation_clients', None))
            audio_file = str(self.observation_session.common_audio)
        self.processing_timing['session_ready_ns'] = time.monotonic_ns()
        try:
            if overlap:
                # Worker requests already run in parallel. Resolve ASR only after
                # the video loop, then join its saved observations at original PTS.
                self._process_video(video_source, with_audio=True,
                                    resolve_speech=lambda: self._resolve_file_speech(asr_backend, audio_file))
            else:
                self._resolve_file_speech(asr_backend, audio_file)
                self._process_video(video_source, with_audio=True)
        finally:
            self.processing_timing['completed_ns'] = time.monotonic_ns()
            if self.config['output']['save_results']:
                (self.output_dir/'pipeline_timing.json').write_text(
                    json.dumps(self.processing_timing, indent=2)+'\n', encoding='utf-8')

    def _resolve_file_speech(self, asr_backend, audio_file):
        self.processing_timing['asr_wait_started_ns'] = time.monotonic_ns()
        if asr_backend == "sensevoice":
            self.speech_result = self.observation_session.sensevoice_transcript()
            self.speech_features = {"speech_rate": None, "total_words": None, "status": "unavailable",
                                    "method": "not_available_without_word_timestamps"}
        else:
            whisper = self._ensure_whisper()
            self.speech_result = whisper.transcribe_file(
                audio_file,
                word_timestamps=self.config.get("whisper", {}).get("word_timestamps", True),
                task=self.config.get("whisper", {}).get("task", "transcribe"),
            )
            self.speech_features = whisper.analyze_speech_features(self.speech_result)
            self.speech_result["asr_backend"] = "whisper"
        self.processing_timing['asr_ready_ns'] = time.monotonic_ns()
        if self.config["output"]["save_results"]:
            with (self.output_dir / "speech_result.json").open("w", encoding="utf-8") as stream:
                json.dump({"source_audio": str(audio_file), "result": self.speech_result,
                           "features": self.speech_features}, stream, ensure_ascii=False,
                          indent=2, default=_json_value, allow_nan=False)

        print(f"認識テキスト: {self.speech_result['text']}")
        rate = self.speech_features['speech_rate']
        if rate is None:
            print("発話速度・単語数: 未測定（単語時刻なし）\n")
        else:
            print(f"発話速度: {rate:.2f} 単語/秒")
            print(f"総単語数: {self.speech_features['total_words']}\n")

    def _process_video(self, video_source, with_audio: bool = False, *, resolve_speech=None):
        """動画処理メインループ"""

        cap = cv2.VideoCapture(video_source) if isinstance(video_source, (str, int)) else video_source
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")
        is_file = isinstance(video_source, str)
        if with_audio and not is_file:
            cap.release()
            raise ValueError("File audio/transcript requires --video with a shared media timeline")
        origin_ns = time.monotonic_ns()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if is_file and (not np.isfinite(fps) or fps <= 0):
            cap.release()
            raise ValueError("Video requires a valid FPS for timestamp fallback")
        joiner = WindowJoiner(speech_windows(
            self.speech_result or {}, origin_ns,
            float(self.config.get("synchronization", {}).get("audio_offset_seconds", 0)),
        ), min_confidence=float(self.config.get("emotion_recognition", {}).get(
            "smoothing", {}).get("min_confidence", 0.3))) if with_audio and resolve_speech is None else None
        processed_until_ns = origin_ns
        index = 0
        last_pts = -1.0
        pts_fallbacks = 0
        processing_start_ns = time.monotonic_ns()
        timing = getattr(self, 'processing_timing', {})
        timing['video_loop_started_ns'] = processing_start_ns
        completion = "failed"
        frame_log = None

        try:
            if self.config["output"]["save_results"] and self.config["output"].get("save_frame_observations", False):
                frame_log = (self.output_dir / "frame_observations.jsonl").open("w", encoding="utf-8")
            while True:
                ret, frame = cap.read()
                if not ret:
                    completion = "end_of_stream"
                    break

                # FPS計算
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time

                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file else 0.0
                timestamp_source = "capture_monotonic"
                if is_file:
                    valid_pts = np.isfinite(pts) and pts > last_pts
                    timestamp_source = "opencv_pts" if valid_pts else "frame_index_fps_fallback"
                    if not valid_pts:
                        pts_fallbacks += 1
                    pts = pts if valid_pts else index / fps
                    if pts <= last_pts:
                        raise ValueError("Video timestamps must increase")
                    last_pts = pts
                timestamp_ns = origin_ns + round(pts * 1e9) if is_file else time.monotonic_ns()
                processed_until_ns = timestamp_ns + (round(1e9 / fps) if is_file else 0)
                index += 1
                # 表情認識
                inference_started_ns = time.monotonic_ns()
                emotion_results = self.emotion_detector.detect(frame)
                inference_completed_ns = time.monotonic_ns()
                for face in emotion_results:
                    face["monotonic_ns"] = timestamp_ns
                if frame_log is not None:
                    frame_log.write(json.dumps({
                        "schema_version": 1, "frame_index": index - 1,
                        "source_pts_seconds": pts if is_file else None,
                        "media_aligned_monotonic_ns": timestamp_ns,
                        "timestamp_source": timestamp_source,
                        "inference_started_ns": inference_started_ns,
                        "inference_completed_ns": inference_completed_ns,
                        "face_count": len(emotion_results), "faces": emotion_results,
                    }, ensure_ascii=False, default=_json_value, allow_nan=False) + "\n")
                if joiner:
                    joiner.observe(timestamp_ns, emotion_results)
                status = "single_person" if len(emotion_results) == 1 else (
                    "multiple_faces: fusion paused" if emotion_results else "no_face: fusion paused")
                if len(emotion_results) != 1:
                    emotion_results = []

                frame = self.visualizer.draw(frame, emotion_results, self.fps)
                if len(emotion_results) == 1:
                    emotion_data = emotion_results[0]
                    emotion_only_state = self.multimodal_analyzer._emotion_only_state(emotion_data)
                    self._log_emotion_change(emotion_data, emotion_only_state, has_speech=False)

                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 165, 255), 1)
                # 画面表示
                if self.config["display"]["show_window"]:
                    cv2.imshow(self.config["display"]["window_title"], frame)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        completion = "user_stopped"
                        break

                self.frame_count += 1

                if self.frame_count % 100 == 0:
                    self.logger.info(f"Frame: {self.frame_count}, FPS: {self.fps:.1f}")

        finally:
            timing['video_loop_completed_ns'] = time.monotonic_ns()
            if frame_log is not None:
                frame_log.close()
            cap.release()
            if self.config["display"]["show_window"]:
                cv2.destroyAllWindows()

            # 統計表示
            if with_audio and (resolve_speech is None or completion == 'end_of_stream'):
                if resolve_speech is not None:
                    resolve_speech()
                    joiner = WindowJoiner(speech_windows(
                        self.speech_result or {}, origin_ns,
                        float(self.config.get('synchronization', {}).get('audio_offset_seconds', 0))),
                        min_confidence=float(self.config.get('emotion_recognition', {}).get(
                            'smoothing', {}).get('min_confidence', 0.3)))
                    # Replay measurements, never image inference or a new input.
                    with (self.output_dir/'frame_observations.jsonl').open(encoding='utf-8') as stream:
                        for line in stream:
                            observation = json.loads(line)
                            joiner.observe(observation['media_aligned_monotonic_ns'], observation['faces'])
                timing['fusion_started_ns'] = time.monotonic_ns()
                events = list(joiner.results(processed_until_ns))
                for event in events:
                    if event["status"] == "aligned":
                        event["analysis"] = self.multimodal_analyzer.analyze(
                            emotion_data=event["emotion_data"], speech_data=event["speech_data"],
                            timestamp=event["window_end_ns"] / 1e9,
                        )
                if self.config["output"]["save_results"]:
                    with (self.output_dir / "fusion_windows.jsonl").open("w", encoding="utf-8") as stream:
                        for event in events:
                            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._print_final_statistics()
                timing['fusion_completed_ns'] = time.monotonic_ns()
            if self.config["output"]["save_results"]:
                with (self.output_dir / "video_summary.json").open("w", encoding="utf-8") as stream:
                    json.dump({"schema_version": 1, "source": str(video_source),
                               "source_kind": "file" if is_file else "capture",
                               "timeline_origin_ns": origin_ns,
                               "timestamp_basis": "origin_plus_media_pts" if is_file else "capture_monotonic",
                               "fps": fps if is_file else None, "frames_read": index,
                               "frames_processed": self.frame_count,
                               "processed_until_ns": processed_until_ns,
                               "pts_fallback_count": pts_fallbacks, "completion": completion,
                               "processing_wall_seconds": (time.monotonic_ns() - processing_start_ns) / 1e9},
                              stream, indent=2, allow_nan=False)
            session = getattr(self, "observation_session", None)
            if session is not None and completion == "end_of_stream":
                summary = session.finish(self.speech_result or {},
                                         self.config["observation_integration"].get("asr_backend", "whisper"))
                timing['envelope_ready_ns'] = time.monotonic_ns()
                self.logger.info("Observation integration: %s; backend health: %s; speaker verified: 0",
                                 summary["observations_by_kind"], summary["backend_health"])
                cooperative = self.config.get("cooperative_ai", {})
                if cooperative.get("enabled", False):
                    self.analyze_saved_observations(self.output_dir / "observation_envelope_v1.json",
                                                   self.output_dir / "cooperative_ai", cooperative)

    @staticmethod
    def analyze_saved_observations(envelope_path, output_dir, configuration):
        """Run recognition/response on recorded, validated machine observations."""
        from modules.driver_response_policy import run_saved_observations
        cycle = run_saved_observations(envelope_path, output_dir, configuration)
        print(f"協調AI: recognition={cycle['recognition']['status']}, policy={cycle['policy']['policy']}, "
              f"response={cycle['response']['accepted']['text']!r}")
        return cycle

    def _draw_multimodal_results(self, frame, analysis):
        face = analysis.get("observations", {}).get("face") or {}
        confidence = face.get("model_confidence")
        lines = [
            "Comparison: " + analysis.get("status", "unavailable"),
            "Face label: " + face.get("face_emotion", "unknown"),
            "Model confidence: " + (format(confidence, '.2f') if confidence is not None else 'unavailable'),
            "Text polarity (rules): " + analysis.get("observations", {}).get("text_polarity", "UNKNOWN"),
            "Signal mismatches: " + str(len(analysis.get("signal_mismatches", []))),
            "Acoustic emotion: not implemented",
        ]
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, max(0, h - 180)), (w - 10, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (20, max(20, h - 160) + index * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _print_final_statistics(self):
        summary = self.multimodal_analyzer.get_history_summary(last_n=1000)
        print("\n=== 観測比較統計（最大1000区間） ===")
        print(f"分析区間数: {summary['count']}")
        print(f"比較可能: {summary['compared_count']}")
        print(f"判定不能: {summary['unavailable_count']}")
        print(f"観測信号の不一致候補: {summary['total_signal_mismatches']}")

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
        if self.transcript_path:
            if not video_file or audio_file or realtime_audio:
                raise ValueError("--transcript requires --video and cannot be combined with audio inputs")
            with open(self.transcript_path, encoding="utf-8") as stream:
                self.speech_result = json.load(stream)
            self._process_video(source, with_audio=True)
            return
        if realtime_audio:
            if video_file or audio_file or not self.config.get('live', {}).get('enabled'):
                raise ValueError('Live audio requires explicit live configuration and camera input')
            from modules.live_pipeline import run_live
            run_live(self.config)
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
    parser.add_argument("--transcript", help="Whisper-compatible timestamped segment JSON")
    parser.add_argument("--headless", action="store_true", help="Disable window display")
    parser.add_argument("--realtime-audio", action="store_true", help="リアルタイムマイク音声入力を使用")

    args = parser.parse_args()

    if args.realtime_audio:
        if args.video or args.audio or args.transcript:
            parser.error('--realtime-audio requires camera input without file audio/transcript')
        with open(args.config, encoding='utf-8') as stream:
            configuration = yaml.safe_load(stream)
        if not configuration.get('live', {}).get('enabled'):
            parser.error('Explicit live.enabled configuration is required')
        if args.camera is not None:
            configuration['live']['camera'] = args.camera if configuration['live'].get('capture_backend')=='windows' else f'/dev/video{args.camera}'
        from modules.live_pipeline import run_live
        run_live(configuration)
        return

    app = MultiModalEmotionApp(args.config)
    app.transcript_path = args.transcript
    if args.headless:
        app.config["display"]["show_window"] = False
    app.run(args.camera, args.video, args.audio, args.realtime_audio)


if __name__ == "__main__":
    main()
