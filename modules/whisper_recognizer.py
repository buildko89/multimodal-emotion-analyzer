"""
Whisper音声認識モジュール
リアルタイム音声認識と音声ファイルからのテキスト抽出
"""

import os
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import queue

import numpy as np
import whisper


class WhisperRecognizer:
    """Whisper音声認識クラス"""
    
    def __init__(self, model_name: str = "base", device: str = "cpu", language: str = "ja"):
        """
        初期化
        
        Args:
            model_name: Whisperモデル名 ("tiny", "base", "small", "medium", "large", "turbo")
            device: 実行デバイス ("cpu", "cuda")
            language: 認識言語 ("ja", "en", etc.)
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        
        print(f"Whisperモデル '{model_name}' をロード中...")
        self.model = whisper.load_model(model_name, device=device)
        print(f"Whisperモデルのロードが完了しました (device: {device})")
        
        # 音声バッファ（リアルタイム用）
        self.audio_buffer = []
        self.is_recording = False
        
    def transcribe_file(
        self, 
        audio_path: str,
        word_timestamps: bool = True,
        task: str = "transcribe"
    ) -> Dict:
        """
        音声ファイルから文字起こし
        
        Args:
            audio_path: 音声ファイルパス
            word_timestamps: 単語レベルのタイムスタンプを取得するか
            task: "transcribe" (文字起こし) または "translate" (英訳)
            
        Returns:
            認識結果の辞書
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")
        
        print(f"音声ファイルを認識中: {audio_path}")
        start_time = time.time()
        
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=word_timestamps,
            task=task,
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        print(f"認識完了 (所要時間: {elapsed_time:.2f}秒)")
        
        return self._format_result(result)
    
    def transcribe_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        word_timestamps: bool = True
    ) -> Dict:
        """
        音声データ（NumPy配列）から文字起こし
        
        Args:
            audio_data: 音声データ (float32, -1.0 ~ 1.0)
            sample_rate: サンプリングレート
            word_timestamps: 単語レベルのタイムスタンプを取得するか
            
        Returns:
            認識結果の辞書
        """
        # Whisperは16kHzを期待
        if sample_rate != 16000:
            # リサンプリングが必要な場合
            import scipy.signal
            audio_data = scipy.signal.resample(
                audio_data, 
                int(len(audio_data) * 16000 / sample_rate)
            )
        
        # パディング/トリミング（30秒）
        audio_data = whisper.pad_or_trim(audio_data)
        
        # メルスペクトログラムに変換
        mel = whisper.log_mel_spectrogram(audio_data).to(self.model.device)
        
        # 言語検出
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        # デコード
        options = whisper.DecodingOptions(
            language=self.language,
            without_timestamps=not word_timestamps
        )
        result = whisper.decode(self.model, mel, options)
        
        return {
            "text": result.text,
            "language": detected_lang,
            "language_probability": probs[detected_lang]
        }
    
    def get_word_timings(self, result: Dict) -> List[Dict]:
        """
        単語レベルのタイミング情報を抽出
        
        Args:
            result: transcribe結果
            
        Returns:
            単語タイミングのリスト
        """
        words = []
        
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words.append({
                            "word": word_info.get("word", ""),
                            "start": word_info.get("start", 0.0),
                            "end": word_info.get("end", 0.0),
                            "probability": word_info.get("probability", 0.0)
                        })
        
        return words
    
    def calculate_speech_rate(self, result: Dict) -> float:
        """
        発話速度を計算（単語/秒）
        
        Args:
            result: transcribe結果
            
        Returns:
            発話速度（単語/秒）
        """
        words = self.get_word_timings(result)
        
        if not words or len(words) < 2:
            return 0.0
        
        duration = words[-1]["end"] - words[0]["start"]
        if duration <= 0:
            return 0.0
        
        return len(words) / duration
    
    def extract_pauses(self, result: Dict, min_pause: float = 0.5) -> List[Dict]:
        """
        発話間の沈黙（ポーズ）を抽出
        
        Args:
            result: transcribe結果
            min_pause: 最小ポーズ時間（秒）
            
        Returns:
            ポーズ情報のリスト
        """
        words = self.get_word_timings(result)
        pauses = []
        
        for i in range(len(words) - 1):
            pause_duration = words[i + 1]["start"] - words[i]["end"]
            if pause_duration >= min_pause:
                pauses.append({
                    "start": words[i]["end"],
                    "end": words[i + 1]["start"],
                    "duration": pause_duration,
                    "before_word": words[i]["word"],
                    "after_word": words[i + 1]["word"]
                })
        
        return pauses
    
    def analyze_speech_features(self, result: Dict) -> Dict:
        """
        音声特徴を総合的に分析
        
        Args:
            result: transcribe結果
            
        Returns:
            分析結果の辞書
        """
        words = self.get_word_timings(result)
        pauses = self.extract_pauses(result)
        speech_rate = self.calculate_speech_rate(result)
        
        # 統計情報
        total_words = len(words)
        total_duration = words[-1]["end"] - words[0]["start"] if words else 0.0
        avg_pause_duration = np.mean([p["duration"] for p in pauses]) if pauses else 0.0
        max_pause_duration = max([p["duration"] for p in pauses]) if pauses else 0.0
        
        # 平均単語長
        avg_word_length = np.mean([len(w["word"]) for w in words]) if words else 0.0
        
        return {
            "total_words": total_words,
            "total_duration": total_duration,
            "speech_rate": speech_rate,
            "num_pauses": len(pauses),
            "avg_pause_duration": avg_pause_duration,
            "max_pause_duration": max_pause_duration,
            "avg_word_length": avg_word_length,
            "speaking_time_ratio": (total_duration - sum([p["duration"] for p in pauses])) / total_duration if total_duration > 0 else 0.0
        }
    
    def _format_result(self, result: Dict) -> Dict:
        """
        認識結果を整形
        
        Args:
            result: Whisperの生の認識結果
            
        Returns:
            整形された結果
        """
        formatted = {
            "text": result.get("text", ""),
            "language": result.get("language", self.language),
            "segments": []
        }
        
        # セグメント情報を整形
        if "segments" in result:
            for seg in result["segments"]:
                formatted_seg = {
                    "id": seg.get("id", 0),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", ""),
                    "words": []
                }
                
                # 単語情報を整形
                if "words" in seg:
                    for word in seg["words"]:
                        formatted_seg["words"].append({
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "probability": word.get("probability", 0.0)
                        })
                
                formatted["segments"].append(formatted_seg)
        
        return formatted


def test_whisper():
    """テスト関数"""
    print("=== Whisper音声認識モジュール テスト ===\n")
    
    # 初期化
    recognizer = WhisperRecognizer(model_name="base", device="cpu", language="ja")
    
    # テスト用の音声ファイルがあれば認識
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        result = recognizer.transcribe_file(test_file, word_timestamps=True)
        
        print(f"認識テキスト: {result['text']}")
        print(f"言語: {result['language']}")
        print(f"\nセグメント数: {len(result['segments'])}")
        
        # 音声特徴分析
        features = recognizer.analyze_speech_features(result)
        print(f"\n音声特徴:")
        print(f"  総単語数: {features['total_words']}")
        print(f"  発話速度: {features['speech_rate']:.2f} 単語/秒")
        print(f"  ポーズ数: {features['num_pauses']}")
        print(f"  平均ポーズ時間: {features['avg_pause_duration']:.2f}秒")
    else:
        print(f"テスト用音声ファイル '{test_file}' が見つかりません")
        print("実際の使用時には音声ファイルを指定してください")


if __name__ == "__main__":
    test_whisper()
