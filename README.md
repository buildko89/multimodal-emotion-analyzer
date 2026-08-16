# multimodal-emotion-analyzer
# マルチモーダル感情分析システム

**表情認識 × 音声認識 による統合感情分析**

OpenAI Whisper + MediaPipe を使用した、マルチモーダル（複数の入力形式）による高精度な感情・状態分析システムです。

## 概要

このシステムは、以下の2つのモダリティ（入力形式）を統合して、人の感情や心理状態をより正確に把握します：

1. **表情認識** (MediaPipe + 深層学習モデル)
   - 7種類の感情を検出（怒り、嫌悪、恐怖、幸せ、悲しみ、驚き、中立）
   - リアルタイム処理（30-60 FPS）

2. **音声認識** (OpenAI Whisper)
   - 発話内容のテキスト化
   - 単語レベルのタイムスタンプ
   - 言語自動検出

3. **マルチモーダル統合分析**
   - 表情と発言内容の相関分析
   - 矛盾検出（例：笑顔だがネガティブな発言）
   - 総合的な感情・ストレス状態の評価

## 主な機能

### 🎭 表情認識
- **7種類の表情分類**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **高速処理**: CPU環境で30-60 FPS
- **スムージング**: 時系列フィルタリングで安定した結果

### 🎤 音声認識  
- **高精度文字起こし**: Whisperによる多言語対応
- **発話特徴分析**:
  - 発話速度（単語/秒）
  - 沈黙・ポーズの検出
  - 発話時間比率

### 📊 テキスト感情分析
- **極性判定**: ポジティブ/ネガティブ/ニュートラル
- **感情分類**: 喜び、怒り、悲しみ、恐怖など
- **言語的特徴**:
  - 否定語の検出
  - フィラー（えっと、あの、など）の分析
  - 疑問文・繰り返しの検出

### 🔄 マルチモーダル統合
- **矛盾検出**:
  - 作り笑い（笑顔 × ネガティブ発言）
  - 感情の抑制（怒り × 穏やかな発言）
  - 虚偽（悲しい表情 × 「大丈夫」発言）

- **総合評価**:
  - ストレススコア (0-100)
  - ポジティブスコア (0-100)
  - 信頼度スコア (0.0-1.0)

- **推奨アクション**:
  - 状態に応じた具体的なアドバイス
  - 休憩やサポートの提案

## システム構成

```
multimodal-emotion-analyzer/
├── multimodal_app.py              # メインアプリケーション
├── emotion_detector_mediapipe.py  # MediaPipe版表情認識
├── config.yaml                    # 設定ファイル
├── requirements.txt               # 依存パッケージ
│
├── modules/                       # コアモジュール
│   ├── whisper_recognizer.py     # Whisper音声認識
│   ├── text_sentiment.py         # テキスト感情分析
│   └── multimodal_analyzer.py    # マルチモーダル統合分析
│
├── models/                        # 検出・認識モデル
│   └── detector.py
│
├── utils/                         # ユーティリティ
│   ├── visualizer.py             # 可視化
│   └── logger.py                 # ログ
│
└── trained_models/                # 学習済みモデル
    ├── emotion_models/           # 表情認識モデル
    └── mediapipe/                # MediaPipeモデル
```

## インストール

### 1. 必要な環境
- Python 3.8-3.12
- ffmpeg（音声処理用）

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主な依存パッケージ：
- `openai-whisper`: 音声認識
- `mediapipe`: 顔検出
- `opencv-python`: 画像処理
- `tensorflow/keras`: 表情認識
- `numpy`, `scipy`: 数値計算

### 3. ffmpegのインストール

**Windows (Chocolatey)**:
```bash
choco install ffmpeg
```

**Windows (Scoop)**:
```bash
scoop install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt update && sudo apt install ffmpeg
```

## 使い方

### 基本的な使用方法

#### 1. 表情認識のみ（カメラから）

```bash
python multimodal_app.py
```

#### 2. 動画 + 音声ファイルで統合分析

```bash
python multimodal_app.py --video video.mp4 --audio audio.wav
```

#### 3. カメラID指定

```bash
python multimodal_app.py --camera 1
```

#### 4. 設定ファイル指定

```bash
python multimodal_app.py --config my_config.yaml
```

### コマンドラインオプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--config` | 設定ファイルパス | `--config config.yaml` |
| `--camera` | カメラID | `--camera 0` |
| `--video` | 動画ファイルパス | `--video input.mp4` |
| `--audio` | 音声ファイルパス | `--audio speech.wav` |

### Whisper単体での音声認識

```python
from modules.whisper_recognizer import WhisperRecognizer

# 初期化
recognizer = WhisperRecognizer(model_name="base", language="ja")

# ファイルから認識
result = recognizer.transcribe_file("audio.mp3", word_timestamps=True)

print(f"認識テキスト: {result['text']}")
print(f"言語: {result['language']}")

# 音声特徴分析
features = recognizer.analyze_speech_features(result)
print(f"発話速度: {features['speech_rate']:.2f} 単語/秒")
print(f"ポーズ数: {features['num_pauses']}")
```

### テキスト感情分析

```python
from modules.text_sentiment import TextSentimentAnalyzer

analyzer = TextSentimentAnalyzer(language="ja")

text = "今日は疲れたけど楽しかったです"
result = analyzer.analyze_comprehensive(text)

print(f"極性: {result['sentiment']['polarity']}")
print(f"支配的感情: {result['sentiment']['dominant_emotion']}")
print(f"状態: {result['overall_state']['state']}")
```

### マルチモーダル統合分析

```python
from modules.multimodal_analyzer import MultiModalAnalyzer

analyzer = MultiModalAnalyzer()

# 表情 + 音声データで分析
result = analyzer.analyze(
    emotion_data={"emotion": "happy", "confidence": 0.89},
    speech_data={"text": "今日は本当に嬉しいです！"}
)

print(f"総合状態: {result['overall_state']['state']}")
print(f"ストレススコア: {result['overall_state']['stress_score']}")
print(f"矛盾数: {len(result['contradictions'])}")
print(f"信頼度: {result['trust_score']}")
```

## 設定

`config.yaml`で以下の項目を設定できます：

### Whisper設定

```yaml
whisper:
  model: "base"          # tiny, base, small, medium, large, turbo
  device: "cpu"          # cpu or cuda
  language: "ja"         # ja (日本語), en (英語), auto (自動)
  word_timestamps: true
```

### 顔検出設定

```yaml
face_detection:
  model: "mediapipe"     # mediapipe, yolov8, haarcascade
  mediapipe:
    min_detection_confidence: 0.5
```

### 表情認識設定

```yaml
emotion_recognition:
  model: "original"      # original, hsemotion, fer
  smoothing:
    enabled: true
    window_size: 6
```

## 応用例

### 1. メンタルヘルスモニタリング
- 長期的な感情変化の追跡
- ストレスレベルの可視化
- うつ傾向の早期発見

### 2. コミュニケーション分析
- 会議の雰囲気測定
- 話者のエンゲージメント評価
- チームの心理的安全性の評価

### 3. カスタマーサポート
- 顧客満足度のリアルタイム測定
- 不満の早期検出
- 対応品質の改善

### 4. 教育・学習支援
- 学習者の理解度把握
- 集中度モニタリング
- 個別サポートの最適化

## 出力データ

### 統合分析結果の構造

```python
{
    "timestamp": 1234567890.123,
    "emotion_data": {
        "emotion": "happy",
        "confidence": 0.89,
        "bbox": (x1, y1, x2, y2)
    },
    "speech_data": {
        "text": "今日は嬉しいです",
        "language": "ja"
    },
    "text_sentiment": {
        "polarity": "POSITIVE",
        "score": 0.85,
        "dominant_emotion": "joy"
    },
    "contradictions": [
        {
            "type": "fake_smile",
            "severity": "medium",
            "description": "笑顔だが、発言内容はネガティブ"
        }
    ],
    "overall_state": {
        "state": "良好",
        "stress_score": 25.5,
        "positive_score": 75.3,
        "contradiction_count": 1
    },
    "trust_score": 0.85,
    "recommendations": [
        "良好な状態です。現状を維持しましょう"
    ]
}
```

## トラブルシューティング

### MediaPipeのインストールエラー

```bash
pip uninstall mediapipe
pip install mediapipe==0.10.14
```

### Whisperモデルのダウンロードが遅い

初回実行時、モデルファイル（~150MB～1.5GB）がダウンロードされます。
より小さいモデル（`tiny`や`base`）の使用を推奨：

```yaml
whisper:
  model: "base"  # tiny (39M), base (74M), small (244M)
```

### GPU対応

CUDAがインストールされている場合：

```yaml
whisper:
  device: "cuda"
```

## パフォーマンス

### 処理速度（目安）

| モデル | CPU (Intel i5) | GPU (GTX 1060) |
|--------|---------------|----------------|
| 表情認識 | 30-60 FPS | 100+ FPS |
| Whisper (base) | 2-5x realtime | 10-20x realtime |

### メモリ使用量

- 表情認識: ~500MB
- Whisper (base): ~1GB
- 総メモリ: ~2-3GB

## ライセンス

このプロジェクトは以下のオープンソースプロジェクトを使用しています：

- [OpenAI Whisper](https://github.com/openai/whisper) - MIT License
- [MediaPipe](https://github.com/google/mediapipe) - Apache License 2.0
- その他の依存ライブラリについてはrequirements.txtを参照

## 参考資料

- [Whisperドキュメント](https://github.com/openai/whisper)
- [MediaPipeドキュメント](https://developers.google.com/mediapipe)
- [マルチモーダル感情認識の研究論文](https://arxiv.org/abs/2212.04356)

## 今後の拡張

- [ ] リアルタイムマイク入力対応
- [ ] 音響特徴量の追加分析（ピッチ、音量など）
- [ ] 視線追跡の統合
- [ ] 長期的な傾向分析とレポート生成
- [ ] Webダッシュボードの追加

## サポート

問題が発生した場合は、以下を確認してください：

1. `logs/`ディレクトリのログファイル
2. 依存パッケージのバージョン
3. Python環境（`python --version`）

---

**開発**: マルチモーダル感情分析プロジェクト  
**更新日**: 2026年1月8日
