# multimodal-emotion-analyzer
# マルチモーダル感情分析システム (ARA + i.MX95 実機展示デモ統合リポジトリ)

> [!IMPORTANT]
> **11月実機展示デモ（Kinara ARA-2 + NXP i.MX95）向けアセット集約:**  
> 本リポジトリの [`devai_demo/`](devai_demo/) フォルダに、展示会向け自律常時稼働パイプライン、HDMI HUD 表示（Meiryoフォント・豆腐ゼロ）、Qwen2.5-7B 車載相棒対話、PC側運用スクリプト、当日運用手順書（Runbook）を一元集約しています。  
> 詳細は [`devai_demo/README.md`](devai_demo/README.md) を参照してください。

**表情認識 × 音声認識 による統合感情分析**

OpenAI Whisper + MediaPipe を使用した、表情分類結果と発話テキストの観測・比較用プロトタイプです。心理状態、虚偽、医療的ストレスを判定するものではありません。

## 概要

このシステムは、以下の2つのモダリティ（入力形式）から得た観測結果を並べ、評価用ログを作成します：

1. **表情認識** (MediaPipe + 深層学習モデル)
   - 7種類の感情を検出（怒り、嫌悪、恐怖、幸せ、悲しみ、驚き、中立）
   - 実効FPSはハードウェアと入力条件ごとに計測

2. **音声認識** (OpenAI Whisper)
   - 発話内容のテキスト化
   - 単語レベルのタイムスタンプ
   - 言語自動検出

3. **マルチモーダル統合分析**
   - 表情と発言内容の相関分析
   - 観測信号の不一致候補（例：happy分類とNEGATIVEテキスト極性）
   - 表情分類とテキスト極性の不一致を示す試験的ルール

## 主な機能

### 🎭 表情認識
- **7種類の表情分類**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **処理性能**: 使用機材と入力条件ごとに実測
- **スムージング**: 時系列フィルタリングで安定した結果

### 🎤 音声認識
- **文字起こし**: Whisperによる多言語対応（用途・言語ごとの精度評価が必要）
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
- **観測信号の比較**:
  - 表情分類とテキスト極性の組合せを観測
  - 不一致候補を表示（内面、真偽、意図は推定しない）

- **試験的な統合出力**:
  - schema v2: 表情分類、モデル出力確率、入力品質の測定状態、テキスト極性、不一致候補
  - 未知ラベル・低確率・入力不足は判定不能。心理状態のスコアは出力しない
  - [観測APIと移行手順](doc/OBSERVATION_SCHEMA.md)

- **用途境界**:
  - 診断、虚偽判定、採用・査定、懲戒、安全上の自動介入には使用しない

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
- 開発PC検証環境: Python 3.12
- ffmpeg（音声処理用）

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. モデルの導入（必須）

モデルはリポジトリに同梱されず、実行時にも暗黙取得しません。映像用モデルを取得して整合性を検証します。

```bash
python scripts/manage_models.py download --profile default
python scripts/manage_models.py check --profile default
```

音声を使う場合はWhisperも取得します。

```bash
python scripts/manage_models.py download whisper-base
python scripts/manage_models.py check whisper-base
```

取得元、ライセンス、チェックサム、前処理、出力ラベル順は [doc/MODELS.md](doc/MODELS.md) を参照してください。

主な依存パッケージ：
- `openai-whisper`: 音声認識
- `mediapipe`: 顔検出
- `opencv-python`: 画像処理
- `tensorflow/keras`: 表情認識
- `numpy`, `scipy`: 数値計算

### 4. ffmpegのインストール

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

単一人物制約・発話区間の同期と、Whisper不要の事前文字起こしによる検証は
[TIMESTAMP_SYNC.md](doc/TIMESTAMP_SYNC.md) を参照してください。

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

### PC基準実行の記録（ARA-01）

録画＋実音声の実行では`speech_result.json`に実ASR結果、`video_summary.json`に媒体原点・フレーム数・終了状態・映像処理時間を保存します。設定の`output.save_frame_observations: true`を指定すると、`frame_observations.jsonl`へ全フレームの元PTS・frame index・推論時刻・全顔の生scoreを保存します。顔なし・複数顔も記録し、画面表示用に顔を絞る処理とは区別します。

`media_aligned_monotonic_ns`は録画時にorigin＋媒体PTSを表します。`inference_started_ns`／`inference_completed_ns`は実処理のmonotonic時刻です。媒体時刻の一致、音声話者の対応、絶対口／音同期の確認を別に扱ってください。

モデルスモークの過去結果を保持するには新しい出力先を指定します。

```powershell
python scripts/smoke_models.py --audio --pipeline --output-dir <new_output_directory>
```

このワークスペースでの基準実行・固定環境・再現手順は[ARA-01結果](../devai/devai_ara/ARA01_PC_BASELINE_RESULT.md)を参照してください。基準C0には画面内発話／画面外音声／手と口元／頷きのclassifier、SER・状態・LLM・応答はまだ接続されていません。

### コマンドラインオプション一覧

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
print(f"方法: {result['method']}")
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

print(f"比較状態: {result['status']}")
print(f"観測: {result['observations']}")
print(f"不一致候補: {result['signal_mismatches']}")
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
  model: "mediapipe"     # mediapipe, haarcascade, retinaface
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

## 評価用途と保存

同意を得た単一人物の録画を使い、顔追跡、表情分類、ASR、時刻同期、観測比較の動作を評価します。WhisperはASR、テキスト側は日本語キーワード規則であり、音響感情認識は未実装です。

`output.save_results: true` は表情分類CSVと、発話テキストを含むJSONLをPCへ保存します。取得前に利用目的・同意・保存期間・削除方法を対象者へ明示してください。自動削除機能はありません。保存を止める場合は `false` に変更し、既存データは保存先フォルダーの対象ファイルを削除します。

## 出力データ

観測APIは `schema_version: 2`。分類確率と入力品質、判定不能理由、不一致候補を返します。完全な構造と旧APIからの変更は [OBSERVATION_SCHEMA.md](doc/OBSERVATION_SCHEMA.md) を参照してください。

```python
{
    "schema_version": 2,
    "status": "compared",
    "observations": {
        "face": {"face_emotion": "happy", "model_confidence": 0.89,
                 "input_quality_status": "not_measured"},
        "text_polarity": "NEGATIVE",
        "text_method": "japanese_keyword_rules",
        "acoustic_emotion": "not_implemented"
    },
    "uncertainty": [],
    "signal_mismatches": [{"type": "facial_text_polarity_mismatch",
                           "face_emotion": "happy", "text_polarity": "NEGATIVE"}]
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

## ARA-02の録画観測統合（2026-10-11）

元 `MultiModalEmotionApp` の録画＋実音声経路に、既存のFace／Hand Landmarker、SenseVoice／emotion2vecを別環境のworkerから接続した。Whisper基準とSenseVoice ASRを切替でき、同じ16kHz PCM・元PTS・人物対応不明・品質・根拠ファイルを共通envelopeへ保存する。既定設定では無効。

[実装契約](../devai/devai_ara/OBSERVATION_ENVELOPE_V1.md)、[結果と未達](../devai/devai_ara/ARA02_OBSERVATION_INTEGRATION_RESULT.md)、[新runでの再実行](../devai/devai_ara/runs/ara02_observation_integration_20261011_v1/README.md)。元3動画720f・45回帰とASR切替を確認済み。4質問classifier・driver状態・LLM／短文応答・実時間capture・ARA移植は後続タスク。今回のworkerはcold起動で、実時間性能の達成値ではない。

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
