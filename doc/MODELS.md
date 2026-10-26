# モデル導入・検証手順

2026-10-04: 開発PCで以下の3モデルを取得しSHA-256を検証しました。`.venv`（Python 3.12）へ依存パッケージを導入し、MediaPipe/mini-XCEPTION/WhisperのCPU推論と録画アプリ経路を人工入力で確認しました。精度評価ではありません。

```powershell
.\.venv\Scripts\python.exe scripts/smoke_models.py --audio --pipeline
```

結果は `output/model_smoke.json`。PC用の検証環境は `requirements-lock-pc.txt` に記録しています。

このプロジェクトはモデルをソース管理へ含めません。`models/model_manifest.json` をモデル台帳とし、取得元、固定された版、配置先、整合性情報、入出力仕様、ライセンスを管理します。アプリ起動中の暗黙ダウンロードは行いません。

## 1. 台帳を確認する

プロジェクトルートで次を実行します。

```powershell
python scripts/manage_models.py list
```

既定の映像経路は次の二つを使います。

| ID | 用途 | 配置先 | 検証 |
|---|---|---|---|
| `mediapipe-face-short-range` | 顔検出 | `trained_models/mediapipe/blaze_face_short_range.tflite` | SHA-256 |
| `fer2013-mini-xception` | 7クラス表情分類 | `trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5` | 固定commitとSHA-256 |

音声を使う場合だけ `whisper-base`（約142 MiB）も導入します。OpenAI Whisper の公式ローダーと同じ SHA-256 を台帳に記録しています。

## 2. ダウンロードする

映像の既定モデル:

```powershell
python scripts/manage_models.py download --profile default
```

Whisper base:

```powershell
python scripts/manage_models.py download whisper-base
```

ネットワーク接続可能な開発PCで実行し、生成された `trained_models/` をオフライン環境へコピーできます。コピー後は必ず検証します。

```powershell
python scripts/manage_models.py check --profile default
python scripts/manage_models.py check whisper-base
```

失敗時に既存の正常なファイルを壊さないよう、一時ファイルへ取得して検証に合格した場合だけ置換します。モデルは `.gitignore` の対象です。

## 3. 入出力契約

### MediaPipe Face Detector

- 入力: OpenCV BGR画像をコード内でRGBへ変換
- 前後処理: MediaPipe Tasks API
- 出力: face bounding box。アプリ内では `(x1, y1, x2, y2)` に変換
- 用途: 顔crop生成。表情分類は行わない

### FER2013 mini-XCEPTION

- 入力: 顔cropをgrayscale化し64×64へresize
- dtype/layout: `float32`, NHWC `[1, 64, 64, 1]`
- 正規化: `x / 255`, 続けて `(x - 0.5) * 2`。範囲は `[-1, 1]`
- 出力: softmax 7要素
- 学習時の順序: `angry, disgust, fear, happy, sad, surprise, neutral`

このモデルはベースライン確立用です。FER2013上の既存精度を、対象カメラ・照明・対象集団での妥当性とみなしてはいけません。ARA240向け変換前に、固定した顔crop corpusでFP出力を保存します。

### Whisper base

- 入力: 16 kHz mono相当の音声（Whisper側でlog-Melへ変換）
- 出力: segment/word timestampを含む文字起こし
- 役割: ASR。声色に基づく音響感情分類器ではない

## 4. ライセンスと採用判定

台帳の `license` は配布元が示すライセンスの記録であり、製品利用の法的判断ではありません。特に学習データの条件、モデル重みの再配布、依存パッケージ、用途地域の規制は製品化前に別途確認します。

- MediaPipe: Apache-2.0。配布元とNOTICE要件を確認する。
- mini-XCEPTION配布リポジトリ: MIT。FER2013データ由来の制約を別途確認する。
- Whisper: MIT。モデルカードの用途・限界も確認する。

通常の `yolov8n.pt` はCOCO物体検出モデルであり顔検出モデルではありません。本プロジェクトでは `face_detection.model: yolov8` を指定すると明示的に停止します。顔専用YOLOを将来追加する場合は、別ID、固定重み、クラス順、ライセンス、精度評価を台帳へ追加します。

## 5. オフライン持込みの記録

ターゲットへコピーする際は、少なくとも次をリリース記録へ残します。

- `models/model_manifest.json` のGit commit
- `python scripts/manage_models.py check ...` の結果
- モデルファイルのSHA-256（台帳にないhistorical weightも持込み時に記録）
- 使用した設定ファイル
- 評価用crop/audio corpusの版
