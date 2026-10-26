# 観測API v2（計画4.7〜4.9）

`MultiModalAnalyzer.analyze()` は表情分類と発話テキストの観測を返します。本人の心理状態、意図、真偽、信頼性は算出しません。

## schema

| フィールド | 内容 |
|---|---|
| `schema_version` | `2` |
| `rule_version` | `facial-text-observations-v2` |
| `status` | `compared` または `unavailable` |
| `uncertainty` | 未知ラベル、低確率、入力不足などの理由 |
| `observations.face.face_emotion` | 共通7ラベル、または `unknown` |
| `observations.face.model_confidence` | backend分類確率。欠落・不正値はnull。校正済み正解確率や人物の信頼性を意味しない |
| `observations.face.input_quality` | 入力品質情報、未測定はnull |
| `observations.face.input_quality_status` | `reported` または `not_measured`。nullを良好と扱わない |
| `observations.face.model_id` | モデルID。入力にない場合はnull |
| `observations.text_polarity` | POSITIVE / NEGATIVE / NEUTRAL / UNKNOWN |
| `observations.text_method` | `japanese_keyword_rules` |
| `observations.acoustic_emotion` | `not_implemented` |
| `signal_mismatches` | 観測信号の不一致候補。判定不能時は空配列 |

既定の分類確率下限は0.3。アプリでは `emotion_recognition.smoothing.min_confidence` に合わせます。未知ラベル、非有限値、範囲外の確率、低確率、品質のunavailable、発話不足、キーワード根拠なしでは比較しません。表情だけ・テキストだけの観測値は残ります。ルールで矛盾が見つからないことは、信号の一致や人物の信頼性を保証しません。

テキスト分析は単純なキーワード検出です。`sentiment.score` はキーワード比率であり、モデルの確率ではありません。肯定・否定の根拠がない場合は `UNKNOWN` とnullを返します。否定の作用域や文脈を評価する言語モデルは未導入です。

## 4.7 ラベルと出力順

`angry, disgust, fear, happy, sad, surprise, neutral` を使用し、Anger、Happiness、Sadnessなど既知の表記を正規化します。推論結果は `score_labels` に元のclass順を、`canonical_scores` に共通ラベル別の値を持ちます。`Contempt`など未対応classの確率は `unknown` に保持し、7クラスへの再正規化はしません。平滑化は未知・低確率の結果を以前の分類で置き換えません。

HSEmotionのclass順はbackendの `idx_to_class` から取得します。MTL出力に付属するvalence/arousalは分類確率から分離し、入力cropをRGBへ変換します。[配布元の実装](https://github.com/av-savchenko/hsemotion/blob/main/hsemotion/facial_emotions.py)に基づくadapter修正であり、今回HSEmotionのモデル重みは導入・評価していません。

## 4.8 モダリティ

WhisperはASRとして文字起こしと時刻を提供します。そのテキストへ日本語キーワード規則を適用します。声の高さ、スペクトルなどに基づく音響感情分類は未実装です。フィラー、疑問表現、繰り返しから心理状態を推定せず、観測数を記録します。

## 4.9 旧APIからの変更

`contradictions`、`overall_state`、`trust_score`、心理状態のスコア、推奨介入を削除しました。旧フィールドの自動互換出力はありません。呼び出し側は `observations`、`status`、`uncertainty`、`signal_mismatches` を使用してください。

CSVは `output/observation_timeline_v2.csv` へ新しいヘッダーで書き込み、旧 `emotion_timeline.csv` へ追記しません。表情ラベル・状態・判定不能理由・発話有無の変化を記録します。確率だけの変化では行を増やしません。JSONLの同期区間結果も `schema_version: 2` を持ち、比較可能な区間に新APIの `analysis` を付加します。未知サンプルが一部でも含まれる区間、低確率、分類候補が同数の区間は統合不可です。

画面には分類ラベル、モデル出力確率、テキスト規則、比較状態、不一致候補を表示します。既存の録画経路では区間集計が終了時に確定するため、統合結果はJSONLへ保存します。録画中の表示は表情と人物条件です。

## PCでの再検証

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-pc.txt
.\.venv\Scripts\python.exe scripts/manage_models.py download mediapipe-face-short-range fer2013-mini-xception whisper-base
.\.venv\Scripts\python.exe scripts/smoke_models.py --audio --pipeline
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

lockはWindows/Python 3.12で検証した開発PC用です。i.MX95への持込みではtarget用依存構成を使用してください。モデルスモークテストは人工入力でのロード・推論・shape/有限値の確認です。実顔や発話に対する精度、遮蔽、対象環境の品質、ARA240/Neutronの性能は未評価です。
