# 単一人物・発話区間同期のPC検証

録画入力では動画のPTS（取得できない場合はframe index / FPS）を、処理開始時の単調時計を原点として変換します。ASRの秒単位segmentも同じ原点へ変換します。処理速度やASR完了時刻は入力の再生時刻に影響しません。

音声と動画が同時開始なら `synchronization.audio_offset_seconds: 0.0`。音声の0秒が動画の2秒なら `2.0` を指定します。同期した素材を利用者が用意する前提で、自動同期は行いません。

```powershell
python multimodal_app.py --video sample.mp4 --audio sample.wav --headless
```

Whisperの導入前にも、次の形式のJSONを使って同じ処理を検証できます。

```json
{"language":"ja","segments":[{"start":1.0,"end":3.0,"text":"今日は楽しいです"}]}
```

```powershell
python multimodal_app.py --video sample.mp4 --transcript segments.json --headless
```

結果は `output/fusion_windows.jsonl`。各segmentの半開区間 `[start, end)` 内だけから表情ラベル分布と信頼度を集計します。画面内の顔が複数、顔未検出、track切替、未知ラベル、動画の早期終了の場合は `unavailable` と理由を記録し、統合ルールへ渡しません。無発話区間へ全文を適用する旧動作は使用しません。

画面には `single_person` / `multiple_faces: fusion paused` / `no_face: fusion paused` を表示します。画面に1人でも画面外の話者は排除できません。結果の `speaker_association` は単一人物前提を示し、`speaker_id` はnullです。話者分離・本人確認を行った結果ではありません。

この段階で同期を実装したのは録画＋音声/事前文字起こし経路です。従来の `--realtime-audio` は実装されていない録音メソッドを呼んでいたため、説明付きで停止します。マイク取得の単調時刻、bounded queue、遅延ASR結果と過去映像の結合は次段の作業です。

2026-10-04: この開発PCでは3モデルを取得し、SHA-256検証とローカルCPU推論を完了。`scripts/smoke_models.py --audio` で再検証できます。実機NPU/ARA評価は別段階です。

統合結果の `analysis` は [観測schema v2](OBSERVATION_SCHEMA.md) を使用します。未知ラベルを含む区間、低確率、同数の分類候補も判定不能です。
