# 11月実機デモ 運用手順書 (DEMO RUNBOOK)

文書番号: DEVAI-DEMO-RUNBOOK-20261026  
対象環境: NXP i.MX95 EVK + Kinara ARA-2 + PC オペレーター環境  
位置づけ: 展示デモ当日および事前リハーサルにおける実機操作・運用手順の正本  

---

## 1. 運用前提と準備チェックリスト

### 1.1 接続・電源
- [ ] i.MX95 EVK に電源（12V ACアダプタ）が供給され、Power LEDが点灯していること。
- [ ] USBカメラ（エレコム等 UVC準拠）が USB 3.0 ポートに接続されていること。
- [ ] USBマイク（サンワサプライ等 ALSA準拠）が接続されていること。
- [ ] HDMI ケーブルが i.MX95 HDMI OUT から展示用ディスプレイに接続され、画面が認識されていること。
- [ ] スピーカー（MQS オーディオ出力または USB オーディオ）が接続されていること。
- [ ] PC と i.MX95 が LAN ケーブル（または同一スイッチ/ルーター）で接続され、`ping 192.168.1.208` が通ること。

### 1.2 PC 側の操作場所
すべてのオペレーションは PC 側の PowerShell で以下のディレクトリから実行します：
```powershell
cd D:\source\repos\ara\devai_demo\operator
```

---

## 2. デモ開始前の実機健全性点検 (Pre-flight Doctor)

実機で余計なゾンビプロセスが残っていないか、常駐 LLM が ready か、ストレージ容量が十分かを確認します。

### 実行コマンド:
```powershell
python -u ara_demo_continuous_operator.py preflight --release ara_demo_20261026_v25 --session demo_continuous_YYYYMMDD_HHMMSS_preflight
```

### 正常合格基準:
- `doctor_ok: true`
- `health: "ready"`（Kinara ARA-2 上の Qwen2.5-7B が待機中）
- `root_free_bytes >= 201,326,592 bytes`（192MiB 起動基準クリア）
- `processes: []`（余分な古いプロセスがゼロ）
- `device_holders: []`（カメラ・マイクの競合ロックなし）

> [!CAUTION]
> もし `processes` に古い PID が残っている場合や、カメラがロックされている場合は、以下の安全クリーンアップを行ってから再試行してください。
> ```powershell
> python -u ara_demo_continuous_operator.py --release ara_demo_20261026_v24 --session demo_continuous_check_01 --cleanup
> ```

---

## 3. 自律常時稼働デモの開始 (Demo Start)

来場者を迎える前に、自律連続ローリング稼働を開始します。

### 実行コマンド:
```powershell
python -u ara_demo_continuous_operator.py start --release ara_demo_20261026_v25 --session demo_continuous_YYYYMMDD_HHMMSS_run01 --reply "現在、カメラ・マイク入力の準備ができています"
```
※ `YYYYMMDD_HHMMSS` は実行日時に置き換えてください（例: `demo_continuous_20261115_100000_run01`）。

### 起動後の自動挙動:
1. 実機 HDMI 画面にリアルタイムカメラ映像および HUD（感情メーター、アバターバッジ）が表示されます。
2. 8秒ごとに自動で窓が切り出され、表情認識・頭部姿勢・ASR・Qwen2.5-7B推論が自律回転します。
3. 来場者が話しかけると、画面下部に ASR 認識テキスト（字幕）が表示されます。
4. Qwen2.5-7B が表情と発話に応じた親しみやすい対話応答を生成し、画面下部のメッセージカードに表示されます。
5. ディスク上には常に直近3窓、各worker要求は直近2件のみが保持され、過去データは自動削除されます。
6. 連続運転上限は24時間です。正式受入では8時間以上の連続稼働を確認します。

---

## 4. 稼働中のステータス確認と画面キャプチャ (Monitoring)

### 4.1 リアルタイム稼働ステータス確認
別ターミナルから実機の回転状況・メモリ・ストレージ空き容量を確認できます：
```powershell
python -u ara_demo_continuous_operator.py status --release ara_demo_20261026_v25 --session demo_continuous_YYYYMMDD_HHMMSS_run01
```

### 4.2 実機 HDMI 画面のスクリーンショット取得
現在の来場者認識画面や LLM 応答画面を PC 側へ即座に取得・保存できます：
```powershell
python -u ara_demo_continuous_operator.py screenshot --release ara_demo_20261026_v25 --session demo_continuous_YYYYMMDD_HHMMSS_run01
```
- 保存先: `devai/devai_ara/runs/<session>/screenshots/screen_YYYYMMDD_HHMMSS.png`

---

## 5. デモの安全停止とログ回収 (Graceful Stop & Closeout)

展示終了時やセッション切り替え時は、必ず以下のコマンドで安全に停止します。

### 5.1 安全停止（Graceful Stop）
```powershell
python -u ara_demo_continuous_operator.py stop --release ara_demo_20261026_v25 --session demo_continuous_YYYYMMDD_HHMMSS_run01
```
- パイプラインが安全にループを終了し、カメラ・マイクのファイルディスクリプタを完全解放します。
- 常駐 LLM サービスは Native 側で待機状態（ready）を維持します。

### 5.2 実行ログ・証拠の完全回収（Offload）
```powershell
python -u ara_demo_continuous_operator.py collect --release ara_demo_20261026_v25 --session demo_continuous_YYYYMMDD_HHMMSS_run01
```
- 実機から全ログ、推論JSON、スクリーンショットを PC 側へ転送。
- 全ファイルの SHA-256 ハッシュ照合を行い、整合性を確認します。

---

## 6. トラブルシューティング

| 症状 | 原因の可能性 | 対処手順 |
|---|---|---|
| SSH 接続がタイムアウトする | IP アドレス不一致、LAN未接続 | `ping 192.168.1.208` を確認。イーサネットポートのリンクLEDを確認。 |
| HDMI 画面が真っ暗 | Weston サービス停止、ケーブル抜け | HDMI ケーブルを再挿入。SSH経由で `systemctl status weston` を確認。 |
| 音声認識（ASR）が拾わない | マイクのミュート、デバイス番号変更 | `python -u ara_audio_devices.py --check` を実行し、マイク入力を確認。 |
| LLM の応答が画面に出ない | タイムアウト設定、LLMサービス停止 | `curl http://127.0.0.1:18081/health` で `ready` を確認。停止時は `systemctl restart ara-llm`。 |
| ディスク空き容量警告 | ログ蓄積、古いリリースディレクトリ残存 | 不要な古い実行ディレクトリを削除、または `ara_demo_evidence_offload.py --clean-remote` を実行。 |
