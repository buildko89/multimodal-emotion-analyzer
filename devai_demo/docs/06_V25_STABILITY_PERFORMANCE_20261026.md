# V25 展示デモ安定性・性能改善記録

更新日: 2026-10-26 (Asia/Tokyo)  
対象: i.MX95 + ARA-2 / 固定リリース `ara_demo_20261026_v25`

## 確定したV24の停止原因

回収済みセッション `demo_continuous_20261026_111500_v24` は、開始後
559.10秒で `minimum_disk_headroom` が発火して正常系の安全停止へ移行した。
終了時の入力障害はstorageのみで、プロセスクラッシュではない。

- 保持入力窓: 3件、約15.1MB
- 未削除worker要求: backendごとに12件
- Visual worker要求: 約31.2MB
- SenseVoice worker要求: 約0.2MB
- Emotion2Vec worker要求: 約0.6MB
- V24の通常運転上限: `duration_seconds=600`（10分）

したがって「数十分以内に落ちる」現象には、worker要求の無制限蓄積と
10分の固定終了上限という、8時間展示要件に反する2つの原因があった。

## V24実測の処理遅延

`live_000050`（8秒窓）の実測値:

- SenseVoice: 20.92秒
- Visual worker: 22.87秒
- Emotion2Vec: 24.61秒
- ARA応答を含む窓処理全体: 37.86秒
- 80入力フレーム、Visualの重いFace/Hand推論は40回

8秒ごとに取得しても約38秒間処理するため、待ち行列は最新窓だけを残して
古い窓を破棄していた。V24はリアルタイム認識ではなく、継続入力に対する
低頻度バッチ応答である。

## V25の変更

1. `worker_pool/{sensevoice,emotion2vec,visual}` のUUID要求フォルダを、
   各backend直近2件だけに制限する。成果物は各入力窓へhash確認付きで
   コピー済みのため、transport一時フォルダを削除しても窓証拠は保持される。
2. 通常運転上限を600秒から86,400秒（24時間）へ変更した。
3. Visual workerのFace Landmarker / Hand Landmarker実推論を2フレームに1回から
   4フレームに1回へ変更した。全フレームの時刻付き行は保持し、間のフレームは
   直前結果を引き継ぐ。8秒・80フレームなら実推論は40回から20回になる。
4. V24からのハードリンク共有配備を実装し、V25初回差分は71,787 bytesに抑えた。
5. 常駐ARAサービスのロードスレッドだけが終了してhealthが永久に`loading`となる
   問題を修正した。ロード失敗時はプロセスを非0終了し、systemdが5秒後に再試行する。
   5分間3回の上限を設け、無制限再起動は行わない。
6. 中央管理されたoperatorがワークスペースのknown-hostsと証拠保存先を正しく参照する
   よう、ルート探索を修正した。運用手順のCLI構文も実装と一致させた。

## 配備・常駐サービス確認

- V25 doctor: `ok=true`
- `bad_hashes=[]`, `missing_dependencies=[]`, `changed_dependencies=[]`
- 音声入出力: named ALSA PCMを正常解決
- 常駐Qwen: `ready`
- ロード時間: 96.655秒
- systemd再起動回数: 0
- ロード後Swap使用: 約0.3MB

## 録音なし固定素材check結果

セッション: `demo_continuous_20261026_230000_v25_check`

- `ready_reached=true`, `error=null`, `cleanup_ok=true`
- Visual / SenseVoice / Emotion2Vec: すべて`ok`
- 知覚フェーズ: 78.26秒（V24記録103.6秒から約25.3秒短縮）
- 準備全体: 86.22秒
- HDMI ready: 起動要求から110.88秒
- ARA初期化確認: 5.43秒
- ARA 10秒境界確認: 2.51秒
- 新録音窓: 0
- 終了後device holder: 0
- worker process: 全終了
- worker pool: 各backend 1要求（上限2以下）
- 実機一時領域で4 UUID要求を作る境界検査を行い、古い2件だけを削除、
  新しい2件とpersistent logを保持することを確認
- 回収: 73ファイル、16,082,432 bytes、remote前後とPCコピーの全hash一致
- 実機証拠は3コピーを検証後にoffloadし、実機空き容量を224,022,528 bytesへ回復

次の実入力8時間試験セッション
`demo_continuous_20261026_231500_v25_soak` は完全preflightに合格済み。
新録音を開始する前に、本人の現在のカメラ・マイク入力準備確認を行う。

## 未完了の正式受入

- 本人の現在準備確認後の実機カメラ・マイク入力
- Visual stride 4での顔・表情・頭部・手認識品質確認
- ASRおよび総応答遅延の再計測
- ディスク／RAM／process／デバイス保持を監視した8時間以上の連続運転

上記が完了するまで「リアルタイム達成」「8時間連続運転達成」とは記録しない。
