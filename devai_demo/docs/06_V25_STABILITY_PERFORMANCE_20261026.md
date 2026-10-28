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
7. モデルready直後に自動で実入力窓を作っていたため、ready後の本人確認を待つ
   `post_ready_start_gate`とoperatorの`arm`操作を追加した。ゲート待機中もプレビュー用
   デバイス取得は行うが、認識窓は作成しない。

## 2026-10-28 リアルタイム優先プロファイル

前回実入力では8秒窓に対して処理全体35.92秒、SenseVoice 20.59秒、
Visual 18.11秒、Emotion2Vec 24.09秒、ARA 10.01秒だった。処理中に
最新1件以外を捨てるため、本人発話を含む窓も落ち得ることを確認した。

V25の番号と配備先を変えず、次のリアルタイム優先設定を追加した。

1. 実入力窓を8秒から4秒へ短縮する。
2. 待機キューは最大2件とし、音声RMSが最大の窓と最新窓を保持する。
   単なる最新1件上書きによる発話窓の消失を防ぐ。
3. 起動時準備では従来どおりSenseVoice、Emotion2Vec、Visualを確認する。
   実入力経路ではSenseVoiceを4 threadで優先し、Emotion2Vecを待たない。
   したがって、この経路を完全な音響感情評価として扱わない。
4. ARA生成待ちは10秒から4秒へ短縮する。時間内に生成できない場合は、
   検証済みの短いfallbackを表示し、認識結果の表示を不必要に遅らせない。
5. 発話優先窓は最大60秒まで処理待ちを許容する。保存数は2件で固定し、
   無制限のbacklogは作らない。

性能値は実機での新入力確認前であり、改善後の遅延値をまだ断定しない。
PC上のモデル推論・保存動画評価は実施していない。

## 2026-10-28 改善後の本人実入力結果

session `demo_continuous_20261028_140206_v25_realtime_live`で、開始前とready後を
別々に本人確認し、arm後に4秒窓を取得した。自動録り直し・再armは行っていない。

- 発話候補：`live_000003`、RMS 0.030797（周辺窓は約0.008～0.009）
- ASR：`今日はいい天気ですね`。本人が発話本文との一致を確認
- 窓末尾→ASR：14.066秒
- 窓末尾→Visual：19.533秒
- 窓末尾→ARA候補：19.409秒
- 窓処理：14.067秒（前回35.921秒から約21.85秒短縮）
- Visual：顔40/40、geometry 28/40、neutral 25／angry 10／sad 5、表示neutral 62.5%
- ARA：4秒budgetのfallback `必要な案内だけ表示します。`
- 本人がHDMIのARA文と基板音声案内を確認し、問題なしと回答
- 終了：errorなし、cleanup正常、全process／worker終了、device holder 0
- 151 files／20,053,993 bytesを全hash一致で回収し、3コピーoffload

入力待ちキューは高RMS窓を低RMSの新しい窓より優先し、発話窓を処理できた。
ただし、処理後の保存窓を名前順の最新3件だけにする旧パージが、停止前に
`live_000003`の詳細artifactを削除した。このため、処理済み保存窓も
「最大RMS 1件＋最新2件」を保持する方針へ修正した。保持判断と削除対象は
`processed_window_retention.json`および`processed_window_purged`イベントへ記録する。

今回値は前回より大幅に短縮したが、窓末尾からASRまで約14秒、表示まで約19秒であり、
厳密なリアルタイム達成とは扱わない。PC評価・保存動画再評価は実施していない。

## 2026-10-28 長時間安定性を優先する容量改善

ユーザー方針により、追加のリアルタイム化より8時間以上の安定動作を先に完了する。
今回sessionの実行中空きは223,731,712 bytesから201,166,848 bytesまで低下した。
回収証拠20,053,993 bytesの主な内訳は次のとおり。

- 未使用`framebuffer_before.bin`：8,294,400 bytes
- 保持3窓：4,756,751 bytes
- preparation：3,810,364 bytes
- worker pool：559,867 bytes
- `memory_samples.jsonl`：223秒・42標本で108,457 bytes

`framebuffer_before.bin`は書込み以外の参照がなく、画面復帰はsupervisorのWeston leaseで
実施されるため保存を廃止した。これにより毎session約8.3MBを削減する。

資源安全監視は従来どおり5秒ごとに実行し、headroom違反時の停止応答を維持する。
一方、詳細なRAM／Swap process標本のディスク書込みは60秒ごととし、8時間換算で
約15MBから約1.3MBへ抑える。監視周期と証拠書込み周期を分離し、安全性は下げない。

変更後の録音なし実機check
`demo_continuous_20261028_141646_v25_stability_check`で次を確認した。

- 起動要求からHDMI readyまで96.072秒、`ready_reached=true`、`error=null`
- Visual／SenseVoice／Emotion2Vecはすべて`ok`
- 新録音0、全process／worker終了、Weston復帰、device holder 0
- 約98秒で`memory_samples.jsonl`は2標本、`framebuffer_before.bin`は非生成
- 72 files・7,784,730 bytesをremote前後／PCで全hash一致
- 3コピー後にoffloadし、実機空き223,690,752 bytesへ復元

短時間checkは省容量化と終了処理の確認であり、8時間安定動作の達成とはしない。
本人の新しい準備確認を経て8時間運転を完了し、その後に追加のリアルタイム化へ進む。

## 2026-10-28 長時間連続稼働向け ALSA XRUN 自動復帰（recovery）の追加

8時間耐久試験セッション `demo_continuous_20261028_142628_v25_soak8h` において、約31分間（1,868.99秒稼働後、カメラ18,467フレーム、音声2,980万サンプル）安定稼働後、マイク入力で ALSA EPIPE（`-32`）が発生し安全停止した。停止直前の本人発話（「今日は寒いですね」）、HDMI短文表示、基板音声案内はすべて本人確認済み。

短時間評価向けに実装されていた `idle_recovery_allowed`（録音待機中のみ復帰を許可し、録音要求中は停止する）が、常時armed状態となる長時間連続稼働（continuous soak）でも作動し、偶発的な1回のXRUNに対して復帰を試みず即座に例外停止していた。

対策として以下を実装・配備：
- `NativeMicrophone` に `allow_continuous_recovery` オプション（既定: True、最大上限: 100回）を追加。
- 連続稼働モードにおいて ALSA EPIPE (-32) が発生した場合、即時停止せず `snd_pcm_prepare` による安全なPCMリングバッファリセット・新エポック記録（`audio_xrun_recovery` イベント）を行い、キャプチャを自動継続。
- 重複窓は `input_gap`（`reason: audio_window_not_available`）として安全にスキップされ、次窓から新PCMで推論処理が再開される。


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

初回開始では、ready直後の旧自動動作により本人が話していない1窓が作成された。
この窓は本人発話成功には使用せず、セッションを直ちに安全停止し、95ファイル・
22,704,725 bytesを全hash一致で回収・3コピーoffloadした。修正後の
`demo_continuous_20261026_233000_v25_gatecheck`では、ready後も
`recorded_windows=0`、`post_ready_capture_armed=false`で待機することを実機確認済み。

## 未完了の正式受入

- 本人の現在準備確認後の実機カメラ・マイク入力
- Visual stride 4での顔・表情・頭部・手認識品質確認
- ASRおよび総応答遅延の再計測
- ディスク／RAM／process／デバイス保持を監視した8時間以上の連続運転

上記が完了するまで「リアルタイム達成」「8時間連続運転達成」とは記録しない。
