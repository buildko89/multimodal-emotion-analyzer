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

## 2026-10-30 8時間耐久合格後の応答遅延改善（V25のまま）

### 改善前の実機実測（PC推論なし・実機ログの集計のみ）

耐久session `demo_continuous_20261028_170500_v25_soak8h`（14時間09分、178 files／
54,791,395 bytesを全hash一致で回収・3コピーoffload）の`live_events.jsonl`を集計した。

- 閉じた窓12,591件のうち処理4,250件（約66%が処理待ちで破棄）
- ARA応答4,249件すべて`fallback`（生成採用0件）
- 窓確定→ARA表示：p50 18.9秒、p90 26.4秒
- ARA監査：prefillだけでTTFT約3.7秒、完全生成は約10.1秒（約33 token）。
  4秒budgetでは原理的に生成が完了せず、毎窓4秒を待ってからfallbackしていた。
- `demo_continuous_20261028_140206_v25_realtime_live`の窓内訳：SenseVoice 6.5～7.5秒、
  Visual worker 4.5～8.7秒、窓処理全体約11.4秒（うちARA待ち約4秒）。
  無音窓でも同じ時間を使うため、発話窓が直前の無音窓の後ろで約5秒待った。
- Visual表示用`compact_observations.json`がARA呼び出し後に書かれていたため、
  画像認識結果の表示もARA待ちの後になっていた。

### 変更（3点、モデル・窓幅・キュー上限・保持数は不変）

1. **無音窓のSenseVoice省略（`live.asr_energy_gate`）**：窓PCMのRMS < 0.012
   **かつ** peak < 0.03 の場合だけSenseVoiceを呼ばない。どちらかを超えれば従来どおり実行する
   （騒がしい環境では省略されない側に倒れる）。判断値は各窓の`asr_energy_gate.json`と
   `sensevoice_client_audit.json`（`status: skipped`）に記録し、ASR観測は`unavailable`とする。
   起動時準備は対象外で、従来どおり全backendの`ok`を要求する。
   耐久sessionの実測値では、処理窓4,250件中2,941件（約74%）が省略対象になり、
   省略対象窓の非空ASRは`i`等の無音時の誤出力だけだった（意味のある日本語の欠落0件）。
   閾値0.08案は小声の「こんにちは」等を落とすため不採用。
2. **ARA応答の非ブロッキング化（`cooperative_ai.background_generation_seconds: 30`）**：
   窓処理はARAを待たない。同一要求（messages＋schemaのSHA-256が完全一致、temperature 0・seed 42）
   のARA生成済み文があれば即時採用し（`response_source: ara_exact_input_cache`）、
   無ければ検証済みfallbackを即時表示して、裏で1件だけ30秒budgetで生成する。
   常駐runtime内で最大32件を保持し、生成中の重複要求は出さない（`busy`）。
   現行promptは表情区分と案内文だけで構成されるため、数種類の表情が出現した後はARA生成文が
   表示される。ASR本文は現行どおりpromptに含めていない。
3. **画像認識結果をARAより先に表示**：`compact_observations.json`をARA呼び出し前に書く。

### 期待値（未検証）

窓処理は無音窓で約11.4秒→約5秒、発話窓で約11.4秒→約8秒台を見込む。待ち行列の滞留が減るため、
発話窓末尾→ASR／画像表示は約14～19秒から約8～10秒への短縮を見込む。
**これは実機ログからの見積もりであり、本人の実機入力で再計測するまで達成とは扱わない。**

### 配備

- V25の番号・配備先を変えず`build --from-release ara_demo_20261026_v24`→`deploy`。
  前V25からの変更は`config.json`と3 module（`observation_integration.py`、
  `demo_response_transport.py`、`live_pipeline.py`）だけ。
- deploy後doctor：`ok=true`、`bad_hashes=[]`、ARA `ready`、device holder 0、
  空き216,248,320 bytes。
- 配備前に、未offloadだった`demo_continuous_20261028_142628_v25_soak8h`（164 files）も
  3コピー検証後にoffloadし、空き容量を回復した。常駐ARAの`/root/devai_ara/services/ara_llm/ara_audit.jsonl`
  （約7.5MB）が要求ごとに増える点は残件。今回の変更でARA呼出し数は大きく減る。

### 2026-10-30 22:30 本人実入力での再計測（`demo_continuous_20261030_222417_v25_perf_live`）

開始前「準備できてます。」、ready後「日本語でお願いします。準備はできてます。」を別々に受けてarm。
168 files／25,460,647 bytesを全hash一致で回収し、3コピーoffload。

- 発話窓：`live_000004`「今日はいい天ですね」、`live_000008`「今日はいい天気ですね」（RMS 0.045／0.048）。
  本人申告「字幕がでるまではちょっと遅い感じ」。
- 窓末尾→ASR：`live_000008` 11.9秒、`live_000004` 18.6秒（処理待ち10.4秒を含む）。
  窓末尾→画像表示13.7／20.3秒。
- **ARA：初回生成後、ARA生成文を即時採用**（`ara_exact_input_cache`）。本人がHDMIで
  「今暫く休憩して、その後は更なる旅をサポートしますね。」を確認。改善前は4,249件中採用0件。
- 窓処理（ARA待ちなし）：8.1～11.3秒。SenseVoice 6.4～6.8秒、Visual 5.8～7.8秒。
- **無音窓のASR省略は1回も作動しなかった**：この日の環境の無音窓はRMS 0.0163以上、peak 0.063以上で、
  耐久試験時（RMS約0.0084、peak約0.021）の約2倍。固定閾値（0.012／0.03）を下回る窓がなく、
  閉窓33件中、処理12件で、処理待ち5～10秒が残った。
- 停止要求後に処理中だった`live_000032`は`perception_worker_stopped_or_deadline`で中断（停止時の正常系）。

### 2026-10-30 22:35 騒音床に追従する閾値へ変更

固定閾値は環境が変わると作動しないため、各窓より**前に**閉じた直近30窓の20パーセンタイルを騒音床とし、
RMS閾値＝max(0.012, 床×1.2)（上限0.022）、peak閾値＝max(0.03, 床×1.5)（上限0.1）とした。
直近10窓未満の起動直後は固定閾値。各窓の`asr_energy_gate.json`に騒音床と採用閾値を記録する。

ログだけを使った再集計（モデル推論なし）：
- 今回session：閉窓の55%が省略対象、発話2窓はどちらも処理対象のまま（欠落0）。
- 耐久session：閉窓の83%が省略対象、意味のない断片2件（「干ま」「我」）以外の欠落なし。

再配備（V25、doctor ok、bad_hashes 0、ARA ready、空き216,195,072 bytes）と次session
`demo_continuous_20261030_223809_v25_perf_live2`のpreflightまで合格。効果は本人実入力での再計測後に判断する。

### 2026-10-30 22:50 騒音床追従版の本人実入力再計測（`demo_continuous_20261030_223809_v25_perf_live2`）

開始前「計測できます。」、ready後「はい。話せます。」を別々に受けてarm。168 files／24,962,538 bytesを
全hash一致で回収し、3コピーoffload。停止要求後の`live_000021`中断は停止時の正常系。

- 閉窓21件、処理8件。発話窓：`live_000006`「今日はいい天気で」、`live_000010`「今日はいい天気です」、
  `live_000013`・`live_000016`「今日はいい天気ですね」（`live_000007`は語尾「すね」）。
- 窓末尾→ASR：発話窓で8.3／11.9／9.6／15.0秒（前回11.9／18.6秒）。窓末尾→ARA表示10.0～16.7秒。
- ARA：同一表情区分はARA生成文「今日のルートは順調そうですね。…」を即時採用、初出の区分はfallback。
- 騒音床追従は保持窓で`basis: recent_noise_floor`を確認（発話窓は`skipped: false`）。無音窓`live_000015`は
  ASR確定5.7秒と短く省略された可能性が高いが、窓が保持上限で削除済みのため判定ファイルでは未確認。
- 起動直後の10窓は騒音床の学習中で固定閾値になり、本人の無言40秒の間は省略されなかった。
- 窓処理は1件8.4～9.4秒（SenseVoice約6.5秒とVisual約6～7秒が並行）で4秒窓より長く、
  発話が続くと次の発話窓が4～7秒待つ。**字幕まで約8～15秒で、リアルタイム達成とは扱わない。**

残る主因は窓1件の処理時間そのもの。次の候補：ゲート判定を`live_events.jsonl`へ記録（削除窓でも検証可能に）、
騒音床の学習を録音開始前のプレビュー音声でも行う、SenseVoice／Visualのthread配分とVisual間引きの見直し。

### 2026-10-30 23:05 (a)(b)(c)版の本人実入力再計測（`demo_continuous_20261030_225306_v25_perf_live3`）

開始前「はい。対応できます。」、ready後「話せます」を別々に受けてarm。168 files／25,018,923 bytesを
全hash一致で回収し、3コピーoffload。本人申告「字幕が出るまではちょっと早くなった気がする。顔の表示も少し早くなった。
ARA文は『必要な案内だけ表示します。』などいろいろ出る」。

- (b) arm前学習：13標本（音声保存なし）、arm時点で騒音床RMS 0.0082／peak 0.022、閾値RMS 0.012／peak 0.033。
  この日の騒音は前2回より低かったが、無音窓にも一時的な物音（peak 0.05～0.29）が多く、省略は処理15窓中2窓のみ。
- (a) 全15窓で`window_perception_summary`を記録できた（削除済み窓でも判定・時間を確認可能）。
- 発話窓：「今日はいい天気ですね」、「こんにちは」×2。窓末尾→ASRは10.8／15.6／20.3秒（処理待ち3.2～12.3秒を含む）。
- SenseVoice 6.4～7.7秒（前回6.9～7.1秒）で変化なし。Visual 4.7～9.4秒（前回5.8～6.0秒）で遅くなった。
  ASR省略窓ではVisualが律速となり表示が遅れた。**nice 5はASRを速くせずVisualを遅くしたため取り消した。**
  stride 5は維持する。顔検出は全窓40/40。geometry 0の窓が6件あったが、その窓は削除済みで原因は未確認
  （保持窓2件はすべて`ok`）。
- 客観値では2回目（8.3～15.0秒）から明確な短縮は確認できない。本人の体感は改善。リアルタイム達成とは扱わない。

結論：窓1件の処理（SenseVoice約6.5秒が下限）が4秒窓より長い限り、連続発話で待ちが生じる。
次の構造的な候補は、窓をまたいだASRとVisualの並行処理、または発話窓のみASRを実行しVisualを軽量化して窓処理を4秒以下にすること。

### 2026-10-30 23:30 (e)持続音ゲート＋(d)窓をまたいだ並行処理（本人承認「e → dで良いです」）

(e) 無音窓判定を「音の持続時間」に変更した。20ms frameのRMSが max(0.012, 窓内20パーセンタイル×2) を超える
時間の合計が0.25秒未満なら、SenseVoiceを呼ばない（`policy: skip_sensevoice_without_sustained_sound`、
`active_seconds`を窓ごとに記録）。回収済み保持窓の音声で信号処理のみ検算：発話窓0.42～1.64秒、
無音・物音窓0～0.10秒（peak 0.18の物音を含む）。基板の案内音声を拾った窓（ASR「必要な案内だけを表示します」）も
0.06秒で省略対象。

(d) `live.cross_window_overlap: true`。常駐runtimeは窓Nのフレームループ（約2秒）後、残り（worker応答待ち・
統合・ARA）を1本の背景threadで実行し、その間に窓N+1のフレームループを開始する。背景threadは同時に1本までで、
窓N+1の残り処理の前に必ずjoinする。consumerは窓Nの`frames_done.json`を見て次の1窓だけ先送りする
（`processing_started.sent_ahead: true`）。同時に処理中の窓は最大2、各backendのworker要求は実行1＋待ち1に収まる。
設定をfalseにすれば従来の逐次処理に戻る。

配備：doctor ok、bad_hashes 0、ARA ready、空き216,031,232 bytes、変更ファイルhash一致、実機Python 3.12でcompile確認。
**効果・安定性は未計測**（本人実入力で確認する）。

### 2026-10-30 23:25 (e)(d)版の本人実入力再計測（`demo_continuous_20261030_231357_v25_perf_live4`）

開始前「はい。準備できてます。」、ready後「はい。話せます」を別々に受けてarm。149 files／20,487,427 bytesを
全hash一致で回収し、3コピーoffload。本人申告「顔の表示は良い。音声認識は表示が遅くなっている気がする」。

- (e) 持続音ゲート：発話9窓はすべてASR実行（active 0.44～1.24秒）、無音20窓はすべて省略（0～0.22秒）。誤判定0。
- (d) 並行処理：閉窓46件中処理29件（63%、前回38%）。window_failureは停止要求後の1件のみ。表示順の乱れの申告なし。
- ただしSenseVoiceが6.3～9.6秒（前回6.4～7.7秒）に遅化。ASR実行中に次窓のフレームループ・Visualが同時に走りCPUを奪った。
  発話窓末尾→ASRは10.3～17.7秒で、2回目（8.3～15.0秒）より悪化。本人の体感と一致。

対策（同日配備）：`overlap_only_when_asr_skipped: true`。現在の窓がASRを省略した（無音）場合だけ次窓を先送りし、
発話窓のASRは単独で走らせる。commit後に再配備（doctor ok、hash一致、実機compile ok、空き215,977,984 bytes）。効果は未計測。

### 2026-10-30 23:35 5回目（`demo_continuous_20261030_232338_v25_perf_live5`）で発生した不具合と修正

開始前「準備できてます。」、ready後「話せます。」を別々に受けてarm。181 files／26,547,125 bytesを全hash一致で回収し、
3コピーoffload。本人がHDMIで「Error2 No such file or directory」表示を確認。

- 原因：発話優先キューは音量の大きい窓から渡すため、古い`live_000009`が新しい`live_000010`・`live_000011`の後に
  取り出された。先送りでencodeした直後、保持パージ（最大RMS＋最新2件）が`live_000009`を古い窓として削除し、
  直後の`processing_config.json`書込みが`FileNotFoundError`になった。window_failureにより以後の認識が停止した。
  この削除条件は逐次処理でも潜在していたが、先送りで顕在化した。
- 修正：`result.json`のない（処理中の）窓はパージ対象にしない（方針名どおり「完了窓」だけを削除）。
  一時ディレクトリでパージ関数単体を確認し、処理中の2窓を保持して完了済みの古い窓だけを削除した。
  保持数は完了3窓＋処理中最大2窓で有界。
- 今回sessionは失敗が開始約20秒で起きたため、性能値の比較には使わない。
- 再配備：doctor ok、bad_hashes 0、ARA ready、空き215,924,736 bytes、hash一致、実機compile ok。

### 2026-10-31 Visual workerの常駐化とSenseVoiceの発話区間入力（本人承認「案2,4はOKです。手の認識は間引かないでください。」）

根拠（回収済み`..._231357_v25_perf_live4`のログ読解のみ。新しい録音・推論なし）：
- フレームループ（Neutron）は約1.8～2.5秒で終わるが、Visual worker（MediaPipe Face＋Hand Landmarker、CPU）は
  4.3～11.2秒で29窓中28窓が4秒超。無音窓でもfusion完了→envelope完成に最大8秒待ち、次窓の開始が2～6.5秒遅れた。
- Visual workerは常駐processなのに、窓ごとにHand modelのSHA256、Face bundleのSHA256、両graphの生成を行っていた。
- 発話窓の持続音は0.44～1.24秒／4秒窓。SenseVoiceは毎回4秒全体を処理していた。

変更：
- 案2：Visual workerはprocess内で1回だけ両modelを検証・生成し、以後の窓で再利用する。Face LandmarkerのVIDEO graphは
  窓ごとに時刻0から始まるため、graph内部の時刻だけ前窓末尾＋1秒のoffsetを加える（記録する`timestamp_ms`は窓内時刻のまま、
  `graph_timestamp_ms`を併記）。窓の処理が失敗した場合はgraphを破棄し次窓で作り直す。手の認識は従来どおり全推論frameで実行。
  `visual_summary.json`に`worker_models`（reused、setup_seconds、calls）を記録。
- 案4：`asr_energy_gate.activity.crop_margin_seconds: 0.3`。持続音ゲートで閾値を超えた最初～最後のframeに前後0.3秒を加えた
  区間だけをSenseVoiceへ渡す（emotion2vec等は窓全体のまま）。区間は`asr_energy_gate.json`／`window_perception_summary`の
  `asr_sample_interval`に記録。設定を外せば従来の4秒全体入力に戻る。
- PC上では切り出し区間の算術（合成信号1.0～2.0秒→0.7～2.3秒、hash一致）だけを確認した。効果・認識への影響は実機で未計測。

### 2026-10-31 7回目（`demo_continuous_20261031_082609_v25_perf_live7`）：案2・案4の実測

開始前「実測はできます。」、ready後「はい。話せます」を別々に受けてarm。録音は約59秒（閉窓14件）。162 files／20,160,942 bytesを
全hash一致で回収し、3コピーoffload（停止時にwindow runtimeがSIGKILLされ`workers_exited`が未記録になったため、
offloadに`--allow-unrecorded-worker-exit`を追加。実機のprocess走査で`observation_worker.py`の残存なしを確認する条件は維持）。

- 処理窓／閉窓：12/14（86%、4回目67%）。次窓の開始待ち：全窓0秒（4回目2～6.5秒）。
- Visual worker：3.4～5.4秒（4回目4.3～11.2秒）。`worker_models.reused: true`、準備0秒。
- 窓確定→画像表示：7.0～12.8秒（4回目10.0～23.5秒）。窓確定→字幕：8.9／7.5秒（4回目10.1～17.5秒）。
- SenseVoice：5.7秒（入力1.96秒）／4.9秒（入力1.18秒）。4回目6.3～9.6秒（入力4秒）。入力長に比べて短縮幅が小さく、固定費が大きい。
- 認識：「今日はいい天気ですね」「こんにちは」。
- **退行**：保持窓live_000005で、フレームループの顔検出（MediaPipe BlazeFace、CPU）は40フレームすべてで顔1件を検出したのに、MediaPipe Face Landmarkerは
  前半20フレーム（約2秒）が`face_unavailable`だった。3回目・4回目の保持窓は全フレーム`ok`。窓をまたいでVIDEO graphを
  再利用した（時刻offset付き）ことが原因と判断した。最終表示でも「画像結果は不明」が出た。
- 停止時：live_000013の画像処理が途中（14/40フレーム）で停止要求を受けた。window_failureは停止時の1件だけ。

修正：Face LandmarkerのVIDEO graphは窓ごとに作り直す（`reset_graph`）。modelのhash検証は1回、Hand Landmarker（IMAGE mode）は
引き続き再利用。graph作成時間は`worker_models.setup_seconds`に記録。方式A（SenseVoiceの動的INT8）と一緒に配備し、次回計測する。

### 2026-10-31 8回目（`demo_continuous_20261031_084220_v25_perf_live8`）：顔graph修正＋方式A（動的INT8）

開始前「計測開始しましょう。」、ready後「はい話せます。」を別々に受けてarm。録音275秒、閉窓68件・処理53件（78%）。
209 files／27,311,189 bytesを全hash一致で回収し、3コピーoffload（今回は`workers_exited: true`で通常どおり）。
window_failureは停止時の1件だけ。本人申告「音声認識は、何か反応が悪いですが、これが限界な気がしました。」

- INT8：準備・live窓とも`quantization: dynamic_int8_linear`（qnnpack、変換3.8秒）。準備用音声は「今日はいい天気ですね」と正しく認識された。
- SenseVoice：2.4～4.7秒（入力0.6～3.2秒）。FP32の切り出し版（7回目）は4.9／5.7秒、4秒入力（4回目）は6.3～9.6秒。
- 窓確定→字幕：5.1～9.9秒（7回目7.5／8.9秒、4回目10.1～17.5秒）。
- Visual worker：3.2～5.5秒。顔graphの作り直しは0.27～0.51秒。窓確定→画像表示：多くは6.4～17秒、外れ値1件42.6秒。
- 顔：保持3窓のうち2窓は全フレーム`ok`。live_000065は先頭1秒以降が`face_unavailable`だった（フレームループの顔検出（MediaPipe BlazeFace、CPU）は全フレームで顔1件）。
  graphは作り直しているので、7回目の欠けをgraphの再利用だけで説明することはできない。姿勢や手の重なりなどの影響を含めて未解決。
- 認識の誤りは窓の境目に集中した。持続音の区間が窓末尾（4.0秒）や窓先頭（0.0秒）に接する窓で、発話が分断されていた。
  - 「今日は寒いです」（2.76～4.03秒）→次窓「すね」（0.0～0.6秒）
  - 「二比人」（3.36～4.03秒）→次窓「二識がおかしいぞ」（0.0～3.18秒）＝「認識がおかしいぞ」が分断された
  - 短い断片（0.3～0.5秒）では`language="auto"`が中国語の「嗯」を2回出した
  - 境目に接していない窓は「こんにちは」「今日はいい天気ですね」「音声認識が遅いね」「どうでしょう」など正しく認識した
- 結論：速度はINT8で約3～4割短縮した。誤認識の主因はモデルの精度ではなく、固定4秒窓による発話の分断と、言語の自動判定。

### 2026-10-31 ①言語のja固定と②発話に合わせた窓の延長（本人承認「①②両方で進めましょう。」）

- ①`observation_integration.audio_language_by_backend.sensevoice: "ja"`。SenseVoiceの`language`を`auto`から`ja`へ固定する（短い断片が中国語の「嗯」になるのを防ぐ）。cacheされたadapterと設定が違えば拒否する。設定を消せば`auto`へ戻る。
- ②`live.window_extension`（`max_extra_seconds: 3.0`、`tail_seconds: 0.3`、`min_tail_active_frames: 3`）。4秒の時点で、直近0.3秒に持続音（ASRゲートと同じ閾値：max(0.012, 窓内p20×2)を超える20ms frameが3個以上）があれば窓を閉じずに延ばす。無音になった時点、または最長7秒で閉じる。窓は連続したままで、窓どうしの依存や字幕の重複は生じない。
  延長した秒数は`window_closed.boundary_extension_seconds`と`capture_manifest.json`に記録する。設定を消せば固定4秒窓へ戻る。
- 代償：境目をまたぐ発話だけ、窓が最大3秒長くなり、その分の画像処理と字幕が遅れる。PCでは文法の確認（compile）だけを行った。

### 2026-10-31 9回目（`demo_continuous_20261031_090604_v25_perf_live9`）：①ja固定＋②窓延長

開始前「計測始めましょう」、ready後「開始しましょう」を別々に受けてarm。録音450秒、閉窓110件・処理88件（80%）。155 filesを全hash一致で回収し、3コピーoffload（停止は正常）。
window_failureは停止時の1件だけ。

- ①中国語の出力：0件（8回目は「嗯」2件）。本人の「これ中国語だ」も日本語で認識した。
- ②実際に延長した窓：7件（0.23～1.74秒）。他の窓の0.028／0.032秒は、カメラのフレーム間隔による端数で延長ではない。
  - 1つの字幕にまとまった例：「こんにちは今日は寒いですね」（延長1.04秒）、「ちょっと認識が遅いですね」（1.04秒）、「音声認識遅いですね」（1.74秒）。
  - まだ分断した例：「今日はいい天気で」（延長0.43秒）→次窓「すねちょっと寒いです」。「て」（0.23秒延長）の断片。延長中に言葉の間の短い無音（約0.25秒以上）で窓を閉じていた。
  - 保持窓live_000100の検算では、フレーズ内の無音は0.1秒以下、フレーズ間は1.1秒以上だった。
- 速度：SenseVoice 2.9～6.6秒（延長した長い窓ほど遅い）。窓確定→字幕 4.8～15.1秒。画像表示 中央値12.0秒（6.9～36.1秒）。Visual worker 2.4～7.0秒。
- 顔：保持窓は全フレーム`ok`。
- 誤認識の残り：「眠てち」「もうちょっと早いでいいですね」など（音響的な誤りで、境目とは関係ない）。

調整：`window_extension.tail_seconds` 0.3→0.6（直近0.6秒に持続音が3 frame以上あれば延長を続ける＝約0.55秒の無音で閉じる）。

### 2026-10-31 10回目（`demo_continuous_20261031_091946_v25_perf_live10`）：延長の閉じ判定0.6秒

開始前「Aで計測しましょう。」、ready後「OKです」を別々に受けてarm。録音270秒、閉窓65件・処理46件（71%、低音量窓18件は音量優先で省略）。
185 filesを全hash一致で回収し、3コピーoffload（停止は正常）。window_failureは停止時の1件だけ。

- 延長した窓：10件（0.23～3.05秒、最長7.05秒の窓が1件）。
- 「…で｜すね」型の文中での分断：0件（9回目は2件）。「大丈夫疲れていないよ」「ルートは順調ですか」「でも眉（前）より良くなったな確かにね」「今日は寒いですね」が1つの字幕にまとまった。
- 残った誤り：
  - 窓の終わり際に話し始めた発話の語頭が欠ける：「性を変えないとたダメだ」、断片「い」「まい」。4秒の時点で、まだ持続音が3 frameに達していなかった。
  - 音響的な誤り：「眉より」（前より）、「緊張ばしてないね」、「疲労困憊」など。
  - 中国語の出力：0件。
- 速度：SenseVoice 2.6～7.1秒（7秒窓で7.1秒）。窓確定→字幕 中央値9.0秒（5.0～17.8秒）。画像表示 中央値12.2秒（5.6～20.7秒）。
- 顔：保持窓live_000063は、フレームループの顔検出（MediaPipe BlazeFace、CPU）が全フレームで顔を検出したのに、MediaPipeは0/40だった。graphは窓ごとに作り直しているので、graphの再利用とは別の原因（未解決）。

### 2026-10-31 記述の訂正

- 7～10回目の記録で「NPUは全フレームで顔を検出した」と書いた箇所は誤り。比較に使った`frame_observations.jsonl`の`face_count`は、
  フレームループの顔検出（MediaPipe FaceDetection＝BlazeFace short range、**CPU**）の結果。NPUで動いているのは表情認識（FER）だけ。
  上の該当箇所は訂正した。`face_landmarker_adapter.py`の`reset_graph`のコメントにも同じ誤りが残っているが、固定した配備版（`ab3cb3b`）と
  ソースを一致させるため、コードは変更していない（次にコードを変更するときに直す）。
- 構造・ボトルネック・操作手順は `ara_spec/`（01～03）にまとめた。

### 2026-10-31 ①Emotion2Vecの停止 ②顔検出のNPU化（まず並走で比較） ③encode時間の計測（本人指示「Emotion2Vecの停止 → 顔検出のNPU化 → encode時間の計測 → 顔ランドマーク・手のNPU化までやってみましょう。」）

- ①`live.release_unused_audio_workers: true`：起動時の準備（Emotion2Vecを含む全backendの健全性チェック）が終わった後、live窓で使わない音声workerを閉じる。結果は`live/released_workers.json`と`window_runtime_final.json`に記録する。
- ②`face_detection.npu_shadow: {min_score: 0.4}`：リポジトリにあった未使用のNPU用BlazeFace（`models/blazeface_neutron.tflite`、10/24変換）を、
  主の顔検出（MediaPipe BlazeFace、CPU）と**並走**させ、フレームごとに結果と時間を記録する（表示・後段は主の検出のまま）。
  - 切り替えずに並走させた理由：実機でモデルの量子化パラメータを読んだところ、スコア出力が`scale 0.6609, zero_point 127`で、表せる範囲が −168.5～0 だった。つまりsigmoidの最大が0.5に張り付く。
    入力も0～1の範囲で量子化されている（MediaPipeの元モデルは−1～1を想定）。変換時の校正が不適切だった可能性が高く、検証なしには使えない。
  - 新規の`models/blazeface_decode.py`：896アンカーの生成、letterbox、box復号、NMS（IoU 0.3）。PCでは合成テンソルで算術だけを確認した（中心・大きさの復元と重複除去）。
  - 窓ごとの比較を`window_perception_summary.face_detection`（主・NPUの顔あり数、IoU≥0.5で一致したフレーム数、最大スコア、平均時間）に記録する。
- ③`processing_started.encode_seconds`：窓の映像・音声のencode（libx264＋時刻の検証）にかかった時間。
- `face_landmarker_adapter.py`の`reset_graph`のコメントの誤り（「NPU detector」）を訂正した。

### 2026-10-31 11回目（`demo_continuous_20261031_095456_v25_perf_live11`）：①～③の実測

開始前「計測始めましょう」、ready後「はい、話せます」を別々に受けてarm。録音201秒、閉窓46件・処理33件。172 filesを全hash一致で回収し、3コピーoffload（停止は正常）。

- ①Emotion2Vecの停止：準備の後に閉じた（`released_workers.json`：exited true）。live後半のMemAvailableの最小値は3.48→**4.09 GiB**、RSS合計の最大は4.31→**3.60 GiB**（10回目比）。Swap使用は222 MiBで変わらず（一度swapへ出たページが残るため）。
- ②NPU BlazeFaceの並走：1,342フレーム中、主の検出（CPU）と**99.5%一致**（顔の数が同じで、IoU≥0.5）。エラー0件。不一致が1割を超えた窓は32窓中1窓だけだった。
  1フレームあたりの時間は、NPUが**11.4 ms**（letterboxとUDSを含む）、CPUのMediaPipeが32.8 ms。スコアは予想どおり全フレームで0.5に張り付いたが、顔の有無と位置は正しく出た。
  並走したぶん、フレームループは中央値1.83→2.43秒に遅くなった。
- ③encode：1窓あたり**中央値0.65秒**（0.56～0.92秒）。VPUへ移しても縮むのは最大で約0.65秒なので、優先度は低い。
- 字幕：窓確定から中央値10.6秒（12件）。画像表示は中央値12.0秒。並走の負荷と標本の少なさのため、10回目（9.0／12.2秒）との差は判断できない。

対応：`face_detection.npu_blazeface: {min_score: 0.4, mode: "primary"}`。NPUの結果を主の顔検出として使う。NPUがエラーのときは、設定された検出（MediaPipe、CPU）に戻る。

### 2026-10-31 12回目（`demo_continuous_20261031_101230_v25_perf_live12`）：NPU顔検出を主に／ここで固める

開始前「1だけをやりましょう。」、ready後「はい、話せます」を別々に受けてarm。録音134秒、閉窓31件・処理21件（68%）。155 filesを全hash一致で回収し、3コピーoffload（停止は正常）。

| 項目 | 10回目 | 11回目（並走） | 12回目（NPUが主） |
|---|---|---|---|
| フレームループ（中央値） | 1.83秒 | 2.43秒 | **0.64秒** |
| 顔検出（1フレーム平均） | ― | CPU 32.8 ms | **NPU 8.8 ms** |
| 顔ありフレーム | ― | 1,334/1,342 | 814/815 |
| Visual worker（中央値） | 4.2秒 | 4.2秒 | 5.2秒 |
| SenseVoice（中央値） | 3.8秒 | 3.9秒 | 3.5秒 |
| 窓確定→字幕（中央値） | 9.0秒（16件） | 10.6秒（12件） | 10.0秒（6件） |
| 窓確定→画像表示（中央値） | 12.2秒 | 12.0秒 | 12.8秒 |
| MemAvailable最小／RSS合計最大 | 3.48／4.31 GiB | 4.09／3.60 GiB | 4.17／3.56 GiB |

- 保持窓95フレームはすべて`neutron_blazeface`で検出し、`npu_error`は0件。保持2窓のFace Landmarkerは全フレーム`ok`。
- 結論：①（Emotion2Vecの停止）と②（NPU顔検出）で、**メモリ（+0.7 GiB）とフレームループのCPU（約65%減）には効いた**。しかし、**字幕と画像表示の遅れは変わらなかった**。律速は、フレームループの後ろで動くVisual worker（MediaPipe Face/Hand、CPU）とSenseVoiceにあり、フレームループは元々それらと並行していたため。③encodeは0.65秒で小さい。④（Face/HandのNPU化）は、PCでの校正が現環境では難しいため、本人判断で見送った。
- 観察：字幕に「必要な案内だけ表示します」「何か必要な情報があれば」が出た。基板スピーカーのARA音声案内をマイクが拾った可能性が高い（スピーカーからマイクへの回り込み。未対策）。
- 本人の方針（2026-10-31）に従い、**この版（commit `6723061`）を12月版候補として固める**。次はYocto Linuxへの組み込みレシピを作る。
