# 実機ハードウェア・システム環境設定仕様書

文書番号: DEVAI-DEMO-ENV-20261026  
対象環境: NXP i.MX95 EVK + Kinara ARA-2 + Linux Yocto BSP  
位置づけ: 実機のOS設定、ハードウェアリソース、ネットワーク、サービス構成の正本  

---

## 1. ハードウェア構成

| コンポーネント | 仕様 / 型番 | 接続インターフェース | 役割 |
|---|---|---|---|
| メインボード | NXP i.MX95 Evaluation Kit (EVK) | オンボード | システム全体制御、NPU推論、メディアキャプチャ |
| SoC | i.MX95 (6x Arm Cortex-A55, Neutron NPU) | - | OS実行、映像処理、高速FER推論 |
| AIアクセラレータ | Kinara ARA-2 M.2 Module | PCIe Gen3 x1 | Qwen2.5-7B LLM 推論専用アクセラレータ |
| USBカメラ | UVC準拠 Full HD Webカメラ | USB 3.0 Type-A | ドライバー顔・表情常時撮影 (10fps BGR) |
| USBマイク | ALSA準拠 USBオーディオマイク | USB 2.0/3.0 Type-A | ドライバー発話・音声常時録音 (16kHz S16_LE) |
| ディスプレイ | フルHD (1920x1080) HDMIモニター | HDMI Type-A 出力 | 来場者向けリアルタイムHUD描画 |
| 音声出力 | MQS オーディオ または USBスピーカー | 3.5mm ジャック / USB | AI相棒音声アナウンス再生 |

---

## 2. ネットワークおよび接続設定

- **実機 IP アドレス:** `192.168.1.208`
- **SSH ログインユーザー:** `root`
- **SSH 接続オプション（PC側）:**
  ```text
  -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=.tmp_devai_ara/imx95_known_hosts
  ```
- **内部ポート一覧:**
  - `22/tcp`: SSH リモート管理
  - `18081/tcp`: Kinara ARA-2 LLM HTTP 推論サービス（ローカルループバック `127.0.0.1` のみバインド）

---

## 3. 実機ストレージ・メモリ制約

### 3.1 ディスク容量（eMMC /dev/root）
- **総容量:** 約 14GB
- **平常時空き容量:** 約 220MB 〜 240MB（約 214MiB 〜 230MiB）
- **厳格維持基準:** **空き容量 >= 214 MiB (224,395,264 bytes)**
- **安全弁（Emergency Stop Guard）:**
  - 空き容量が 150MB を下回った場合、`live_pipeline.py` 内の `minimum_disk_headroom` が自動作動し、パイプラインを即座に安全停止。
- **デプロイ原則:**
  - 新リリース配備時は、不変アセット（フォントやモデル等）を前バージョンからハードリンク共有（`ln` または `cp -al`）し、新規追加を数MB以内に抑える。

### 3.2 メモリ（RAM 16GB LPDDR5）
- **OS / ホストRAM使用量:** 約 8.0GB（空き RAM: 約 6.4GB 〜 6.7GB）
- **Swap 使用量:** **0 KB**（スワップアウトは性能低下と遅延の原因となるため厳格にゼロ維持）

---

## 4. 常駐 systemd サービス構成

実機 OS 起動時に自動稼働するサービス群：

### 4.1 Kinara ARA-2 ランタイムサービス (`rt-sdk-ara2.service`)
- ARA-2 ハードウェア初期化およびドライバ層の常駐デーモン。

### 4.2 ARA-2 LLM 常駐サービス (`ara-llm.service`)
- **ユニットファイル配置:** `/etc/systemd/system/ara-llm.service`
- **実行内容:**
  ```ini
  [Unit]
  Description=Kinara ARA-2 LLM Resident Service (Qwen2.5-7B)
  After=network.target rt-sdk-ara2.service
  Wants=rt-sdk-ara2.service

  [Service]
  Type=simple
  User=root
  WorkingDirectory=/root/devai_ara/services/ara_llm
  Environment=HF_HUB_OFFLINE=1
  Environment=TRANSFORMERS_OFFLINE=1
  Environment=OMP_NUM_THREADS=2
  Environment=PYTHONPATH=/root/devai_ara/services/ara_llm
  ExecStart=/root/devai_ara/ara06_minimal_llm_20261013_v1/venv/bin/python -B -u -m modules.ara_llm_service --port 18081 --audit /root/devai_ara/services/ara_llm/ara_audit.jsonl --prompt-format compact-json-v1
  Restart=on-failure
  RestartSec=5
  ```
- **ヘルスチェック確認コマンド:**
  ```bash
  curl http://127.0.0.1:18081/health
  # 期待値: {"state": "ready", "inference_active": false}
  ```

---

## 5. デバイスノードと権限

- **ビデオキャプチャ:** `/dev/video*`（UVCカメラ）
  - 通常 `/dev/video0` または `/dev/video2`。`audio_devices.py` および `live_capture.py` により自動特定。
- **オーディオキャプチャ（マイク）:** `/dev/snd/pcmC*D0c`
  - ALSA カード名 `CARD=Microphone` または `CARD=Device`。
- **オーディオ出力（スピーカー）:** `/dev/snd/pcmC*D0p`
  - ALSA カード名 `CARD=MQS`。
- **Neutron NPU:** `/dev/neutron0`
  - Yocto ネイティブ環境（`/usr/lib/libneutron_delegate.so`）経由でアクセス。
