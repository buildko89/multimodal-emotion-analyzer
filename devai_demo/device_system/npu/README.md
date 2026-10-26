# i.MX95 Neutron NPU サービス＆モデル定義

## 1. 概要
本ディレクトリには、NXP i.MX95 内蔵 Neutron NPU（`/dev/neutron0`）を活用した高速推論（< 0.2ms）を実現するためのモデルバイナリ、UDS IPC サービススクリプト、テストスクリプトを集約しています。

## 2. 構成ファイル一覧

- **`blazeface_neutron.tflite`**: Neutron NPU 向けにコンパイルされた BlazeFace 顔検出モデル
- **`fer_mini_xception_neutron.tflite`**: Neutron NPU 向けにコンパイルされた Mini-Xception 表情認識モデル
- **`neutron_fer_service.py`**: Yocto システム Python（3.14）環境で常駐し、Unix Domain Socket (`/tmp/neutron_fer.sock`) 経由で推論リクエストを捌く UDS サービス
- **`neutron_fer_service_standalone.py`**: 単体テスト・デバッグ用スタンドアロン NPU サービス
- **`convert_fer_to_tflite.py`**: Keras/TFLite から INT8 量子化モデルへの変換スクリプト
- **各種テストスクリプト (`test_*.py`)**: NPU デリゲート・レイテンシ・クライアント検証スクリプト

## 3. ABI 分離の原則（Yocto System Python vs App venv）
- i.MX95 提供の `libneutron_delegate.so` は Yocto BSP ネイティブ環境（Python 3.14 + `tflite_runtime`）専用です。
- アプリ仮想環境（Python 3.12 や pip 版 TensorFlow）から直接ロードすると SIGSEGV クラッシュを起こすため、本サービスをシステム側で起動し、アプリ側からは UDS ソケット経由でテンソルを送受信します。
