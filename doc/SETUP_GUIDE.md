# セットアップガイド

Windows/Linux PC環境での表情認識システムのセットアップ手順

---

## 目次

1. [動作環境](#動作環境)
2. [Python 3.12 インストール](#python-312-インストール)
3. [依存パッケージのインストール](#依存パッケージのインストール)
4. [実行方法](#実行方法)
5. [トラブルシューティング](#トラブルシューティング)
6. [GPU対応](#gpu対応)

---

## 動作環境

### 最小要件
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **CPU**: Intel Core i5 (第8世代以降) または同等
- **RAM**: 8GB以上
- **Python**: 3.12.x
- **カメラ**: USB Webカメラ (オプション)

### 推奨要件
- **CPU**: Intel Core i7 (第10世代以降) または同等
- **RAM**: 16GB以上
- **GPU**: NVIDIA GeForce GTX 1660以上 (CUDA対応、オプション)

---

## Python 3.12 インストール

### Windows

#### 方法1: 公式インストーラー (推奨)
1. [Python公式サイト](https://www.python.org/downloads/) にアクセス
2. "Download Python 3.12.x" をクリック
3. インストーラーを実行
4. **重要**: "Add Python 3.12 to PATH" にチェックを入れる
5. "Install Now" をクリック

#### 方法2: Microsoft Store
1. Microsoft Storeを開く
2. "Python 3.12" を検索
3. インストール

#### 確認
```powershell
python --version
# Python 3.12.x と表示されればOK
```

---

### Linux (Ubuntu/Debian)

#### deadsnakes PPAを使用 (推奨)
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

#### シンボリックリンク作成 (オプション)
```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
```

#### 確認
```bash
python3.12 --version
# Python 3.12.x と表示されればOK
```

---

### macOS

#### Homebrew使用 (推奨)
```bash
# Homebrewがない場合はインストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.12インストール
brew install python@3.12
```

#### 確認
```bash
python3.12 --version
# Python 3.12.x と表示されればOK
```

---

## 依存パッケージのインストール

### 1. プロジェクトディレクトリに移動
```bash
cd face_classification/face-classification-migrated/custom
```

### 2. 仮想環境作成 (推奨)

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux/macOS
```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. pipアップグレード
```bash
python -m pip install --upgrade pip
```

### 4. 依存パッケージインストール

#### 基本構成 (CPU版)
```bash
pip install -r requirements.txt
```

#### GPU対応 (NVIDIA GPU使用時)
```bash
# CUDA 11.8対応版 (CUDA Toolkit 11.8が必要)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

# その他のパッケージ
pip install -r requirements.txt
```

### 5. インストール確認
```bash
python -c "import cv2, numpy, ultralytics, hsemotion; print('OK')"
# "OK" と表示されればインストール成功
```

---

## 実行方法

### 基本実行 (カメラ入力)
```bash
python emotion_detector.py
```

### オプション指定

#### カメラID指定
```bash
python emotion_detector.py --camera 0
```

#### 動画ファイル処理
```bash
python emotion_detector.py --input sample_video.mp4
```

#### 画像ファイル処理
```bash
python emotion_detector.py --input sample_image.jpg
```

#### 設定ファイル指定
```bash
python emotion_detector.py --config my_config.yaml
```

### 終了方法
- ウィンドウ上で **`q`** キーを押す
- または **`ESC`** キーを押す

---

## トラブルシューティング

### カメラが開けない

**症状**: `ビデオソースを開けませんでした` エラー

**対処法**:
1. カメラID確認
   ```bash
   python emotion_detector.py --camera 1  # 別のIDを試す
   ```

2. カメラのアクセス許可確認
   - Windows: 設定 → プライバシー → カメラ
   - Linux: `sudo usermod -a -G video $USER` (再ログイン)

3. カメラが他のアプリで使用中でないか確認

---

### ModuleNotFoundError

**症状**: `ModuleNotFoundError: No module named 'xxx'`

**対処法**:
```bash
# 仮想環境がアクティブか確認
# Windows
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate

# 再インストール
pip install -r requirements.txt
```

---

### YOLOv8モデルダウンロードエラー

**症状**: モデルダウンロードが失敗する

**対処法**:
1. 手動ダウンロード
   ```bash
   # Ultralyticsリポジトリから直接ダウンロード
   # yolov8n.pt を custom/ ディレクトリに配置
   ```

2. プロキシ設定 (企業環境の場合)
   ```bash
   export HTTP_PROXY=http://proxy.example.com:8080
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```

---

### Windows で Visual C++ エラー

**症状**: `Microsoft Visual C++ 14.0 or greater is required`

**対処法**:
1. [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/) をダウンロード
2. "C++ によるデスクトップ開発" をインストール

---

### Linux で OpenCV エラー

**症状**: `ImportError: libGL.so.1: cannot open shared object file`

**対処法**:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

---

### 処理が遅い

**対処法**:

1. **軽量モデルに変更** (config.yaml)
   ```yaml
   face_detection:
     model: "mediapipe"  # yolov8 → mediapipe
   ```

2. **フレームスキップ** (config.yaml)
   ```yaml
   performance:
     frame_skip: 1  # 1フレームおきに処理
   ```

3. **解像度を下げる** (config.yaml)
   ```yaml
   camera:
     width: 320
     height: 240
   ```

4. **GPU使用** (GPU対応セクション参照)

---

## GPU対応

### NVIDIA GPU使用時の追加セットアップ

#### 1. CUDA Toolkit インストール

**Windows**:
1. [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) をダウンロード
2. インストーラーを実行

**Linux**:
```bash
# Ubuntu 22.04の場合
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### 2. cuDNN インストール

1. [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (要アカウント登録)
2. CUDA 11.x対応版 (cuDNN 8.6+) をダウンロード
3. インストール

#### 3. PyTorch GPU版インストール
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. ONNX Runtime GPU版インストール
```bash
pip install onnxruntime-gpu
```

#### 5. GPU使用を有効化 (config.yaml)
```yaml
performance:
  use_gpu: true
```

#### 6. 動作確認
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# CUDA: True と表示されればOK
```

---

## 設定のカスタマイズ

`config.yaml` を編集して動作をカスタマイズできます。

### 主要な設定項目

#### 顔検出モデル変更
```yaml
face_detection:
  model: "yolov8"  # "mediapipe", "retinaface"
```

#### 表情認識モデル変更
```yaml
emotion_recognition:
  model: "hsemotion"  # "fer", "deepface"
```

#### 結果保存
```yaml
output:
  save_results: true
  save_video: true
  save_csv: true
```

詳細は `config.yaml` のコメントを参照してください。

---

## アンインストール

### 仮想環境削除
```bash
# Windows
deactivate
Remove-Item -Recurse -Force venv

# Linux/macOS
deactivate
rm -rf venv
```

---

## サポート

問題が解決しない場合は、以下の情報を含めて報告してください:

1. OS とバージョン
2. Python バージョン (`python --version`)
3. エラーメッセージ全文
4. 実行したコマンド

---

## ライセンス

MIT License

---

## 参考リンク

- [Python公式サイト](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)
- [PyTorch](https://pytorch.org/)
