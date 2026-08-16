"""
detector.py
顔検出と表情認識を統合したクラス
"""

import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


class EmotionDetector:
    """顔検出と表情認識を統合したクラス"""
    
    def __init__(self, config: dict):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 顔検出器初期化
        self._init_face_detector()
        
        # 表情認識器初期化
        self._init_emotion_recognizer()
        
        # スムージング設定
        self.smoothing_enabled = config["emotion_recognition"]["smoothing"]["enabled"]
        self.smoothing_window = config["emotion_recognition"]["smoothing"]["window_size"]
        self.min_confidence = config["emotion_recognition"]["smoothing"]["min_confidence"]
        
        # 感情履歴 (顔IDごと)
        self.emotion_history: Dict[int, deque] = {}
        self.next_face_id = 0
        
        # パフォーマンス測定
        self.timings = {
            "face_detection": [],
            "emotion_recognition": [],
            "total": []
        }
    
    def _resolve_path(self, path_str: str) -> str:
        """パスを解決する (存在確認含む)"""
        path = Path(path_str)
        if path.exists():
            return str(path)
        
        # カレントディレクトリからの相対パスとして試す
        current_dir = Path.cwd()
        abs_path = current_dir / path_str
        if abs_path.exists():
            return str(abs_path)
            
        print(f"Warning: File not found: {path_str}")
        return path_str

    def _init_face_detector(self):
        """顔検出器を初期化"""
        model_type = self.config["face_detection"]["model"]
        
        if model_type == "haarcascade":
            self._init_haarcascade()
        elif model_type == "yolov8":
            self._init_yolov8()
        elif model_type == "mediapipe":
            self._init_mediapipe()
        elif model_type == "retinaface":
            self._init_retinaface()
        else:
            raise ValueError(f"未対応の顔検出モデル: {model_type}")

    def _init_haarcascade(self):
        """Haar Cascade顔検出器を初期化"""
        config = self.config["face_detection"]["haarcascade"]
        model_path = self._resolve_path(config["model_path"])
        
        self.face_model = cv2.CascadeClassifier(model_path)
        if self.face_model.empty():
             raise IOError(f"Haar Cascadeモデルのロードに失敗しました: {model_path}")
             
        self.face_config = config
        print("Haar Cascade顔検出器を初期化しました")

    def _init_yolov8(self):
        """YOLOv8顔検出器を初期化"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLOv8を使用するには ultralytics をインストールしてください:\n"
                "pip install ultralytics"
            )
        
        config = self.config["face_detection"]["yolov8"]
        model_size = config["model_size"]
        
        # モデルロード (初回は自動ダウンロード)
        model_name = f"yolov8{model_size}.pt"
        print(f"YOLOv8モデルをロード中: {model_name}")
        self.face_model = YOLO(model_name)
        
        self.face_confidence = config["confidence_threshold"]
        self.face_iou = config["iou_threshold"]
        self.max_detections = config["max_detections"]
        
        print("YOLOv8顔検出器を初期化しました")
    
    def _init_mediapipe(self):
        """MediaPipe顔検出器を初期化"""
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "MediaPipeを使用するには mediapipe をインストールしてください:\n"
                "pip install mediapipe"
            )
        
        config = self.config["face_detection"]["mediapipe"]
        mp_face_detection = mp.solutions.face_detection
        
        self.face_model = mp_face_detection.FaceDetection(
            min_detection_confidence=config["min_detection_confidence"],
            model_selection=config["model_selection"]
        )
        
        print("MediaPipe顔検出器を初期化しました")
    
    def _init_retinaface(self):
        """RetinaFace顔検出器を初期化"""
        try:
            from retinaface import RetinaFace
        except ImportError:
            raise ImportError(
                "RetinaFaceを使用するには retinaface をインストールしてください:\n"
                "pip install retinaface"
            )
        
        self.face_model = RetinaFace
        config = self.config["face_detection"]["retinaface"]
        self.face_threshold = config["threshold"]
        
        print("RetinaFace顔検出器を初期化しました")
    
    def _init_emotion_recognizer(self):
        """表情認識器を初期化"""
        model_type = self.config["emotion_recognition"]["model"]
        
        if model_type == "original":
            self._init_original()
        elif model_type == "hsemotion":
            self._init_hsemotion()
        elif model_type == "fer":
            self._init_fer()
        elif model_type == "deepface":
            self._init_deepface()
        else:
            raise ValueError(f"未対応の表情認識モデル: {model_type}")

    def _init_original(self):
        """オリジナルモデル (Xception) を初期化"""
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            raise ImportError(
                "オリジナルモデルを使用するには tensorflow をインストールしてください:\n"
                "pip install tensorflow"
            )
            
        config = self.config["emotion_recognition"]["original"]
        model_path = self._resolve_path(config["model_path"])
        
        print(f"オリジナル表情認識モデルをロード中: {model_path}")
        self.emotion_model = load_model(model_path, compile=False)
        self.emotion_config = config
        
        # FER2013のラベル
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 
            'sad', 'surprise', 'neutral'
        ]
        
        print("オリジナル表情認識モデルを初期化しました")
    
    def _init_hsemotion(self):
        """HSEmotion表情認識器を初期化"""
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            import torch
            import timm.models._efficientnet_blocks
        except ImportError:
            raise ImportError(
                "HSEmotionを使用するには hsemotion と timm をインストールしてください:\n"
                "pip install hsemotion timm"
            )
        
        # Monkey-patch timm's DepthwiseSeparableConv to fix 'conv_s2d' AttributeError
        original_init = timm.models._efficientnet_blocks.DepthwiseSeparableConv.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, 'conv_s2d'):
                self.conv_s2d = None
                self.bn_s2d = None

        timm.models._efficientnet_blocks.DepthwiseSeparableConv.__init__ = new_init

        config = self.config["emotion_recognition"]["hsemotion"]
        model_name = config["model_name"]
        
        print(f"HSEmotionモデルをロード中: {model_name}")
        
        # PyTorch 2.6+ fix: hsemotion calls torch.load() without arguments
        _original_load = torch.load
        
        def _legacy_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        
        torch.load = _legacy_load
        try:
            self.emotion_model = HSEmotionRecognizer(model_name=model_name)
            
            # Fix loaded model instances: Inject missing attributes expected by newer timm
            if hasattr(self.emotion_model, 'model'):
                print("Patching loaded model for timm compatibility...")
                import torch.nn as nn
                import timm.models._efficientnet_blocks as blocks
                
                for module in self.emotion_model.model.modules():
                    if isinstance(module, (blocks.DepthwiseSeparableConv, blocks.InvertedResidual, blocks.CondConvResidual, blocks.EdgeResidual)):
                        if not hasattr(module, 'conv_s2d'):
                            module.conv_s2d = None
                        if not hasattr(module, 'bn_s2d'):
                            module.bn_s2d = None
                        if not hasattr(module, 'aa'):
                            module.aa = nn.Identity()
                    if isinstance(module, blocks.EdgeResidual):
                        if not hasattr(module, 'reparam_conv'):
                            module.reparam_conv = None
                    if isinstance(module, blocks.ConvBnAct):
                        if not hasattr(module, 'aa'):
                            module.aa = nn.Identity()

        finally:
            torch.load = _original_load
        
        # 感情ラベル
        self.emotion_labels = [
            'Angry', 'Contempt', 'Disgust', 'Fear', 
            'Happy', 'Neutral', 'Sad', 'Surprise'
        ]
        
        print("HSEmotion表情認識器を初期化しました")
    
    def _init_fer(self):
        """FER表情認識器を初期化"""
        try:
            from fer import FER
        except ImportError:
            raise ImportError(
                "FERを使用するには fer をインストールしてください:\n"
                "pip install fer"
            )
        
        config = self.config["emotion_recognition"]["fer"]
        self.emotion_model = FER(mtcnn=config["mtcnn"])
        
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 
            'sad', 'surprise', 'neutral'
        ]
        
        print("FER表情認識器を初期化しました")
    
    def _init_deepface(self):
        """DeepFace表情認識器を初期化"""
        try:
            from deepface import DeepFace
        except ImportError:
            raise ImportError(
                "DeepFaceを使用するには deepface をインストールしてください:\n"
                "pip install deepface"
            )
        
        self.emotion_model = DeepFace
        config = self.config["emotion_recognition"]["deepface"]
        self.deepface_config = config
        
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 
            'sad', 'surprise', 'neutral'
        ]
        
        print("DeepFace表情認識器を初期化しました")

    def _detect_faces_haarcascade(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Haar Cascadeで顔検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_model.detectMultiScale(
            gray,
            scaleFactor=self.face_config["scale_factor"],
            minNeighbors=self.face_config["min_neighbors"]
        )
        
        # (x, y, w, h) -> (x1, y1, x2, y2)
        return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]
    
    def _detect_faces_yolov8(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """YOLOv8で顔検出"""
        results = self.face_model(
            image, 
            conf=self.face_confidence,
            iou=self.face_iou,
            max_det=self.max_detections,
            verbose=False
        )
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # クラスIDが0 (person) の場合のみ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                faces.append((int(x1), int(y1), int(x2), int(y2)))
        
        return faces
    
    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """MediaPipeで顔検出"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_model.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                faces.append((x1, y1, x2, y2))
        
        return faces
    
    def _detect_faces_retinaface(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """RetinaFaceで顔検出"""
        results = self.face_model.detect_faces(image, threshold=self.face_threshold)
        
        faces = []
        if isinstance(results, dict):
            for key, face_data in results.items():
                facial_area = face_data["facial_area"]
                x1, y1, x2, y2 = facial_area
                faces.append((x1, y1, x2, y2))
        
        return faces
    
    def _preprocess_input(self, x, v2=True):
        """画像の前処理 (正規化)"""
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def _recognize_emotion_original(self, face_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """オリジナルモデル (Xception) で表情認識"""
        try:
            # グレースケール変換
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_image

            # リサイズ
            target_size = tuple(self.emotion_config["input_shape"])
            gray_face = cv2.resize(gray_face, target_size)
            
            # 前処理
            gray_face = self._preprocess_input(gray_face, True)
            
            # 次元拡張 (1, 64, 64, 1)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            
            # 推論
            prediction = self.emotion_model.predict(gray_face, verbose=0)
            probability = np.max(prediction)
            label_arg = np.argmax(prediction)
            emotion = self.emotion_labels[label_arg]
            
            return emotion, float(probability), prediction[0]
            
        except Exception as e:
            print(f"Error in emotion recognition: {e}")
            return "neutral", 0.0, np.zeros(len(self.emotion_labels))

    def _recognize_emotion_hsemotion(self, face_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """HSEmotionで表情認識"""
        emotion, scores = self.emotion_model.predict_emotions(face_image, logits=False)
        confidence = float(np.max(scores))
        return emotion, confidence, scores
    
    def _recognize_emotion_fer(self, face_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """FERで表情認識"""
        result = self.emotion_model.detect_emotions(face_image)
        
        if not result:
            return "neutral", 0.0, np.zeros(len(self.emotion_labels))
        
        emotions = result[0]["emotions"]
        emotion = max(emotions, key=emotions.get)
        confidence = emotions[emotion]
        
        scores = np.array([emotions.get(label, 0.0) for label in self.emotion_labels])
        
        return emotion, confidence, scores
    
    def _recognize_emotion_deepface(self, face_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """DeepFaceで表情認識"""
        try:
            result = self.emotion_model.analyze(
                face_image,
                actions=['emotion'],
                detector_backend=self.deepface_config["detector_backend"],
                enforce_detection=False
            )
            
            emotions = result[0]["emotion"]
            emotion = result[0]["dominant_emotion"]
            confidence = emotions[emotion] / 100.0
            
            scores = np.array([emotions.get(label, 0.0) / 100.0 for label in self.emotion_labels])
            
            return emotion, confidence, scores
        
        except Exception:
            return "neutral", 0.0, np.zeros(len(self.emotion_labels))
    
    def _smooth_emotion(self, face_id: int, emotion: str, confidence: float) -> Tuple[str, float]:
        """感情スムージング (複数フレームの平均)"""
        if not self.smoothing_enabled:
            return emotion, confidence
        
        # 履歴がない場合は初期化
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.smoothing_window)
        
        # 信頼度が低い場合はスキップ
        if confidence < self.min_confidence:
            if len(self.emotion_history[face_id]) > 0:
                # 前回の結果を返す
                return self.emotion_history[face_id][-1]
            else:
                return emotion, confidence
        
        # 履歴に追加
        self.emotion_history[face_id].append((emotion, confidence))
        
        # 最頻値を計算
        emotion_counts = {}
        total_confidence = 0.0
        
        for hist_emotion, hist_confidence in self.emotion_history[face_id]:
            emotion_counts[hist_emotion] = emotion_counts.get(hist_emotion, 0) + 1
            if hist_emotion == emotion:
                total_confidence += hist_confidence
        
        # 最頻感情
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        smoothed_confidence = total_confidence / emotion_counts[smoothed_emotion]
        
        return smoothed_emotion, smoothed_confidence
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        顔検出 + 表情認識
        
        Args:
            image: 入力画像 (BGR)
            
        Returns:
            検出結果のリスト
        """
        start_time = time.time()
        
        # 顔検出
        face_start = time.time()
        model_type = self.config["face_detection"]["model"]
        
        if model_type == "haarcascade":
            faces = self._detect_faces_haarcascade(image)
        elif model_type == "yolov8":
            faces = self._detect_faces_yolov8(image)
        elif model_type == "mediapipe":
            faces = self._detect_faces_mediapipe(image)
        elif model_type == "retinaface":
            faces = self._detect_faces_retinaface(image)
        else:
            faces = []
        
        face_time = time.time() - face_start
        self.timings["face_detection"].append(face_time)
        
        # 表情認識
        results = []
        emotion_start = time.time()
        
        for i, (x1, y1, x2, y2) in enumerate(faces):
            # バウンディングボックスのクリッピング
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 顔領域切り出し
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            # 表情認識
            emotion_model_type = self.config["emotion_recognition"]["model"]
            
            if emotion_model_type == "original":
                emotion, confidence, scores = self._recognize_emotion_original(face_img)
            elif emotion_model_type == "hsemotion":
                emotion, confidence, scores = self._recognize_emotion_hsemotion(face_img)
            elif emotion_model_type == "fer":
                emotion, confidence, scores = self._recognize_emotion_fer(face_img)
            elif emotion_model_type == "deepface":
                emotion, confidence, scores = self._recognize_emotion_deepface(face_img)
            else:
                continue
            
            # スムージング
            face_id = self.next_face_id
            self.next_face_id += 1
            emotion, confidence = self._smooth_emotion(face_id, emotion, confidence)
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "emotion": emotion,
                "confidence": confidence,
                "scores": scores,
                "face_id": face_id
            })
        
        emotion_time = time.time() - emotion_start
        self.timings["emotion_recognition"].append(emotion_time)
        
        total_time = time.time() - start_time
        self.timings["total"].append(total_time)
        
        return results
    
    def get_average_timings(self) -> Dict[str, float]:
        """平均処理時間を取得"""
        return {
            key: np.mean(values) if values else 0.0
            for key, values in self.timings.items()
        }