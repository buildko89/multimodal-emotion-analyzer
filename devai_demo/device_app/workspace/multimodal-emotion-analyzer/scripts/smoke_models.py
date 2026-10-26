"""Load verified local artifacts and exercise inference without camera or downloads."""
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_models import load_manifest, verify, sha256_file, artifact_path
from models.factory import create_emotion_detector
from modules.multimodal_analyzer import MultiModalAnalyzer


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--pipeline', action='store_true', help='Exercise the headless video/transcript application')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output',
                        help='Save all smoke artifacts here; use a new directory to preserve earlier results')
    args = parser.parse_args()
    manifest = load_manifest()
    ids = ['mediapipe-face-short-range', 'fer2013-mini-xception']
    if args.audio:
        ids.append('whisper-base')
    report = {'models': {}}
    for model_id in ids:
        spec = manifest['models'][model_id]
        valid, detail = verify(spec)
        if not valid:
            raise RuntimeError(f'{model_id}: {detail}')
        path = artifact_path(spec)
        report['models'][model_id] = {'sha256': sha256_file(path), 'size_bytes': path.stat().st_size}
    config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    detector = create_emotion_detector(config)
    try:
        report['blank_image_faces'] = len(detector.detect(np.zeros((128, 128, 3), dtype=np.uint8)))
        label, confidence, scores = detector._recognize_emotion_original(np.zeros((64, 64, 3), dtype=np.uint8))
        scores = np.asarray(scores)
        if scores.shape != (7,) or not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1, atol=1e-5):
            raise RuntimeError('Invalid mini-XCEPTION softmax output')
        report['synthetic_crop'] = {'label': label, 'confidence': confidence, 'scores': scores.tolist()}
    finally:
        detector.face_model.close()
    report['observation_example'] = MultiModalAnalyzer().analyze(
        {'emotion': 'Happy', 'confidence': 0.8}, {'text': '悲しいです'})
    if args.audio:
        from modules.whisper_recognizer import WhisperRecognizer
        recognizer = WhisperRecognizer(model_path=config['whisper']['model_path'])
        import torch
        import whisper
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(np.zeros(30 * 16000, dtype=np.float32))
            features = recognizer.model.encoder(mel.unsqueeze(0))
            logits = recognizer.model.decoder(torch.tensor([[50258]], dtype=torch.long), features)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Invalid Whisper decoder output')
        report['whisper_synthetic_shapes'] = {'encoder': list(features.shape), 'decoder': list(logits.shape)}
    if args.pipeline:
        import cv2
        from multimodal_app import MultiModalEmotionApp
        directory = args.output_dir / 'smoke_pipeline'
        directory.mkdir(parents=True, exist_ok=True)
        video = directory / 'blank.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 2, (128, 128))
        if not writer.isOpened():
            raise RuntimeError('Cannot create smoke video')
        for _ in range(4):
            writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
        writer.release()
        config['display']['show_window'] = False
        config['output']['output_dir'] = str(directory)
        config['logging']['output_dir'] = str(args.output_dir / 'logs')
        config_path = directory / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
        app = MultiModalEmotionApp(str(config_path))
        try:
            app.speech_result = {'language': 'ja', 'segments': [{'start': 0, 'end': 2, 'text': '悲しいです'}]}
            app._process_video(str(video), with_audio=True)
        finally:
            app.emotion_detector.face_model.close()
        events = [json.loads(line) for line in (directory / 'fusion_windows.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(events) != 1 or events[0]['status'] != 'unavailable' or 'no_face' not in events[0]['uncertainty']:
            raise RuntimeError('Blank video must produce an unavailable observation window')
        report['video_pipeline'] = {'frames': app.frame_count, 'status': events[0]['status'],
                                    'uncertainty': events[0]['uncertainty']}
    output = args.output_dir / 'model_smoke.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'[ok] Model load and synthetic inference; report: {output}')


if __name__ == '__main__':
    main()
