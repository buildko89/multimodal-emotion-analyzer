"""Optional bounded speech and assistant-only avatar for the existing live display."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from modules.driver_response_policy import resolve_preferences
from modules.framebuffer_display import FramebufferDisplay


class SpeechOutput:
    def __init__(self, output, catalog, *, device=None, cooldown_seconds=15, popen=None):
        self.output=Path(output); self.device=device; self.cooldown=cooldown_seconds
        self.popen=popen or subprocess.Popen; self.process=None;self.revision=None
        self.seen=set();self.last_started=None;self.last_text_key=None;self.current_text='';self.status='muted' if not device else 'idle'
        self.log=None;self.error=None;self.catalog={}
        for row in catalog.get('records',[]):
            if not row.get('generated'):continue
            path=Path(row['path']).resolve()
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('speech_catalog_hash_mismatch')
            self.catalog[row['text']]=path

    def snapshot(self):
        return dict(status=self.status,text=self.current_text,device=self.device,error=self.error,
                    physical_speaker_verified=False)

    def stop(self, reason='stopped'):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            self.process=None
        if self.log is not None:self.log.close();self.log=None
        self.current_text='';self.status=reason

    def update(self, snapshot, now_ns):
        revision=snapshot.get('preference_revision')
        if revision!=self.revision:
            self.stop('preference_changed');self.revision=revision
        if self.process is not None and self.process.poll() is not None:
            code=self.process.returncode
            self.stop('idle' if code==0 else 'audio_error')
            if code:self.error='audio_process_exit_'+str(code)
        result=snapshot.get('response') or {}
        active=resolve_preferences(snapshot.get('preference_history',[]))['active']
        age=(now_ns-result.get('source_end_ns',now_ns))/1e9
        allowed=(not snapshot.get('input_faults') and active!='no_conversation'
            and result.get('preference_revision')==revision
            and result.get('status') in ('accepted','policy_output','fallback')
            and 0<=age<=snapshot.get('max_result_age_seconds',300))
        text=result.get('text','')
        if not allowed or not text:
            self.stop('suppressed');return self.snapshot()
        key=(revision,result.get('window_id'),result.get('available_ns'),text)
        if key in self.seen:return self.snapshot()
        if len(self.seen)>=256:self.seen.clear()
        self.seen.add(key)
        if not self.device:self.status='muted';return self.snapshot()
        if text not in self.catalog:self.status='text_only';return self.snapshot()
        if self.last_text_key==(revision,text):
            if self.process is None:self.status='duplicate_suppressed'
            return self.snapshot()
        if active!='repeat_guidance' and self.last_started is not None and (
                now_ns-self.last_started<self.cooldown*1e9):
            self.status='cooldown';return self.snapshot()
        self.stop('replaced')
        try:
            self.log=(self.output/'speech_playback.log').open('ab')
            self.process=self.popen(['aplay','-q','-D',self.device,str(self.catalog[text])],
                stdin=subprocess.DEVNULL,stdout=self.log,stderr=self.log)
            self.current_text=text;self.last_started=now_ns;self.last_text_key=(revision,text);self.status='speaking'
        except OSError as error:
            self.stop('audio_error');self.error=str(error)
        return self.snapshot()


def avatar_image(font, state, size=(310,125), snapshot=None):
    from PIL import Image, ImageDraw
    image = Image.new('RGB', size, '#203449')
    draw = ImageDraw.Draw(image)

    visual = snapshot.get('visual_result') if isinstance(snapshot, dict) else None
    display_emotion = visual.get('display_emotion') if visual else 'neutral'

    # 状態・表情によるスタイル分岐（絵文字豆腐を完全排除し、Meiryo対応テキストマーカーを使用）
    if state == 'speaking':
        face_color = '#00b894'  # エメラルドグリーン
        label_title = 'ARA 相棒AI'
        label_desc = '● 音声案内中'
        desc_color = '#55efc4'
        eye_type = 'happy'
        mouth_type = 'talking'
    elif display_emotion == 'happy':
        face_color = '#ffd166'  # 明るいイエロー
        label_title = 'ARA 相棒AI'
        label_desc = '● 笑顔を検知！'
        desc_color = '#ffeaa7'
        eye_type = 'arc_happy'
        mouth_type = 'big_smile'
    elif display_emotion == 'yawn':
        face_color = '#a29bfe'  # パステルパープル
        label_title = 'ARA 相棒AI'
        label_desc = '● 眠気・あくび検知'
        desc_color = '#dfe6e9'
        eye_type = 'sleepy'
        mouth_type = 'yawn_circle'
    elif display_emotion == 'sad':
        face_color = '#74b9ff'  # パステルブルー
        label_title = 'ARA 相棒AI'
        label_desc = '● 疲労・困惑検知'
        desc_color = '#dff9fb'
        eye_type = 'worried'
        mouth_type = 'sad_down'
    elif display_emotion == 'angry':
        face_color = '#ff7675'  # サーモンオレンジ
        label_title = 'ARA 相棒AI'
        label_desc = '● 緊張・集中検知'
        desc_color = '#ffbe76'
        eye_type = 'focused'
        mouth_type = 'firm'
    else:
        # neutral / idle
        face_color = '#76c6d4'  # シアンミント
        label_title = 'ARA 相棒AI'
        label_desc = '● 運転見守り中'
        desc_color = '#bad6e8'
        eye_type = 'normal'
        mouth_type = 'gentle_smile'

    # 顔の輪郭描画
    draw.ellipse((12, 12, 110, 110), fill=face_color)

    # 目の描画
    if eye_type == 'arc_happy':
        draw.arc((32, 40, 52, 56), start=180, end=360, fill='#132337', width=4)
        draw.arc((70, 40, 90, 56), start=180, end=360, fill='#132337', width=4)
    elif eye_type == 'sleepy':
        draw.line((34, 48, 52, 48), fill='#132337', width=4)
        draw.line((70, 48, 88, 48), fill='#132337', width=4)
    elif eye_type == 'worried':
        draw.line((32, 42, 52, 46), fill='#132337', width=3)
        draw.line((88, 42, 68, 46), fill='#132337', width=3)
        draw.ellipse((38, 48, 48, 56), fill='#132337')
        draw.ellipse((74, 48, 84, 56), fill='#132337')
    elif eye_type == 'focused':
        draw.line((32, 46, 52, 43), fill='#132337', width=3)
        draw.line((88, 46, 68, 43), fill='#132337', width=3)
        draw.ellipse((38, 47, 48, 55), fill='#132337')
        draw.ellipse((74, 47, 84, 55), fill='#132337')
    else:
        draw.ellipse((38, 43, 48, 53), fill='#132337')
        draw.ellipse((74, 43, 84, 53), fill='#132337')

    # 口の描画
    if mouth_type == 'talking':
        draw.ellipse((46, 68, 76, 90), fill='#132337')
    elif mouth_type == 'big_smile':
        draw.arc((42, 62, 80, 88), start=0, end=180, fill='#132337', width=5)
    elif mouth_type == 'yawn_circle':
        draw.ellipse((48, 66, 74, 92), fill='#132337')
    elif mouth_type == 'sad_down':
        draw.arc((44, 76, 78, 96), start=180, end=360, fill='#132337', width=4)
    else:
        draw.arc((46, 68, 76, 84), start=0, end=180, fill='#132337', width=3)

    # テキスト描画
    draw.text((125, 26), label_title, font=font, fill='white')
    draw.text((125, 68), label_desc, font=font, fill=desc_color)
    return image


class DemoFramebufferDisplay(FramebufferDisplay):
    """Opt-in staged live profile; the source baseline display remains unchanged."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=Path(os.environ.get('ARA_DEMO_SPEECH_CATALOG') or
                  Path(__file__).resolve().parents[2]/'assets/speech_manifest.json')
        catalog=json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {'records':[]}
        self.speech=SpeechOutput(self.output,catalog,device=os.environ.get('ARA_DEMO_AUDIO_DEVICE') or None)
        self.last_speech=None

    def draw(self,snapshot,*,save=False):
        import numpy as np
        from PIL import Image
        record=super().draw(snapshot,save=False)
        state=self.speech.update(snapshot,time.monotonic_ns())
        if state!=self.last_speech:
            from modules.live_interaction import atomic_json
            atomic_json(self.output/'speech_state.json',state);self.last_speech=state
        patch=avatar_image(self.fonts[24],state['status'],snapshot=snapshot)
        v=self.var
        if v.xres<640 or v.yres<480:raise ValueError('demo_display_too_small')
        pixels=np.asarray(patch)
        packed=np.zeros(pixels.shape[:2],dtype=np.uint32)
        for channel,name in enumerate(('red','green','blue')):
            packed|=pixels[:,:,channel].astype(np.uint32)<<getattr(v,name).offset
        if v.transp.length:packed|=np.uint32((2**v.transp.length-1)<<v.transp.offset)
        x=v.xoffset+v.xres-patch.width-18;y=v.yoffset+12
        self.rows[y:y+patch.height,x:x+patch.width]=packed
        if save:
            target=self.rows[v.yoffset:v.yoffset+v.yres,v.xoffset:v.xoffset+v.xres]
            rgb=np.stack([((target>>getattr(v,n).offset)&255).astype(np.uint8) for n in ('red','green','blue')],axis=2)
            Image.fromarray(rgb).save(self.output/'display_latest.png')
            if not np.array_equal(self.rows[y:y+patch.height,x:x+patch.width],packed):
                raise RuntimeError('demo_avatar_readback_mismatch')
        record.update(readback_checked=save,assistant_avatar=state['status'],speech=state,
                      avatar_represents='assistant_output_only_not_driver_state')
        return record

    def close(self):
        try:self.speech.stop('session_closed')
        finally:super().close()
