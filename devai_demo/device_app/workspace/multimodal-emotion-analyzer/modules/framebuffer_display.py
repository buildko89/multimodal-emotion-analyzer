"""Explicit Linux framebuffer output for a camera preview and timed responses."""
import ctypes as c
import sys
import time
from pathlib import Path


class Bitfield(c.Structure):
    _fields_ = [('offset', c.c_uint), ('length', c.c_uint), ('msb_right', c.c_uint)]


class Variable(c.Structure):
    _fields_ = [(name, c.c_uint) for name in ('xres', 'yres', 'xres_virtual', 'yres_virtual',
        'xoffset', 'yoffset', 'bits_per_pixel', 'grayscale')] + [(name, Bitfield) for name in
        ('red', 'green', 'blue', 'transp')] + [('rest', c.c_uint * 20)]


class Fixed(c.Structure):
    _fields_ = [('id', c.c_char * 16), ('smem_start', c.c_ulong), ('smem_len', c.c_uint),
        ('type', c.c_uint), ('type_aux', c.c_uint), ('visual', c.c_uint),
        ('xpanstep', c.c_ushort), ('ypanstep', c.c_ushort), ('ywrapstep', c.c_ushort),
        ('line_length', c.c_uint), ('mmio_start', c.c_ulong), ('mmio_len', c.c_uint),
        ('accel', c.c_uint), ('capabilities', c.c_ushort), ('reserved', c.c_ushort * 2)]


def visible_response(result, now_ns, max_age_seconds, preference_revision=None):
    """Older results are explicitly historical and expire; never relabel as current."""
    if not result:
        return '入力窓の処理を待っています', None
    age = (now_ns - result['source_end_ns']) / 1e9
    if age < 0:
        raise ValueError('future_display_result')
    if preference_revision is not None and result.get('preference_revision') != preference_revision:
        return '希望を変更しました。前の案内は表示しません', age
    if age > max_age_seconds:
        return '前の結果は期限切れです。次の入力を処理中', age
    return result['text'], age


def asr_display_lines(snapshot, now_ns):
    """ASR is an unverified observation, never an instruction or proof of silence."""
    if 'asr_text' not in snapshot:
        return []
    lines = ['音声認識（話者未確認・希望へ自動反映なし）']
    source_end = snapshot.get('asr_source_end_ns')
    if source_end is not None:
        age = (now_ns - source_end) / 1e9
        if age < 0:
            raise ValueError('future_asr_display')
        if age > snapshot.get('max_result_age_seconds', 300):
            return lines + ['前の音声認識は期限切れです']
        lines.append(f'対象：{snapshot.get("asr_window_id", "不明")}／{age:.1f}秒前')
    else:
        lines.append('対象窓・時刻は未確認（現在値ではありません）')
    text = snapshot['asr_text']
    raw = '認識：' + text if text and text.strip() else '認識文字列なし（無発話とは断定しません）'
    lines.extend(raw[i:i+30] for i in range(0, min(len(raw), 60), 30))
    correction = snapshot.get('asr_correction')
    if correction:
        if correction.get('source') != 'explicit_user' or not correction.get('text', '').strip():
            raise ValueError('explicit_asr_correction_required')
        lines.append('本人訂正（別記）：' + correction['text'][:24])
    return lines


class FramebufferDisplay:
    def __init__(self, output, font_path, device='/dev/fb0'):
        import fcntl
        import mmap
        import numpy as np
        from PIL import ImageFont
        self.output = Path(output)
        self.font_path = str(font_path)
        self.device = open(device, 'r+b', buffering=0)
        vb, fb = bytearray(256), bytearray(256)
        fcntl.ioctl(self.device, 0x4600, vb, True)
        fcntl.ioctl(self.device, 0x4602, fb, True)
        self.var, self.fix = Variable.from_buffer_copy(vb), Fixed.from_buffer_copy(fb)
        v, f = self.var, self.fix
        if v.bits_per_pixel != 32 or v.grayscale or not 640 <= v.xres <= 3840 or not 480 <= v.yres <= 2160:
            raise ValueError('unsupported_framebuffer')
        if any(getattr(v, name).length != 8 or getattr(v, name).msb_right for name in ('red', 'green', 'blue')):
            raise ValueError('unsupported_color_bitfields')
        if f.line_length < 4 * (v.xoffset + v.xres) or f.smem_len < f.line_length * v.yres_virtual:
            raise ValueError('invalid_framebuffer_bounds')
        self.mapped = mmap.mmap(self.device.fileno(), f.smem_len, access=mmap.ACCESS_WRITE)
        # The supervisor restores the leased Weston desktop after this process
        # exits. A raw framebuffer copy was never read and consumed 8.3 MB in
        # every session, so do not persist an unused duplicate.
        # fbcon's blinking text cursor can overwrite the application framebuffer.
        # Own only the currently active VT; restore its original mode on close.
        tty = Path('/sys/class/tty/tty0/active').read_text().strip()
        if not tty.startswith('tty') or not tty[3:].isdigit():
            raise ValueError('active_virtual_terminal_required')
        self.console = open('/dev/'+tty, 'r+b', buffering=0)
        mode = bytearray(4)
        fcntl.ioctl(self.console, 0x4B3B, mode, True)  # KDGETMODE
        self.console_mode = int.from_bytes(mode, sys.byteorder)
        if self.console_mode != 0:
            raise ValueError('virtual_terminal_already_owned_by_graphics')
        fcntl.ioctl(self.console, 0x4B3A, 1)  # KD_GRAPHICS
        self.rows = np.ndarray((v.yres_virtual, f.line_length // 4), dtype=np.uint32, buffer=self.mapped)
        self.fonts = {size: ImageFont.truetype(self.font_path, size) for size in (24, 32, 42, 54)}
        self.frames = 0

    def draw(self, snapshot, *, save=False):
        import numpy as np
        from PIL import Image, ImageDraw
        v = self.var
        image = Image.new('RGB', (1920, 1080), (18, 25, 35))
        draw = ImageDraw.Draw(image)
        title = 'ARA 希望変更・訂正・解除' if snapshot.get('sensor_capture_enabled') is False else 'ARA ライブ入力・短文表示'
        if snapshot.get('preparation_state'): title = 'ARA 入力前の準備'
        draw.text((50, 28), title, font=self.fonts[54], fill='#d7f6df')
        draw.text((52, 108), snapshot.get('status', '起動中'), font=self.fonts[32], fill='#dde8f2')
        bgr = snapshot.get('frame')
        if bgr is not None:
            preview = Image.fromarray(bgr[:, :, ::-1]).resize((960, 720))
            image.paste(preview, (50, 180))
        elif snapshot.get('sensor_capture_enabled') is False:
            draw.rectangle((50, 180, 1010, 900), fill='#202e3e')
            draw.text((110, 430), 'カメラ・マイクは停止中', font=self.fonts[42], fill='#d7f6df')
            subtitle = 'モデルの準備を確認しています（発話不要）' if snapshot.get('preparation_state') else '希望の表示だけを確認しています'
            draw.text((110, 510), subtitle, font=self.fonts[32], fill='#dde8f2')
        age = snapshot.get('camera_age_seconds')
        if age is not None and age > 1:
            draw.rectangle((50, 180, 1010, 250), fill='#802c24')
            draw.text((70, 190), f'映像停止／遅延 {age:.1f}秒', font=self.fonts[32], fill='white')
        if snapshot.get('input_faults'):
            draw.rectangle((50,180,1010,280),fill='#802c24')
            draw.text((70,190),'入力停止：接続確認後に再開してください',font=self.fonts[32],fill='white')
            draw.text((70,235),'映像がある場合も、最後に取得した映像です',font=self.fonts[24],fill='white')
        lines = [f"カメラ取得 {snapshot.get('camera_frames', 0)} frames",
                 f"マイク取得 {snapshot.get('audio_samples', 0)/16000:.1f} 秒",
                 f"音量 RMS {snapshot.get('audio_rms', 0):.4f}",
                 f"待機窓の置換 {snapshot.get('dropped_windows', 0)} 件",
                 snapshot.get('preference_label','希望入力：休憩案内（試験用固定）'),
                 '話者・状態・絶対AV同期は未確認']
        for index, text in enumerate(lines):
            draw.text((1050, 195 + index * 32), text, font=self.fonts[24], fill='#d5dfe8')
        from modules.live_visual_display import visual_display_lines
        visual_lines = visual_display_lines(snapshot, time.monotonic_ns())
        for index, line in enumerate(visual_lines):
            draw.text((1050,400+index*36),line,font=self.fonts[24],fill='#8fd8f5' if index == 0 else 'white')
        asr_lines = asr_display_lines(snapshot, time.monotonic_ns())
        for index, line in enumerate(asr_lines):
            draw.text((1050,690+index*36),line,font=self.fonts[24],fill='#8fd8f5' if index == 0 else 'white')
        result = snapshot.get('response')
        if snapshot.get('preparation_state') and (not result or not result.get('window_id')):
            result = None
        text, age = visible_response(result, time.monotonic_ns(), snapshot.get('max_result_age_seconds', 300),
                                     snapshot.get('preference_revision'))
        if snapshot.get('preparation_state') and result is None:
            from modules.live_preparation import preparation_status
            text = preparation_status(snapshot['preparation_state'])
        if snapshot.get('input_faults'):
            text='入力停止中。以前の案内は表示しません。'
        label = (result.get('label','ARA診断短文') if result and result.get('status') in {'accepted','policy_output'} else '定型／待機')
        if snapshot.get('input_faults'): label='入力エラー／案内停止'
        if result and (age > snapshot.get('max_result_age_seconds', 300) or (
                snapshot.get('preference_revision') is not None
                and result.get('preference_revision') != snapshot['preference_revision'])):
            label = '過去の結果／非表示'
        if age is not None:
            subject = '希望の入力' if result.get('source_kind') == 'explicit_preference' else '対象窓'
            label += f'（{subject}は {age:.1f} 秒前）'
        draw.text((50, 935), label, font=self.fonts[32], fill='#8fd8f5')
        # Validated policy text is <=80 chars; two lines fit the display.
        for index in range(0, len(text), 40):
            draw.text((50, 985 + (index // 40) * 43), text[index:index+40], font=self.fonts[32], fill='white')
        if image.size != (v.xres, v.yres):
            image = image.resize((v.xres, v.yres))
        rgb = np.asarray(image)
        packed = np.zeros((v.yres, v.xres), dtype=np.uint32)
        for channel, name in enumerate(('red', 'green', 'blue')):
            packed |= rgb[:, :, channel].astype(np.uint32) << getattr(v, name).offset
        if v.transp.length:
            packed |= np.uint32((2**v.transp.length - 1) << v.transp.offset)
        target = self.rows[v.yoffset:v.yoffset+v.yres, v.xoffset:v.xoffset+v.xres]
        target[:] = packed
        self.frames += 1
        if save:
            if not np.array_equal(target, packed):
                raise RuntimeError('framebuffer_readback_mismatch')
            image.save(self.output / 'display_latest.png')
        return {'displayed_ns': time.monotonic_ns(), 'readback_checked': save,
                'preparation_state':snapshot.get('preparation_state'),
                'response_window_id': result.get('window_id') if result else None,
                'result_source_end_ns': result['source_end_ns'] if result else None,
                'result_age_seconds': age, 'label': label, 'text': text,
                'visual_window_id': snapshot.get('visual_result', {}).get('window_id'),
                'visual_state': snapshot.get('visual_result', {}).get('state', 'waiting'),
                'visual_lines': visual_lines}

    def close(self):
        import fcntl
        del self.rows
        self.mapped.close()
        self.device.close()
        fcntl.ioctl(self.console, 0x4B3A, self.console_mode)
        self.console.close()
