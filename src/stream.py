import math
import os
import subprocess
import threading
import time

import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GAME_WIDTH = 640
GAME_HEIGHT = 480
PANEL_WIDTH = 320
FRAME_WIDTH = GAME_WIDTH + PANEL_WIDTH
FPS = 30

FONT_CANDIDATES = (
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    '/System/Library/Fonts/Menlo.ttc',
    '/System/Library/Fonts/Monaco.ttf',
)

METRIC_LABELS = (
    ('time/total_timesteps', 'steps'),
    ('rollout/ep_hits_mean', 'train hits'),
    ('rollout/ep_rew_mean', 'train reward'),
    ('rollout/ep_frames_mean', 'train frames'),
    ('time/fps', 'fps'),
)

EVAL_DELTA_LABELS = (
    ('eval/mean_ep_frames', 'frames to 1st hit', True),
    ('eval/mean_ep_hits', 'eval hits', False),
    ('eval/mean_ep_cleared', 'eval clear rate', True),
    ('eval/mean_reward', 'eval reward', True),
)


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def format_value(value):
    if value is None:
        return '-'
    if abs(value) >= 1e5:
        return f'{value:,.0f}'
    if abs(value) >= 100:
        return f'{value:.0f}'
    return f'{value:.3g}'


def format_delta(delta):
    if delta is None:
        return ''
    sign = '+' if delta > 0 else ''
    if abs(delta) >= 100:
        return f'{sign}{delta:.0f}'
    return f'{sign}{delta:.2g}'


def delta_color(delta, higher_is_better):
    if delta is None or delta == 0:
        return (150, 150, 170)
    improving = (delta > 0) == higher_is_better
    return (120, 220, 140) if improving else (240, 110, 110)


class TwitchStream:
    def __init__(self, stream_key, title='touhou-ai', bitrate='2500k'):
        self.title = title
        self.font = load_font(18)
        self.small_font = load_font(14)
        self.metrics = {}
        self.eval_history = []
        self.status = 'starting'
        self.phase = 'starting'
        self.progress = None
        self.rate_samples = []
        self.lock = threading.Lock()
        self.latest_game_frame = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{FRAME_WIDTH}x{GAME_HEIGHT}', '-r', str(FPS), '-i', '-',
            '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency', '-pix_fmt', 'yuv420p',
            '-g', str(FPS * 2), '-b:v', bitrate, '-minrate', bitrate, '-maxrate', bitrate, '-bufsize', '5000k',
            '-c:a', 'aac', '-b:a', '64k', '-shortest',
            '-f', 'flv', f'rtmp://live.twitch.tv/app/{stream_key}',
        ], stdin=subprocess.PIPE)
        self.running = True
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def update_metrics(self, metrics):
        with self.lock:
            self.metrics.update(metrics)

    def set_status(self, status):
        with self.lock:
            self.status = status

    def record_eval(self, metrics):
        with self.lock:
            self.eval_history.append(dict(metrics))
            self.eval_history = self.eval_history[-2:]

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def set_progress(self, timesteps, steps_to_eval):
        now = time.time()
        with self.lock:
            self.rate_samples.append((now, timesteps))
            self.rate_samples = [(t, n) for t, n in self.rate_samples if now - t < 120]
            self.progress = (timesteps, steps_to_eval)

    def _eta(self):
        if len(self.rate_samples) < 2 or self.progress is None:
            return None
        (t0, n0), (t1, n1) = self.rate_samples[0], self.rate_samples[-1]
        if t1 <= t0 or n1 <= n0:
            return None
        return self.progress[1] / ((n1 - n0) / (t1 - t0))

    def push_frame(self, game_frame):
        with self.lock:
            self.latest_game_frame = game_frame

    def _compose(self):
        with self.lock:
            game = self.latest_game_frame
            metrics = dict(self.metrics)
            status = self.status
            phase = self.phase
            progress = self.progress
            eta = self._eta()
            history = list(self.eval_history)
        canvas = Image.new('RGB', (FRAME_WIDTH, GAME_HEIGHT), (16, 16, 24))
        canvas.paste(Image.fromarray(game), (0, 0))
        draw = ImageDraw.Draw(canvas)
        x = GAME_WIDTH + 16
        draw.text((x, 12), self.title, font=self.font, fill=(240, 240, 240))
        draw.text((x, 38), phase, font=self.small_font, fill=(255, 210, 120))
        y = 58
        for line in textwrap.wrap(status, 34)[:1]:
            draw.text((x, y), line, font=self.small_font, fill=(180, 180, 200))
        y = 82
        if progress is not None and not phase.startswith('evaluating'):
            timesteps, remaining = progress
            eval_freq = remaining if remaining else 1
            eta_text = f'~{int(eta // 60)}m {int(eta % 60):02d}s' if eta is not None else '...'
            draw.text((x, y), f'next eval in {remaining:,} steps  {eta_text}', font=self.small_font, fill=(180, 180, 200))
            draw.rectangle((x, y + 18, x + 288, y + 26), outline=(90, 90, 110))
            span = self.metrics.get('_eval_span', remaining)
            filled = 0 if span <= 0 else int(288 * max(0.0, 1.0 - remaining / span))
            if filled > 0:
                draw.rectangle((x, y + 18, x + filled, y + 26), fill=(120, 180, 255))
            y += 36
        else:
            y += 8
        pulse = 0.6 + 0.4 * abs(math.sin(time.time() * 2))
        draw.ellipse((x + 276, 40, x + 288, 52), fill=(int(255 * pulse), int(80 * pulse), int(80 * pulse)))
        latest = history[-1] if history else {}
        previous = history[-2] if len(history) > 1 else {}
        draw.text((x, y), 'latest eval vs previous', font=self.small_font, fill=(200, 200, 220))
        y += 18
        for key, label, higher_is_better in EVAL_DELTA_LABELS:
            value = latest.get(key)
            delta = value - previous[key] if value is not None and key in previous else None
            draw.text((x, y), label, font=self.small_font, fill=(150, 150, 170))
            draw.text((x + 160, y), format_value(value), font=self.font, fill=(240, 240, 240))
            draw.text((x + 232, y), format_delta(delta), font=self.font, fill=delta_color(delta, higher_is_better))
            y += 26
        y += 8
        for key, label in METRIC_LABELS:
            draw.text((x, y), label, font=self.small_font, fill=(150, 150, 170))
            draw.text((x + 160, y), format_value(metrics.get(key)), font=self.font, fill=(240, 240, 240))
            y += 26
        return np.asarray(canvas)

    def _pump(self):
        interval = 1.0 / FPS
        while self.running:
            start = time.time()
            frame = self._compose()
            try:
                self.process.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError):
                self.running = False
                break
            time.sleep(max(0.0, interval - (time.time() - start)))

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
