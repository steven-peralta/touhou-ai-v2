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
    ('rollout/ep_hits_mean', 'train hits/stage'),
    ('rollout/ep_rew_mean', 'train reward'),
    ('rollout/ep_frames_mean', 'train frames/stage'),
    ('eval/mean_ep_frames', 'eval frames to hit'),
    ('eval/mean_ep_hits', 'eval hits'),
    ('eval/mean_ep_cleared', 'eval clear rate'),
    ('train/entropy_loss', 'entropy'),
    ('train/explained_variance', 'explained var'),
    ('train/clip_fraction', 'clip fraction'),
    ('time/fps', 'fps'),
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


class TwitchStream:
    def __init__(self, stream_key, title='touhou-ai', bitrate='2500k'):
        self.title = title
        self.font = load_font(18)
        self.small_font = load_font(14)
        self.metrics = {}
        self.status = 'waiting for eval'
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

    def push_frame(self, game_frame):
        with self.lock:
            self.latest_game_frame = game_frame

    def _compose(self):
        with self.lock:
            game = self.latest_game_frame
            metrics = dict(self.metrics)
            status = self.status
        canvas = Image.new('RGB', (FRAME_WIDTH, GAME_HEIGHT), (16, 16, 24))
        canvas.paste(Image.fromarray(game), (0, 0))
        draw = ImageDraw.Draw(canvas)
        x = GAME_WIDTH + 16
        draw.text((x, 14), self.title, font=self.font, fill=(240, 240, 240))
        y = 40
        for line in textwrap.wrap(status, 34)[:2]:
            draw.text((x, y), line, font=self.small_font, fill=(180, 180, 200))
            y += 18
        y = 84
        for key, label in METRIC_LABELS:
            draw.text((x, y), label, font=self.small_font, fill=(150, 150, 170))
            draw.text((x, y + 16), format_value(metrics.get(key)), font=self.font, fill=(240, 240, 240))
            y += 40
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
