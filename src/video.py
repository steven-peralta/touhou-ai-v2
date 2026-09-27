import subprocess

import numpy as np

from sb3_contrib import RecurrentPPO
from touhou_gym import TouhouGym


def record_video(output_path, load_from_checkpoint, stage_num, game_res_path, action_repeat, device, seed=0):
    model = RecurrentPPO.load(load_from_checkpoint, device=device)
    env = TouhouGym(disable_render=False, unlock_fps=True, game_path=game_res_path, stage_num=stage_num, action_repeat=1)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '60', '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23', '-movflags', '+faststart',
        output_path,
    ], stdin=subprocess.PIPE)

    obs, _ = env.reset(seed=seed)
    lstm_state = None
    episode_start = np.ones(1, dtype=bool)
    frames = 0
    hits_at = []
    last_hits = 0
    action = None
    try:
        while True:
            if frames % action_repeat == 0:
                action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=False)
                episode_start[:] = False
            obs, _, terminated, _, info = env.step(action)
            frames += 1
            encoder.stdin.write(np.ascontiguousarray(env.render()[..., :3]).tobytes())
            if info['hits'] > last_hits:
                hits_at.append(round(frames / 60))
                last_hits = info['hits']
            if terminated:
                break
    finally:
        encoder.stdin.close()
        encoder.wait()
        env.close()

    print(f"Saved {output_path}: stage {stage_num}, {info['hits']} hits at seconds {hits_at}, {frames // 60}s, cleared {info['cleared']}")
