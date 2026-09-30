import os
from datetime import datetime

import numpy as np

from touhou_gym import TouhouGym


def save_episode(path, observations, actions, rewards, hits, terminated, stage_num, action_repeat):
    data = {f'obs/{key}': np.stack([obs[key] for obs in observations]) for key in observations[0]}
    data['actions'] = np.stack(actions)
    data['rewards'] = np.array(rewards, dtype=np.float32)
    data['hits'] = np.array(hits, dtype=bool)
    data['terminated'] = np.array(terminated)
    data['stage'] = np.array(stage_num)
    data['action_repeat'] = np.array(action_repeat)
    np.savez_compressed(path, **data)


def record(
        output_dir,
        n_episodes,
        stage_num,
        random_stage,
        stages=None,
        lives=0,
        action_repeat=2,
        min_steps=60,
        game_res_path='./res/game/',
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    session = datetime.now().strftime("demo-%Y-%m-%d_%H-%M-%S")

    env = TouhouGym(
        disable_render=False,
        stage_num=stage_num,
        random_stage=random_stage,
        stages=stages,
        fps_limit=60,
        unlock_fps=False,
        game_path=game_res_path,
        lives=lives,
        action_repeat=action_repeat,
    )

    print("Recording: arrow keys to move, Z to shoot, LEFT SHIFT to focus, ESC to stop")

    total_steps = 0
    saved = 0
    quit_requested = False

    try:
        while saved < n_episodes and not quit_requested:
            observations, actions, rewards, hits = [], [], [], []
            episode_hits = 0
            terminated = False
            obs, _ = env.reset()

            try:
                while not terminated:
                    action = env.human_action()
                    next_obs, reward, terminated, _, info = env.step(action)
                    if info['quit']:
                        quit_requested = True
                        break
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    hits.append(info['hits'] > episode_hits)
                    episode_hits = info['hits']
                    obs = next_obs
            except KeyboardInterrupt:
                quit_requested = True

            if len(actions) < min_steps:
                print(f"Discarded episode with {len(actions)} steps")
                continue

            path = os.path.join(output_dir, f'{session}-stage{env.stage_num}-ep{saved}.npz')
            save_episode(path, observations, actions, rewards, hits, terminated, env.stage_num, action_repeat)
            total_steps += len(actions)
            saved += 1
            print(f"Saved {path}: {len(actions)} steps, return {sum(rewards):.2f}")
    finally:
        env.close()

    print(f"Recorded {saved} episodes, {total_steps} steps in {output_dir}")
