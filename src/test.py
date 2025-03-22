import multiprocessing
from touhou_gym import TouhouGym

import torch as th

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO


def start():
    run_name = 'run1'

    save_path = f'train/checkpoints/{run_name}'
    best_path = f'train/best/{run_name}'
    logs_path = f'train/logs'

    total_timesteps = 100_000_000
    save_freq = 100_000
    eval_freq = 100_000

    n_envs = multiprocessing.cpu_count()
    n_eval_envs = 1

    save_freq = max(save_freq // n_envs, 1)
    eval_freq = max(eval_freq // n_envs, 1)

    # training envs
    env = TouhouGym()

    # eval env
    eval_env = TouhouGym()

    # callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=logs_path,
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq, 1),
        save_path=save_path,
        name_prefix='touhou-ai',
        save_vecnormalize=True
    )

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]))

    model = PPO("CnnPolicy", env, device="mps", verbose=2, tensorboard_log=logs_path, policy_kwargs=policy_kwargs)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=run_name
        )
    except Exception:
        pass


start()
