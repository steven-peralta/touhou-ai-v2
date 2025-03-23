import os
from datetime import datetime
import multiprocessing
from touhou_gym import TouhouGym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecCheckNan, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO

def train(
        save_base_path,
        total_steps,
        n_envs,
        frame_stack_size,
        load_from_checkpoint,
        image_scale,
        greyscale,
        stage_num,
        random_stage,
        device
):
    run_name = datetime.now().strftime("touhou-%Y-%m-%d_%H-%M-%S")

    run = wandb.init(
        entity="k9rosie",
        project='touhou-ai-v2',
        sync_tensorboard=True,
        name=run_name
    )

    save_path = os.path.join(save_base_path, f'checkpoints/{run_name}')
    best_path = os.path.join(save_base_path, f'best/{run_name}')
    logs_path = os.path.join(save_base_path, f'logs')

    save_freq = 100_000
    eval_freq = 100_000

    learning_rate = 1e-4

    n_eval_envs = 1

    save_freq = max(save_freq // n_envs, 1)
    eval_freq = max(eval_freq // n_envs, 1)

    # training envs
    env = SubprocVecEnv([lambda: TouhouGym(image_scale=image_scale, greyscale=greyscale, stage_num=stage_num, random_stage=random_stage) for _ in range(n_envs)], start_method='spawn')
    env = VecFrameStack(env, n_stack=frame_stack_size)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecTransposeImage(env)

    # eval env
    eval_env = SubprocVecEnv([lambda: TouhouGym(image_scale=image_scale, greyscale=greyscale, stage_num=stage_num, random_stage=random_stage) for _ in range(n_eval_envs)], start_method='spawn')
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack_size)
    eval_env = VecCheckNan(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecTransposeImage(eval_env)

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
    wandb_callback = WandbCallback(
        verbose=2,
    )

    if load_from_checkpoint:
        model = PPO.load(load_from_checkpoint, env, device=device,
                         tensorboard_log=logs_path)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            device=device,
            verbose=2,
            tensorboard_log=logs_path,
            learning_rate=learning_rate,
        )

    try:
        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback, wandb_callback],
            tb_log_name=run_name
        )
    except Exception as e:
        print(e)
        run.alert(title="Run crashed", text=f"Run crashed with this error: {e}")

# if __name__ == '__main__':
#     run_name = datetime.now().strftime("touhou-%Y-%m-%d_%H-%M-%S")
#
#     run = wandb.init(
#         entity="k9rosie",
#         project='touhou-ai-v2',
#         sync_tensorboard=True,
#         name=run_name
#     )
#
#     save_path = f'train/checkpoints/{run_name}'
#     best_path = f'train/best/{run_name}'
#     logs_path = f'train/logs'
#
#     total_timesteps = 100_000_000
#     save_freq = 100_000
#     eval_freq = 100_000
#
#     learning_rate = 1e-4
#
#     n_envs = multiprocessing.cpu_count()
#     n_eval_envs = 1
#
#     save_freq = max(save_freq // n_envs, 1)
#     eval_freq = max(eval_freq // n_envs, 1)
#
#     # training envs
#     env = SubprocVecEnv([lambda: TouhouGym() for _ in range(n_envs)], start_method='spawn')
#     env = VecFrameStack(env, n_stack=2)
#     env = VecCheckNan(env)
#     env = VecMonitor(env)
#     env = VecTransposeImage(env)
#
#     # eval env
#     eval_env = SubprocVecEnv([lambda: TouhouGym() for _ in range(n_eval_envs)], start_method='spawn')
#     eval_env = VecFrameStack(eval_env, n_stack=2)
#     eval_env = VecCheckNan(eval_env)
#     eval_env = VecMonitor(eval_env)
#     eval_env = VecTransposeImage(eval_env)
#
#     # callbacks
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=best_path,
#         log_path=logs_path,
#         eval_freq=eval_freq,
#         n_eval_episodes=1,
#         deterministic=True
#     )
#     checkpoint_callback = CheckpointCallback(
#         save_freq=max(save_freq, 1),
#         save_path=save_path,
#         name_prefix='touhou-ai',
#         save_vecnormalize=True
#     )
#     wandb_callback = WandbCallback(
#         verbose=2,
#     )
#
#     # model = PPO(
#     #     "MultiInputPolicy",
#     #     env,
#     #     device="cuda",
#     #     verbose=2,
#     #     tensorboard_log=logs_path,
#     #     learning_rate=learning_rate,
#     # )
#
#     model = PPO.load("train/checkpoints/touhou-2025-03-23_03-48-39/touhou-ai_3600000_steps", env, device="cuda", tensorboard_log=logs_path)
#
#     #model = PPO.load(f"train/checkpoints/new_run/touhou-ai_1900000_steps", env, device="cuda", tensorboard_log=logs_path)
#
#     try:
#         model.learn(
#             total_timesteps=total_timesteps,
#             reset_num_timesteps=False,
#             progress_bar=True,
#             callback=[checkpoint_callback, eval_callback, wandb_callback],
#             tb_log_name=run_name
#         )
#     except Exception as e:
#         print(e)
#         run.alert(title="Run crashed", text=f"Run crashed with this error: {e}")
