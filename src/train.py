import os
from datetime import datetime

from PIFE import PIFEFeatureExtractor, CombinedPIFEFeatureExtractor
from touhou_gym import TouhouGym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecTransposeImage, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def train(
        save_base_path,
        total_steps,
        n_envs,
        n_eval_envs,
        load_from_checkpoint,
        image_scale,
        greyscale,
        stage_num,
        frame_stack_size,
        random_stage,
        device,
        n_steps,
        batch_size,
        n_epochs,
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

    learning_rate = linear_schedule(3e-4)
    clip_range = linear_schedule(0.2)

    save_freq = max(save_freq // n_envs, 1)
    eval_freq = max(eval_freq // n_envs, 1)

    # training envs
    env = SubprocVecEnv([lambda: TouhouGym(disable_render=True, stage_num=stage_num, random_stage=random_stage) for _ in range(n_envs)], start_method='spawn')
    env = VecFrameStack(env, n_stack=frame_stack_size)
    env = VecMonitor(env)

    # eval env
    eval_env = SubprocVecEnv([lambda: TouhouGym(disable_render=False, stage_num=stage_num, random_stage=random_stage, fps_limit=60, unlock_fps=False) for _ in range(n_eval_envs)], start_method='spawn')
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack_size)
    eval_env = VecMonitor(eval_env)

    # callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=logs_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_envs,
        deterministic=True
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq, 1),
        save_path=save_path,
        name_prefix='touhou-ai',
        save_vecnormalize=True
    )
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run_name}",
        gradient_save_freq=100,
        verbose=2,
    )

    policy_kwargs = dict(
        features_extractor_class=CombinedPIFEFeatureExtractor,
    )

    if load_from_checkpoint:
        model = PPO.load(load_from_checkpoint, env, device=device,
                         tensorboard_log=logs_path)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
            verbose=2,
            tensorboard_log=logs_path,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
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
