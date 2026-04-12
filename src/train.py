import os
from datetime import datetime

from PIFE import PIFEFeatureExtractor, CombinedPIFEFeatureExtractor
from touhou_gym import TouhouGym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecTransposeImage, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO


def linear_schedule(initial_value, min_value=0.0):
    def func(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return func

def train(
        save_base_path,
        total_steps,
        n_envs,
        n_eval_envs,
        load_from_checkpoint,
        stage_num,
        frame_stack_size,
        random_stage,
        device,
        n_steps,
        batch_size,
        n_epochs,
        n_eval_episodes=5,
        wandb_entity=None,
        game_res_path='./res/game/',
        learning_rate=3e-4,
        ent_coef=0.0,
        reset_timesteps=False,
        eval_freq=100_000,
        train_stages=None,
        eval_stages=None,
        render_train=False,
):
    run_name = datetime.now().strftime("touhou-%Y-%m-%d_%H-%M-%S")

    run = wandb.init(
        entity=wandb_entity or os.getenv('WANDB_ENTITY', 'k9rosie'),
        project='touhou-ai-v2',
        sync_tensorboard=True,
        name=run_name
    )

    save_path = os.path.join(save_base_path, f'checkpoints/{run_name}')
    best_path = os.path.join(save_base_path, f'best/{run_name}')
    logs_path = os.path.join(save_base_path, f'logs')

    save_freq = 100_000

    lr_schedule = learning_rate  # constant LR
    clip_range = linear_schedule(0.2, min_value=0.05)

    save_freq = max(save_freq // n_envs, 1)
    eval_freq = max(eval_freq // n_envs, 1)

    # training envs
    env = SubprocVecEnv([lambda: TouhouGym(disable_render=not render_train, stage_num=stage_num, random_stage=random_stage, stages=train_stages, game_path=game_res_path) for _ in range(n_envs)], start_method='spawn')
    env = VecFrameStack(env, n_stack=frame_stack_size)
    env = VecMonitor(env)

    # eval env — render only if a display is available
    has_display = os.environ.get('DISPLAY') is not None
    eval_stages_list = eval_stages or train_stages
    eval_env = SubprocVecEnv([lambda: TouhouGym(disable_render=not has_display, stage_num=stage_num, random_stage=random_stage, stages=eval_stages_list, fps_limit=60, unlock_fps=False, game_path=game_res_path) for _ in range(n_eval_envs)], start_method='spawn')
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack_size)
    eval_env = VecMonitor(eval_env)

    # callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=logs_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
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
            learning_rate=lr_schedule,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
        )

    try:
        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback, wandb_callback],
            tb_log_name=run_name
        )
    except Exception as e:
        print(e)
        run.alert(title="Run crashed", text=f"Run crashed with this error: {e}")
        raise
    finally:
        run.finish()
