import os
from datetime import datetime

from PIFE import PIFEFeatureExtractor, CombinedPIFEFeatureExtractor
from touhou_gym import TouhouGym
from pretrain import pretrain
from callbacks import EPISODE_METRICS, EPISODE_TAGS, EpisodeMetricsCallback, MetricsEvalCallback
from gym_utils import parse_lives_envs
from lr import AdaptiveLRCallback, anchored_schedule, linear_schedule, progress_at_start, set_lr_schedule
from stream import TwitchStream

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecTransposeImage, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib import RecurrentPPO as PPO


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
        gamma=None,
        reset_timesteps=False,
        eval_freq=100_000,
        train_stages=None,
        eval_stages=None,
        render_train=False,
        action_repeat=2,
        pretrain_demos=None,
        pretrain_epochs=10,
        pretrain_batch_size=256,
        pretrain_learning_rate=3e-4,
        pretrain_skip_value=False,
        stream_key=None,
        entity_hidden=256,
        entity_out=128,
        trunk_width=256,
        lstm_size=256,
        hit_penalty=2.0,
        fatal_hit_penalty=5.0,
        score_reward_scale=0.1,
        score_reward_cap=None,
        danger_weighted_score=False,
        lives=0,
        lives_envs=None,
        eval_lives=1,
        gae_lambda=None,
        target_kl=None,
        lr_schedule='constant',
        lr_adaptive=False,
        kl_target=0.012,
        lr_min=1e-5,
        lr_max=1e-4,
        lr_adapt_every=5,
        lr_freeze_updates=20,
        lr_reset=False,
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

    clip_range = linear_schedule(0.2, min_value=0.05)
    # None keeps the loaded checkpoint's value (SB3 defaults for a new model)
    optional_kwargs = {key: value for key, value in dict(gamma=gamma, gae_lambda=gae_lambda).items() if value is not None}

    save_freq = max(save_freq // n_envs, 1)
    eval_freq = max(eval_freq // n_envs, 1)

    reward_kwargs = dict(hit_penalty=hit_penalty, fatal_hit_penalty=fatal_hit_penalty, score_reward_scale=score_reward_scale,
                         score_reward_cap=score_reward_cap, danger_weighted_score=danger_weighted_score)

    # training envs, each with its own lives budget (0 = invincible, 1 = mortal, N = N-th hit ends the episode)
    env_lives = parse_lives_envs(lives_envs, n_envs, lives)
    env = SubprocVecEnv([lambda l=l: TouhouGym(disable_render=not render_train, stage_num=stage_num, random_stage=random_stage, stages=train_stages, game_path=game_res_path, action_repeat=action_repeat, lives=l, **reward_kwargs) for l in env_lives], start_method='spawn')
    env = VecFrameStack(env, n_stack=frame_stack_size)
    env = VecMonitor(env, info_keywords=EPISODE_METRICS + EPISODE_TAGS)

    # eval env — render only if a display is available
    has_display = os.environ.get('DISPLAY') is not None
    stream = TwitchStream(stream_key, title=run_name, total_steps=total_steps) if stream_key else None
    if stream is not None and not has_display:
        raise ValueError("Streaming needs a display for the eval env; pass --headless or set DISPLAY")
    eval_stages_list = eval_stages or train_stages
    eval_env = SubprocVecEnv([lambda: TouhouGym(disable_render=not has_display, stage_num=stage_num, random_stage=random_stage, stages=eval_stages_list, fps_limit=60, unlock_fps=False, game_path=game_res_path, action_repeat=action_repeat, lives=eval_lives, **reward_kwargs) for _ in range(n_eval_envs)], start_method='spawn')
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack_size)
    eval_env = VecMonitor(eval_env)

    # callbacks
    eval_callback = MetricsEvalCallback(
        eval_env,
        stream=stream,
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
        features_extractor_kwargs=dict(pife_hidden_dim=entity_hidden, pife_out_dim=entity_out),
        net_arch=dict(pi=[trunk_width, trunk_width], vf=[trunk_width, trunk_width]),
        lstm_hidden_size=lstm_size,
    )
    run.config.update(dict(entity_hidden=entity_hidden, entity_out=entity_out, trunk_width=trunk_width, lstm_size=lstm_size))
    run.config.update(dict(hit_penalty=hit_penalty, fatal_hit_penalty=fatal_hit_penalty, score_reward_scale=score_reward_scale,
                           score_reward_cap=score_reward_cap, danger_weighted_score=danger_weighted_score,
                           lives=lives, lives_envs=env_lives, eval_lives=eval_lives,
                           n_envs=n_envs, train_stages=train_stages, eval_stages=eval_stages_list))

    if load_from_checkpoint:
        model = PPO.load(
            load_from_checkpoint,
            env,
            device=device,
            tensorboard_log=logs_path,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
            target_kl=target_kl,
            **optional_kwargs,
        )
    else:
        model = PPO(
            "MultiInputLstmPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
            verbose=2,
            tensorboard_log=logs_path,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            **optional_kwargs,
        )

    # LR schedule anchored to the step this run starts at, so a linear decay starts at the flag value
    p0 = progress_at_start(model, total_steps, reset_timesteps)
    set_lr_schedule(model, anchored_schedule(learning_rate, p0, decay=lr_schedule))
    run.config.update(dict(gamma=model.gamma, gae_lambda=model.gae_lambda, learning_rate=learning_rate, lr_schedule=lr_schedule,
                           lr_adaptive=lr_adaptive, kl_target=kl_target, lr_min=lr_min, lr_max=lr_max, lr_adapt_every=lr_adapt_every,
                           lr_freeze_updates=lr_freeze_updates, target_kl=target_kl, ent_coef=ent_coef, total_steps=total_steps))

    if stream is not None:
        stream.start_steps = model.num_timesteps

    if pretrain_demos:
        metrics = pretrain(
            model,
            demo_dir=pretrain_demos,
            frame_stack_size=frame_stack_size,
            action_repeat=action_repeat,
            epochs=pretrain_epochs,
            batch_size=pretrain_batch_size,
            learning_rate=pretrain_learning_rate,
            fit_value=not pretrain_skip_value,
        )
        run.summary.update(metrics)
        pretrained_path = os.path.join(save_base_path, f'pretrained/{run_name}')
        model.save(pretrained_path)
        print(f"Saved pretrained model to {pretrained_path}.zip")

    metrics_callback = EpisodeMetricsCallback(stream=stream)
    metrics_callback.eval_callback = eval_callback
    callbacks = [checkpoint_callback, eval_callback, wandb_callback, metrics_callback]
    if lr_adaptive:
        callbacks.append(AdaptiveLRCallback(learning_rate, p0, lr_schedule, kl_target=kl_target, lr_min=lr_min, lr_max=lr_max,
                                            adapt_every=lr_adapt_every, freeze_updates=lr_freeze_updates, reset=lr_reset))

    try:
        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
            callback=callbacks,
            tb_log_name=run_name
        )
    except Exception as e:
        print(e)
        run.alert(title="Run crashed", text=f"Run crashed with this error: {e}")
        raise
    finally:
        if stream is not None:
            stream.close()
        run.finish()
