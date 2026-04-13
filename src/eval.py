import sys

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecCheckNan, VecMonitor, VecTransposeImage, DummyVecEnv
from stable_baselines3 import PPO
from touhou_gym import TouhouGym
from stable_baselines3.common.evaluation import evaluate_policy

def eval_model(
        n_eval_envs,
        n_eval_episodes,
        frame_stack_size,
        load_from_checkpoint,
        random_stage,
        stage_num,
        device,
        game_res_path='./res/game/',
        stages=None,
):
    if not load_from_checkpoint:
        print("Error: --load is required for evaluation")
        sys.exit(1)

    eval_env = SubprocVecEnv([lambda: TouhouGym(disable_render=False, stage_num=stage_num,
                                                random_stage=random_stage, stages=stages, fps_limit=60, unlock_fps=False, game_path=game_res_path, mortal=True) for _ in
                              range(n_eval_envs)], start_method='spawn')
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack_size)
    eval_env = VecMonitor(eval_env)

    model = PPO.load(
        load_from_checkpoint,
        eval_env,
        verbose=2,
        device=device
    )

    evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
