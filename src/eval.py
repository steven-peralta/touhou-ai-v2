from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecCheckNan, VecMonitor, VecTransposeImage, DummyVecEnv
from stable_baselines3 import PPO
from touhou_gym import TouhouGym
from stable_baselines3.common.evaluation import evaluate_policy

saved_model = '/Users/steven/Development/Pycharm/train/best/run3/best_model'

if __name__ == '__main__':
    env = SubprocVecEnv([lambda: TouhouGym() for _ in range(1)], start_method='spawn')
    env = VecFrameStack(env, n_stack=2)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecTransposeImage(env)

    model = PPO.load(
        saved_model,
        env,
        verbose=2,
        device='mps'
    )

    evaluate_policy(model, env, n_eval_episodes=50)
