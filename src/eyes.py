from touhou_gym import TouhouGym

image_scale = 6
greyscale = False
stage_num = 1
env = TouhouGym(image_scale=image_scale, greyscale=greyscale, stage_num=stage_num)
obs, info = env.reset()
n_steps = 500
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env._agent_eyes()
