import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import CustomEnv
end_steps = 10240
episodes = 100
episode_write_freq = 10

total_timesteps = episodes*end_steps

env = CustomEnv()
env.max_steps = end_steps
env.episode_write_freq = episode_write_freq
log_path = env.dataHandler.get_data_directory()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
model.learn(total_timesteps)#play with this time steps, bigger better

vec_env = model.get_env()
obs = vec_env.reset()
vec_env.write = True
for i in range(end_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()
env.dataHandler.create_dataframe()
env.close()

#David Silver RL lectures
#Evaluate the reward
#PLOT EVERYTHING!!!