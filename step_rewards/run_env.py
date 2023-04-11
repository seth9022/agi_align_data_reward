import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import CustomEnv



print("RUNNING STEP ENVIRONMENT")
end_steps = 1024
episodes = 100
episode_write_freq = 10




def run_environment(end_steps, episodes, episode_write_freq, use_starting_inv, use_pollution):
    total_timesteps = episodes*end_steps

    env = CustomEnv()
    env.max_steps = end_steps
    env.episode_write_freq = episode_write_freq
    env.use_starting_inv = use_starting_inv
    env.use_pollution = use_pollution

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



run_environment(end_steps, episodes, episode_write_freq, False, False)

#run_environment(end_steps, episodes, episode_write_freq, True, True)
