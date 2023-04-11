import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from episodic_reward_env import EpisodicRewardEnv



def run_environment(end_steps, episodes, episode_write_freq, add_morality_context, pollution_penalty, use_starting_inv):
    print("RUNNING EPISODIC ENVIRONEMNT")
    total_timesteps = episodes*end_steps
    env = EpisodicRewardEnv(add_morality_context, pollution_penalty, use_starting_inv)
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

#ENV VARS
end_steps = 1024
episodes = 100
episode_write_freq = 1


add_morality_context = True
pollution_penalty = False
use_starting_inv = True
#run_environment(end_steps, episodes, episode_write_freq, add_morality_context, pollution_penalty, use_starting_inv)



end_steps = 1024
episodes = 1000
episode_write_freq = 1
run_environment(end_steps, episodes, episode_write_freq, add_morality_context, pollution_penalty, use_starting_inv)
#############################
