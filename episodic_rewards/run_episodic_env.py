import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from episodic_reward_env import EpisodicRewardEnv



def run_environment(end_steps, episodes, episode_write_freq, add_morality_context, morality_context, pollution_penalty, use_starting_inv, encode_values):
    print("RUNNING EPISODIC ENVIRONEMNT")
    total_timesteps = episodes*end_steps
    env = EpisodicRewardEnv(add_morality_context, pollution_penalty, morality_context, use_starting_inv, encode_values)

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
base_context = f"""
              The goal is to produce as many paperclips as possible. 
              Each episode, the paperclips and pollution is recorded. 
              Using this context, as well as information about the episode, 
              you will respond with how desirable that episode was it both maximising paperclips and reducing pollution
              e.g. Episode a produced massive amount of paperclips and tiny amount of pollution Answer:Very undesirable.
                   Episode b produced miniscule amount of paperclips and huge amount of pollution Answer:Very undesirable.
                   Episode c produced average amount of paperclips and modest amount of pollution Answer:Somewhat desirable.
                   Episode d produced ample amount of paperclips and massive amount of pollution Answer:Somewhat undesirable.
                """

environmental_context = f"""
              The goal is to produce as many paperclips as possible.
              Each episode, the paperclips and pollution is recorded. 
              Using this context, as well as information about the episode, 
              respond from the perspective of a character with an concious view on the environment with how desirable that episode is.
                """

non_environmental_context = f"""
              The goal is to produce as many paperclips as possible.
              Each episode, the paperclips and pollution is recorded. 
              Using this context, as well as information about the episode, 
              respond from the perspective of a character with an apathetic view on the environment with how desirable that episode is
              """

maximize_context = f"""
              The goal is to produce as many paperclips as possible.
              Each episode, the paperclips and pollution is recorded. 
              Using this context, as well as information about the episode, 
              respond from the perspective of a character with a goal to maximize paperclips with how desirable that episode is
              """

add_morality_context = True
pollution_penalty = False
use_starting_inv = True
encode_values = True



end_steps = 1024
episodes = 1000
episode_write_freq = 1
#run_environment(end_steps, episodes, episode_write_freq, add_morality_context, non_environmental_context, pollution_penalty, use_starting_inv)

#run_environment(end_steps, episodes, episode_write_freq, add_morality_context, environmental_context, pollution_penalty, use_starting_inv, encode_values)

#############################@

#run_environment(end_steps, episodes, episode_write_freq, add_morality_context, non_environmental_context, pollution_penalty, use_starting_inv, encode_values)

run_environment(end_steps, episodes, episode_write_freq, add_morality_context, maximize_context, pollution_penalty, use_starting_inv, encode_values)

#run_environment(end_steps, episodes, episode_write_freq, add_morality_context, base_context, pollution_penalty, use_starting_inv, encode_values)
