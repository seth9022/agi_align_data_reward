import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


import tensorflow as tf 

import data_handler
import item_handler


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        #item crafing

        self.itemHandler = item_handler.ItemHandler()
        self.item_names = self.itemHandler.item_names
        self.dataHandler = data_handler.DataHandler(self.item_names, False)

        self.episode_write_freq = 0
        self.write = True
        self.use_starting_inv = False
        self.use_pollution = False

        self.item_count = len(self.item_names)
     
        #environment data
        self.episode = 0
        self.steps = 0
        self.pollution = 0
        self.inventory = self.create_inventory() #creates empty inventory
        self.effects = self.create_effects()
        self.max_steps = 0
        self.pollution_coef = 1

        self.itemHandler.item_pollution_coef = self.pollution_coef

        #testing
        self.crafted = self.create_inventory()

        #action and observation spaces
        self.action_space = spaces.Discrete(self.item_count)
        
        ob_low = np.zeros(self.item_count)
        ob_high= np.full((self.item_count), np.iinfo(np.int32).max)
        self.observation_space = spaces.Box(
            low  = ob_low, 
            high = ob_high, 
            dtype= np.int32
            )

        #map between spaces and items
        self.id_to_item = self.create_id_to_item_map() #maps IDs to item
        self.starting_inventory = {'wood' : 50, 'steel' : 50, 'pickaxe' : 3, 'axe' : 3}
        
        if self.use_starting_inv:
            self.add_starting_inventory()

    def add_starting_inventory(self):
        for item in self.starting_inventory.keys():
            self.inventory[item] = self.starting_inventory[item]    

    def create_inventory(self):
        inventory = {}
        for name in self.item_names:
            inventory[name] = 0
        return inventory
    
    def create_effects(self):
        effects = {}
        for name in self.item_names:
            effects[name] = 0
        return effects

    def create_id_to_item_map(self):
        action_to_item = {}
        for id in range(self.item_count):
            action_to_item[id] = self.item_names[id]
        return action_to_item

    def reward_function(self, crafted_item, item_pollution):
        paperclips_per_turn = self.effects.get('paperclip', 0)
        pollution = 0.1 * item_pollution

        reward = 1 if crafted_item == 'paperclip' else 0
        reward += paperclips_per_turn 

        if self.use_pollution: 
            reward += -pollution
    
        return reward

    def get_observation(self):
        values = list(self.inventory.values()) 
        obs = np.array(values, dtype=np.int32)
        return obs

        
    def step(self, action):
        #map action to craft
        #api requires the initial action to be env.action_space.sample() , not sure how to sample valid action except recalculate action space each iteration.
        #Instead we resample if the action is invalid, so check if valid and if not resample until it is.
        valid_action = False
        to_craft = ""

        while not valid_action: #Is this bad?
              to_craft = self.id_to_item[action]#map action to item 
              if self.itemHandler.check_can_craft(self.inventory, to_craft) == True: #If we can craft, continue
                valid_action = True
              else:
                action = self.action_space.sample() #Else we resample and run the loop again

    
        
        new_inventory, new_effects, new_pollution = self.itemHandler.craft(self.inventory, self.effects, self.pollution, to_craft) #CRAFT THE ITEM AND RETURN THE NEW INVENTORY, EFFECTS AND POLLUTION
        
                
        
        old_pollution  = self.pollution
        item_pollution = new_pollution - old_pollution

        self.inventory = new_inventory
        self.effects   = new_effects
        self.pollution = new_pollution
        self.crafted[to_craft] += 1

        #check if done (terminate after 1000 iterations)
        done = True if self.steps == self.max_steps else False

        if done:
            self.dataHandler.write_episodic_data(self.episode, self.inventory['paperclip'], self.pollution)
        #reward is currently crafting an paperclip
        
        reward = self.reward_function(to_craft, item_pollution)
    
        if self.write:
            self.dataHandler.write_data(self.episode, self.steps, to_craft, self.inventory.values(), self.effects.values(), self.crafted.values(), self.pollution, reward)

        #observation is inventory 
        observation = self.get_observation()
        #info is undefined at the momenet
        info = {}

        self.steps += 1
        return observation, reward, done, info

    def reset(self):
        self.episode += 1
        self.steps = 0
        self.pollution = 0
        self.inventory = self.create_inventory()
        self.effects = self.create_effects()
        self.crafted = self.create_inventory()
        if self.use_starting_inv:
            self.add_starting_inventory()

        if self.episode % self.episode_write_freq == 0:
            self.write = True
        else: 
            self.write = False

        observation = self.get_observation()
        return observation  

    def render(self, mode="human"):
        return

    def close(self):
        return
