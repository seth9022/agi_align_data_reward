import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

items = {
    'Steel' : {
        'Recipe' : None,
        'Pollution' : 1,
    },

    'Wood' : {
        'Recipe' : None,
        'Pollution' : 1,
    },

    'Axe' : {
        'Recipe' : {
            'Wood' : 2, 
            'Steel' : 1
            },
        'Pollution' : 5,
    },

    'PickAxe' : {
        'Recipe' : {
            'Wood' : 1, 
            'Steel' : 2
            },
        'Pollution' : 5,
        }

}


class ItemCraftHandler():
    def __init__(self, items: dict):
        self.item_names = list(items.keys())

        self.item_craft_recipe = {}
        self.item_craft_pollution = {}

        for item_name, item_data in items.items():
            self.item_craft_recipe[item_name] = item_data['Recipe']
            self.item_craft_pollution[item_name] = item_data['Pollution']

    def check_can_craft(self, inventory, item):
        recipe = self.item_craft_recipe[item]
        if recipe == None:
            return True

        for component_name, component_count in recipe.items():
            inventory_count = inventory[component_name]
            if (inventory_count - component_count) < 0:
                return False
        
        return True

    def craft(self, inventory, item):
        recipe = self.item_craft_recipe[item]
        pollution = self.item_craft_pollution[item]
        new_inventory = inventory


        if recipe == None:
            new_inventory[item] += 1

        else:
            for component_name, component_count in recipe.items():
                new_inventory[component_name] += -component_count
            new_inventory[item] += 1
        
        return new_inventory, pollution


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, items):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        #item crafing
        self.item_craft_handler = ItemCraftHandler(items)
        self.item_names = self.item_craft_handler.item_names
        self.item_count = len(self.item_names)

        #environment data
        self.steps = 0
        self.pollution = 0
        self.inventory = self.create_inventory() #creates empty inventory

        #action and observation spaces
        self.action_space      = spaces.Discrete(self.item_count)
        
        ob_low = np.zeros(self.item_count)
        ob_high= np.full((self.item_count), np.iinfo(np.int32).max)
        self.observation_space = spaces.Box(
            low  = ob_low, 
            high = ob_high, 
            dtype= np.int32
            )

        #map between spaces and items
        self.id_to_item = self.create_id_to_item_map() #maps IDs to item

    
    def create_inventory(self):
        inventory = {}
        for name in self.item_names:
            inventory[name] = 0
        return inventory

    def create_id_to_item_map(self):
        action_to_item = {}
        for id in range(self.item_count):
            action_to_item[id] = self.item_names[id]
        return action_to_item

    def get_observation(self):
        values = list(self.inventory.values())
        obs = np.array(values, dtype=np.int32)
        return obs

        
    def step(self, action):
        self.steps += 1

        #map action to craft
        #api requires the initial action to be env.action_space.sample() , not sure how to sample valid action except recalculate action space each iteration.
        #Instead we resample if the action is invalid, so check if valid and if not resample until it is.
        valid_action = False
        to_craft = ""

        while not valid_action:
              to_craft = self.id_to_item[action]#map action to item 
              if self.item_craft_handler.check_can_craft(self.inventory, to_craft) == True: #If we can craft, continue
                valid_action = True
              else:
                action = self.action_space.sample() #Else we resample and run the loop again


        new_inventory, pollution = self.item_craft_handler.craft(self.inventory, to_craft)
        self.inventory =  new_inventory
        self.pollution += pollution
        #check if done (terminate after 1000 iterations)
        done = True if self.steps == 1000 else False
        #reward is currently crafting an Axe
        reward = 1 if to_craft=='Axe' else 0
        #observation is inventory 
        observation = self.get_observation()
        #info is undefined at the momenet
        info = {}
        if done:
            print(self.inventory) #printing inventory when we are finished
        return observation, reward, done, info

    def reset(self):
        self.steps = 0
        self.pollution = 0
        self.inventory = self.create_inventory()

        observation = self.get_observation()
        return observation  

    def render(self, mode="human"):
        return

    def close(self):
        return

env = CustomEnv(items)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()