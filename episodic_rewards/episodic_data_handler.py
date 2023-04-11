from pathlib import Path
import csv
import pandas as pd 
import visualize_data as visualize
import matplotlib.pyplot as plt

class DataHandler():
    def __init__(self, item_names, training):
        
        self.path = "data"
        if training:
            self.path = "train"
            
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents = True, exist_ok = True)
        
        self.item_names = item_names
        self.episodes_to_plot = 0

        self.data_directory = ""
        self.create_data_directory()

        self.inventory_data = []
        self.inventory_df = ""
        
        self.effects_data = []
        self.effects_df = ""

        self.crafted_data = []
        self.crafted_df = ""

        self.pollution_data =[]
        self.pollution_df = ""

        self.reward_data =[]
        self.reward_df = ""

        self.episodic_data = []
        self.episodic_df = ""


    def create_data_directory(self):
        #creates directory for this simulation
        folder_name = "simulation"

        directory_count = len(list(Path(self.path).glob(f"{folder_name}_*")))

        directory_name = "/" + folder_name + "_" + str(directory_count)

        data_directory = self.path + directory_name
        Path(data_directory).mkdir(parents = True, exist_ok = True)
        
        self.data_directory = data_directory
    
    def create_dataframe(self):

        self.episodic_df = pd.DataFrame(self.episodic_data)
        self.episodic_df.to_csv(self.data_directory + '/episodic_data.csv', index = False)
        #self.episodic_df.plot(subplots=True)
        #plt.tight_layout()
        #plt.show()

        
    def write_episodic_data(self, episode, paperclips, pollution, reward, sentiment):
        row_data = [episode, paperclips, pollution, reward, sentiment]
        self.episodic_data.append(row_data)
    

    def write_data(self, episode, step, action, inventory, effects, crafted, pollution, reward):
        inventory_row_data = [episode, step, action] + list(inventory)
        self.inventory_data.append(inventory_row_data)
        
        effect_row_data = [episode, step, action] + list(effects)
        self.effects_data.append(effect_row_data)

        crafted_row_data = [episode, step, action] + list(crafted)
        self.crafted_data.append(crafted_row_data)

        pollution_row_data = [episode, step, pollution] 
        self.pollution_data.append(pollution_row_data)

        reward_row_data = [episode, step, reward]
        self.reward_data.append(reward_row_data)


    
    def get_data_directory(self):
        return self.data_directory

    def get_train_csv(self):
        return self.train_csv
    
    def get_test_csv(self):
        return self.test_csv
       
