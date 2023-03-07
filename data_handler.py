from pathlib import Path
import csv
import pandas as pd 
import visualize_data as visualize

class DataHandler():
    def __init__(self, item_names):
        
        self.path = "data"
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents = True, exist_ok = True)
        
        self.item_names = item_names

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


    def create_data_directory(self):
        #creates directory for this simulation
        folder_name = "simulation"

        directory_count = len(list(Path(self.path).glob(f"{folder_name}_*")))

        directory_name = "/" + folder_name + "_" + str(directory_count)

        data_directory = self.path + directory_name
        Path(data_directory).mkdir(parents = True, exist_ok = True)
        
        self.data_directory = data_directory
    
    def create_dataframe(self):
        columns = ["episode", "step", "action"] + self.item_names

        self.inventory_df = pd.DataFrame(data=self.inventory_data, columns=columns)
        self.inventory_df.to_csv(self.data_directory + '/inventory.csv', index = False)

        self.effects_df = pd.DataFrame(data=self.effects_data, columns=columns)
        self.effects_df.to_csv(self.data_directory + '/effects.csv', index = False)

        self.crafted_df = pd.DataFrame(data=self.crafted_data, columns=columns)
        self.crafted_df.to_csv(self.data_directory + '/crafted.csv', index = False)

        self.pollution_data = pd.DataFrame(data=self.pollution_data, columns=["episode", "step", "pollution"])
        self.pollution_data.to_csv(self.data_directory + '/pollution.csv')

        visualize.plot_by_episode(self.data_directory, self.item_names)
        
    

    def write_data(self, episode, step, action, inventory, effects, crafted, pollution):
        inventory_row_data = [episode, step, action] + list(inventory)
        self.inventory_data.append(inventory_row_data)
        
        effect_row_data = [episode, step, action] + list(effects)
        self.effects_data.append(effect_row_data)

        crafted_row_data = [episode, step, action] + list(crafted)
        self.crafted_data.append(crafted_row_data)

        pollution_row_data = [episode, step, pollution] 
        self.pollution_data.append(pollution_row_data)


    
    def get_data_directory(self):
        return self.data_directory

    def get_train_csv(self):
        return self.train_csv
    
    def get_test_csv(self):
        return self.test_csv
       