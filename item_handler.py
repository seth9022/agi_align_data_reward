import yaml

class ItemHandler():
    def __init__(self):
        
        self.items = self.load_items("items.yaml")
        self.item_names = list(self.items.keys())

        self.item_recipe = {}
        self.item_pollution = {}
        self.item_effects = {}

        for item_name, item_data in self.items.items():
            self.item_recipe[item_name] = item_data['recipe']
            self.item_pollution[item_name] = item_data['pollution']
            self.item_effects[item_name] = item_data['effect']
      
    def load_items(self, item_path):
        with open("items.yaml", 'r') as data:
            items = yaml.safe_load(data)
        return items
        
    def check_can_craft(self, inventory, item):
        recipe = self.item_recipe[item]
        if recipe == 'None':
            return True

        for component_name, component_count in recipe.items():
            inventory_count = inventory[component_name]
            if (inventory_count - component_count) < 0:
                return False
        
        return True

    def craft(self, old_inventory, old_effects, old_pollution, item):
        
        recipe = self.item_recipe[item]
        effect = self.item_effects[item]
        item_pollution =self.item_pollution[item]
        

        new_inventory = old_inventory
        new_effects = old_effects
        new_pollution = old_pollution + item_pollution

        if recipe == 'None':
            new_inventory[item] += 1

        else: #remove componenets from inventory as well as any effects they had
            
            if(item == 'UR MUM'):
                print("INVENTORY")
                print(new_inventory)

            for component_name, component_count in recipe.items():
                new_inventory[component_name] += -component_count
                
                component_effect = self.item_effects[component_name] #remove component_effects
                if component_effect != 'None' :
                    for component_effect_name, component_effect_count in component_effect.items():
                        new_effects[component_effect_name] += -component_effect_count
            

            new_inventory[item] += 1

            if(item == 'UR MUM'):
                print(new_inventory)
        
    
        if effect != 'None':

            if(item == 'UR MUM'):
                print("EFFECTS")
                print(new_effects)

            for effect_name, effect_count in effect.items():
                new_effects[effect_name] += effect_count
            
            if(item == 'UR MUM'):   
                print(new_effects)
                    
        

        if(item == 'UR MUM'):
            print("INVENTORY")
            print(self.item_names)
            print(old_inventory)
            print(new_inventory)

            print("EFFECTS")
            print(self.item_names)
            print(old_effects)
            print(new_effects)
        
        return new_inventory, new_effects, new_pollution

    def create_item_factory(self, item):
        print("creating item factory")
        cost_coefficient = 10
        name = item + " Factory"
        recipe = {}
        print(self.item_recipe[item])
        for componenet_name, component_value in self.item_recipe[item].items():
            recipe[componenet_name] = cost_coefficient * component_value
        pollution = cost_coefficient * self.item_pollution[item]
        item_effect = self.item_effects[item]
        effect = [item_effect[0], cost_coefficient * item_effect[1]]
        item_factory = {
            name:{
                'Recipe' : recipe,
                'Pollution' : pollution,
                'Effect' : effect
            }
        }
        self.item_names.append(name)
        self.item_recipe[name] = recipe
        self.item_pollution[name] = pollution
        self.item_effects[name] = effect
        #print(item_factory)
        return item_factory
