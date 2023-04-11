import torch
import tensorflow as tf
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import random

path = 'data/simulation_4/'

def load_data(path):
    inventory = pd.read_csv(path + 'inventory.csv')
    reward = pd.read_csv(path + 'reward.csv')
    pollution = pd.read_csv(path + 'pollution.csv')

    states = inventory
    # select columns to merge into a single array
    cols = ['paperclip', 'steel', 'wood', 'axe', 'pickaxe', 'miner', 'chopper', 'paperclip_factory']

    # create a new column "inventory" containing an array of the selected columns
    states['inventory'] = states[cols].to_numpy().tolist()

    # drop the selected columns since they are already included in the "inventory" column
    states = states.drop(cols, axis=1)

    # Merge data frames based on episode and steps columns
    states = pd.merge(states, reward, on=['episode', 'step']).merge(pollution, on=['episode', 'step'])
    return states

# Print the merged data frame

states = load_data(path=path)

states_as_text = []

for row in states.iterrows():
    string = f"Action:{row[1]['action']}, Inventory:{str(row[1]['inventory'])}, Pollution:{row[1]['pollution']}, Reward:{row[1]['reward']}"
    states_as_text.append(string)

random.shuffle(states_as_text)  # randomise order for testing and training

# Load the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Set the padding token

input_ids = []
attention_masks = []
for state in states_as_text:
    encoded_dict = tokenizer.encode_plus(state, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the input_ids and attention_masks to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)


#TRYING TO EVEN LOAD THE MODEL

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text using the model
generated_text = model.generate(
    input_ids=input_ids,
    attention_mask=attention_masks,
    max_length=1024,
    do_sample=True
)

# Decode the generated text
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

print(decoded_text)
