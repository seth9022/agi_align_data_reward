import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def load_data(path):
    states = pd.read_csv('data/simulation_3/inventory.csv')
    reward = pd.read_csv('data/simulation_3/reward.csv')
    pollution = pd.read_csv('data/simulation_3/pollution.csv')
    # Merge data frames based on episode and steps columns
    merged_df = pd.merge(states, reward, on=['episode', 'step']).merge(pollution, on=['episode', 'step'])
    print(merged_df)

# Print the merged data frame

load_data("a")
import sys; sys.exit(1)



# Load the pretrained transformer model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the state (inventory, reward, and pollution values)
state = {'inventory': [3, 5, 2, 0, 1, 0, 0, 0, 0], 'reward': 1, 'pollution': -5}

# Convert the state to a formatted string that the transformer can process
formatted_state = f"Inventory: {state['inventory']} Reward: {state['reward']} Pollution: {state['pollution']}"

# Tokenize the formatted state
tokenized_state = tokenizer.encode(formatted_state, return_tensors='pt')

# Evaluate the state with the transformer
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
result = classifier(tokenizer.decode(tokenized_state[0], skip_special_tokens=True))

# Print the results
print(result)