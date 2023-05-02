import os
import csv
def encode_counts(count, encodings, values):
    for i in range(len(values)):
            if count <= values[i]:
                return encodings[i]
    return encodings[-1]
        



encodings = ["miniscule","tiny","small","modest","average", "ample","large","huge","massive"]
paperclip_encode_values    = [80, 100, 120, 140, 160, 180, 200, 220]
pollution_encode_values    = [1600,1700,1800,1900,2000,2100,2200,2300]

paperclip_counts =[]
pollution_counts =[]
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        # Open the CSV file and read in the data
        with open(filename, newline='') as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                # Extract the pollution and paperclip counts
                paperclip_count = float(row[1])
                pollution_count = float(row[2])
                
                # Add the counts to the corresponding list
                paperclip_counts.append(paperclip_count)
                pollution_counts.append(pollution_count)
                

paperclip_counts  = sorted(paperclip_counts)
last_element = ""
for count in paperclip_counts:
    paperclip_count_as_encoded_word = encode_counts(count, encodings, paperclip_encode_values)
    
    if paperclip_count_as_encoded_word != last_element:
        print(paperclip_count_as_encoded_word, " ",count)      
        last_element = paperclip_count_as_encoded_word

print("--------------------------------")
pollution_counts  = sorted(pollution_counts)
last_element = ""
for count in pollution_counts:
    pollution_count_as_encoded_word = encode_counts(count, encodings, pollution_encode_values)
    
    if pollution_count_as_encoded_word != last_element:
        print(pollution_count_as_encoded_word, " ",count)      
        last_element = pollution_count_as_encoded_word
   
