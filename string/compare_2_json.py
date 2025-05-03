import json
import os

def compare_json_files(file_path_1, file_path_2):
    # Load the JSON data from both files
    with open(file_path_1, 'r', encoding='utf-8') as file1, open(file_path_2, 'r', encoding='utf-8') as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    # Check if the lengths of the JSON arrays are the same
    if len(data1) != len(data2):
        print(f"The lengths of the JSON arrays are different between {file_path_1} and {file_path_2}.")
        return None

    # Initialize a counter for the number of matching strings
    match_count = 0

    # Iterate over the JSON arrays and compare the strings
    for sublist1, sublist2 in zip(data1, data2):
        # Assuming each sublist contains exactly one string
        if sublist1[0] == sublist2[0]:
            match_count += 1

    # Calculate the proportion of matching strings
    match_ratio = match_count / len(data1)
    return match_ratio

# Example inputs list
inputs = ["string/1107.json","string/GLM_result.json", "string/1114.json","string/GLM_no_newline.json","string/openai.json"]

# Extract file names and perform pairwise comparisons
for i in range(len(inputs)):
    for j in range(i + 1, len(inputs)):
        file1 = inputs[i]
        file2 = inputs[j]
        filename1 = os.path.basename(file1).split('.')[0]
        filename2 = os.path.basename(file2).split('.')[0]
        
        ratio = compare_json_files(file1, file2)
        
        if ratio is not None:
            print(f"Matching ratio between {filename1} and {filename2}: {ratio:.2f}")
