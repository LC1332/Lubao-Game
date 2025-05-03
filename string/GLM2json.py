# To accomplish the task, I'll write a Python script that reads from a jsonl file, extracts the 'middle' field from each line, 
# and then saves the list of strings in the desired format to a .json file.

# Sample Python code to demonstrate the process

import json

def extract_middle_from_jsonl(jsonl_file_path, output_json_path, keyword = "middle"):
    # Initialize an empty list to store the 'middle' field values
    middle_values = []

    # Open and read the jsonl file
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as JSON
            json_object = json.loads(line)
            
            # Extract the 'middle' field if it exists
            if keyword in json_object:
                result = [json_object[keyword]]

                suffix = json_object.get("fim_suffix", "")

                # 如果suffix的开头的字母之前的空格\t \n中包含\n 且result尾部有\n，则去掉result尾部的\n
                lstrip_suffix = suffix.lstrip()
                strip_out = suffix[-len(lstrip_suffix):]
                if "\n" in strip_out and "\n" in result[-1]:
                    result[-1] = result[-1].rstrip("\n")

                middle_values.append(result)
    
    # Write the list of 'middle' values to a .json file
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(middle_values, output_file, ensure_ascii=False, indent=4)

# Assuming the jsonl file is named 'input.jsonl' and the output file should be 'output.json'
# extract_middle_from_jsonl('input.jsonl', 'output.json')

# Since I can't actually run this on a real file, I'm providing the code as a reference. 
# You can run this code in your local environment by providing the correct file paths.

input_json = "string/GLM_B_result.jsonl"
output_json = "string/GLM_B_no_newline.json"

extract_middle_from_jsonl(input_json, output_json, keyword = "middle_GLM")