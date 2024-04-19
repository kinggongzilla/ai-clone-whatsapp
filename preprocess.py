import json
import re
import os

def txt_to_json(txt_file_path, json_file_path):
    # Define the system message
    system_message = {
        "role": "system",
        "content": "You are an AI clone of David. You are 28 years old and live in Linz, Austria. You love AI. You study Artificial Intelligence at JKU Linz. You have a 6 year old son called Louis, he loves lego and painting. Your girlfriend is Varisha, she lives in Brighton and works as a vet. Your siblings are Sieglinde, Christoph and Verena."
    }
    
    # Initialize the list with the system message
    messages_list = [system_message]
    
    # Regular expression pattern to match the chat lines
    pattern = r"\[(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)"
    
    # Open and read the .txt file
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        # Skip the first line
        file.readline()
        
        for line in file:
            # Use the provided regex pattern to match the line
            match = re.match(pattern, line)
            if match:
                # Extract the sender and message content
                sender = match.group(3).strip()
                content = match.group(4).strip()
                
                # Determine the role based on the sender
                role = "assistant" if "David Hauser" in sender else "user"
                
                # Append the message to the list
                messages_list.append({
                    "role": role,
                    "content": content
                })
    
    # Save the list of messages to a .json file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(messages_list, json_file, ensure_ascii=False, indent=4)

def process_folder(data_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each .txt file in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(data_folder, filename)
            json_file_name = os.path.splitext(filename)[0] + '.json'
            json_file_path = os.path.join(output_folder, json_file_name)
            txt_to_json(txt_file_path, json_file_path)

#if name main
if __name__ == "__main__":
    data_folder = "data/raw_data"
    output_folder = "data/preprocessed"
    process_folder(data_folder, output_folder)
