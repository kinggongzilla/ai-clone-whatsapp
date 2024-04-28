# Run with python preprocess.py --whatsapp_name "YOUR NAME"
# Input data is expected to be in data/raw_data directory
# Input directory contains .txt files exported directly from WhatsApp
# The expected format is given below. Your format may vary from region to region and you may have to
# adjust the regex pattern in the code below.

#EXPECTED FORMAT FOR REGEX PATTERN MATCHING
# [23.07.23, 10:17:53] ‪+43 12345 566789‬: ‎Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
# [23.07.23, 10:17:53] ‪+43 12345 566789‬: Hi whatsup
# [23.07.23, 10:31:45] YOUR NAME: Hey, I'm good
# [23.07.23, 10:17:53] ‪+43 12345 566789‬: I'm going to the movies tonight
# [23.07.23, 10:31:45] YOUR NAME: Cool! I'll join you

import argparse
import json
import re
import os

def txt_to_json(txt_file_path, json_file_path, whatsapp_name):
   
    # Initialize the conversations list with the system message
    conversations = []
    
    # Regular expression pattern to match the chat lines
    pattern = r"\[(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)"
    
    # Open and read the .txt file
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        # Skip the first line
        file.readline()
        
        last_from = None
        combined_value = ""
        
        for line in file:
            # Use the provided regex pattern to match the line
            match = re.match(pattern, line)
            if match:
                # Extract the sender and message content
                sender = match.group(3).strip()
                content = match.group(4).strip()
                
                # Filter out 'Missed video call' messages
                if "Missed video call" in content or "Missed voice call" in content:
                    continue
                
                # Determine the 'from' field based on the sender
                from_field = "gpt" if (whatsapp_name and whatsapp_name in sender) else "human"

                conversations.append({
                            "from": from_field,
                            "value": content
                        })
    
    # Save the conversations list to a .json file
    if(len(conversations) > 40):
        #Conversations that are too long cause errors with torchtune so we split them into multiple files
        for i in range(0, len(conversations), 40):
            partial_conversations = conversations[i:i+40]
            #skip conversations that are too short
            if(len(conversations) < 3):
                return
            json.dump({"conversations": partial_conversations}, open(json_file_path + str(i) + ".json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    else:
        #skip conversations that are too short
        if(len(conversations) < 3):
            return
        json.dump({"conversations": conversations}, open(json_file_path + ".json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

def process_folder(data_folder, output_folder, whatsapp_name):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each .txt file in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(data_folder, filename)
            json_file_name = os.path.splitext(filename)[0]
            json_file_path = os.path.join(output_folder, json_file_name)
            txt_to_json(txt_file_path, json_file_path, whatsapp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WhatsApp chat .txt files to JSON format.")
    parser.add_argument("whatsapp_name", type=str, help="WhatsApp name to identify messages from that sender")
    args = parser.parse_args()

    if not args.whatsapp_name:
        print("Please provide a WhatsApp name to identify messages from that sender.")
        exit()

    print(args.whatsapp_name)

    data_folder = "data/raw_data"
    output_folder = "data/preprocessed"
    process_folder(data_folder, output_folder, args.whatsapp_name)