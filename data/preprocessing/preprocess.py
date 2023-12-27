# Import the modules
import csv
import glob
import os
import random
import re

# Define a function to generate a random message id
def generate_id():
  # Use a combination of letters and digits
  chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  # Return a string of 8 random characters
  return "".join(random.choice(chars) for _ in range(8))

# Define a function to parse a message from a line of text
def parse_message(line):
  # Use a regular expression to extract the date, time, sender and text
  pattern = r"\[(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)"
  match = re.match(pattern, line)
  # Return a dictionary with the extracted fields
  if match:
    return {
      "date": match.group(1),
      "timestamp": match.group(2),
      "sender": match.group(3),
      "text": '<sender>' + match.group(3) + '</sender>' + match.group(4)
    }
  else:
    return None

# Define a function to convert a txt file to a csv file
def txt_to_csv(txt_path, csv_writer, parent_id):
  # Open the txt file for reading
  with open(txt_path, "r") as txt_file:

    # Initialize the message to None
    message = None
    # Initialize the flag to False
    is_message = False
    # Loop through the lines of the txt file
    for i, line in enumerate(txt_file):
      # Strip the trailing newline character
      line = line.strip()
      # If the line is empty, skip it
      if not line:
        continue

      #skip whatsapp systm message in each thread; skip "ommitted image/video message"; skip 
      if (line[0] == "[" and i == 0) or "omitted" in line:
        continue

      # If the line starts with a [, it is a new message
      if line[0] == "[":
        # If there is a previous message, and the parent id is not None, write it to the csv file
        if message:
          # Generate a message id
          message_id = generate_id()
          # Write the message to the csv file
          csv_writer.writerow([message_id, parent_id, message["text"], message["date"], message["timestamp"], message["sender"]])
          # Update the parent id to the current message id
          parent_id = message_id
        # Parse the message from the line
        message = parse_message(line)
        # Set the flag to True
        is_message = True
      # If the line does not start with a [, it is a continuation of the previous message
      else:
        # If the flag is True, append the line to the text field of the message
        if is_message:
          message["text"] += "\n" + line
        # Otherwise, ignore the line
        else:
          pass
    # If there is a remaining message, and the parent id is not None, write it to the csv file
    if message and "omitted" not in message["text"]:
      # Generate a message id
      message_id = generate_id()
      # Write the message to the csv file
      csv_writer.writerow([message_id, parent_id, message["text"], message["date"], message["timestamp"], message["sender"]])
  # Return the last message id
  return message_id

# Define a function to convert a folder of txt files to a csv file
def folder_to_csv(folder_path, csv_path):
  # Get the list of txt files in the folder
  txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
  # Sort the txt files by name
  txt_files.sort()
  # Open the csv file for writing
  with open(csv_path, "w") as csv_file:
    # Create a csv writer object
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(["message_id", "parent_id", "text", "date", "timestamp", "sender"])
    # Initialize the parent id and the previous txt file to None
    parent_id = None
    prev_txt_file = None
    # Loop through the txt files
    for txt_file in txt_files:
      # If the txt file name is different from the previous one, reset the parent id to None
      if txt_file != prev_txt_file:
        parent_id = None

      #skip chat files that have only 1 message
      with open(txt_file, "r") as f:
        lines = f.readlines()
        if len(lines) <= 2:
          continue

      # Convert the txt file to a csv file
      parent_id = txt_to_csv(txt_file, csv_writer, parent_id)
      # Update the previous txt file to the current one
      prev_txt_file = txt_file


#if name main
if __name__ == "__main__":

  #get commandline argument data_folder
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_folder", type=str, default="data/preprocessing/raw_chats")
  parser.add_argument("--output_folder", type=str, default="data/preprocessing/processed_chats")
  args = parser.parse_args()

  # Convert the txt files to a csv file
  folder_to_csv(f"{args.data_folder}/train", f"{args.output_folder}/train/train_chats.csv")
  folder_to_csv(f"{args.data_folder}/validation", f"{args.output_folder}/validation/validation_chats.csv")