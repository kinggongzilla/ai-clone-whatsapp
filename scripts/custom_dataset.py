# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools
import sys
from configs.training import train_config as TRAIN_CONFIG

#numnber of max recursions set to be larger than dataset size (necessary for the forward function)
sys.setrecursionlimit(50000)

B_INST, E_INST = "[INST]", "[/INST]"

train_config = TRAIN_CONFIG()

def tokenize_dialog(dialog, tokenizer):
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip() if prompt['content'] != None else ''} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('csv', data_dir=train_config.data_dir)
    dataset = dataset[split]

    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset.features),)
    

    nodes = {}

    messages = {}
    root_ids = []

    for data in dataset:
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]

    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            new_threads += follow(thread, nodes[current_id][0])
            return new_threads
        else:
            return [thread]

    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads

    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)


    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            sender, text = extract_sender_and_text(content)
            dialog.append({
                "role": "assistant" if sender == train_config.whatsapp_username else "user",
                "content": content,
            })
        return {"dialog": dialog}

    # Define a function that takes a string as input
    def extract_sender_and_text(string):
        # Find the index of the first [sender] tag
        start = string.find("<sender>")
        # Find the index of the first [/sender] tag
        end = string.find("</sender>")
        # Check if both tags are present
        if start != -1 and end != -1:
            # Extract the sender name between the tags
            sender = string[start + 8 : end]
            # Extract the text after the [/sender] tag
            text = string[end + 9 :]
            # Return the sender name and the text as a tuple
            return (sender, text)
        else:
            print('XXXXXXXXX')
            print(string)
            # Return None if the tags are missing or invalid
            return None

    dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset