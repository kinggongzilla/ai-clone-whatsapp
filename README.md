# Create an AI clone of yourself from your WhatsApp chats (using Llama3)

## About
This repository lets you create an AI chatbot clone of yourself, using your WhatsApp chats as training data. This repository builds upon the new **torchtune** library for finetuning and inference. 

This repository includes code to:
* Preprocess exported WhatsApp chats into a suitable format for finetuning
* Finetune a model on your WhatsApp chats, using 4-bit quantized LoRa
* Chat with your finetuned AI clone, via a commandline interface

Currently supported models are:
* Llama3-8B-Instruct
* Mistral-7B-Instruct-v0.2

## Setup
1. Clone this repository
2. Ensure you have pytorch installed in your active environment. If not, follow these instructions: https://pytorch.org/get-started/locally/ 
3. Install torchtune:

```bash
git clone https://github.com/kinggongzilla/torchtune.git
cd torchtune
pip install .
cd ..
```

Note that slight modifications to the torchtune library ChatDataset class code were necessary were necessary, hence we're not installing from the official repo. In particular the validate_messages function call is removed, to allow for message threads which are not strictly alternating between human and assistant roles.

## Downloading the base model
### Llama3
Run ```tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir model/original/llama3 --hf-token <HF_TOKEN>```. Replace <HF_TOKEN> with your hugging face access token. In order to download Llama3 you first need to request access on the Meta Llama3 Huggingface page.

After downloading the output-dir should contain these files:
* consolidated.00.pth
* params.json
* tokenizer.model

### Mistral 7B Instruct
Run ```tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir model/original/mistral```. 

After downloading the output-dir should contain these files:
* config.json
* generation_config.json
* model-00001-of-00003.bin
* model-00002-of-00003.bin
* model-00003-of-00003.bin
* model.safetensors.index.json
* special_tokens_map.json
* tokenizer_config.json
* tokenizer.json

If you downloaded the model in another format (e.g. safetensors), please adjust the checkpoint_files in ```mistral/qlora_train_config.yaml```.

## Obtaining and preprocessing your WhatsApp chats
To prepare your WhatsApp chats for training, follow these steps:

1. Export your WhatsApp chats as .txt files. This can be done directly in the WhatsApp app on your phone, for each chat individually. You can export just one .txt from a single chat or many .txt files from all your chats.
Unfortunately, formatting seems to vary between regions. I am based on Europe, so the regex in the ```preprocess.py``` might have to be adjusted if you are based in a different region.
2. Copy the .txt files you exported into ```data/raw_data```. 
3. Run ```python preprocess.py "YOUR NAME"```. This will convert your raw chats into a sharegpt format suitable for training and saves the JSON files to ```data/preprocessed```. Replace ```YOUR NAME``` with the exact string which represents your name in the exportet WhatsApp .txt files. The script will assign you the "gpt" role and your conversation partners the "user" role.


## Start finetune/training

### Llama3
Run ```tune run lora_finetune_single_device --config config/llama3/qlora_train_config.yaml```

### Mistral
Run ```tune run lora_finetune_single_device --config config/mistral/qlora_train_config.yaml```


## Chatting with your AI clone
### Llama3
Run ```tune run chat.py --config config/llama3/inference_config.yaml```

You can define your own system prompt by changing the ```prompt``` string in the ```config/llama3/inference_config.py``` file.

### Mistral
For mistral to fit onto 24GB I first hat to quantize the trained model. 

1. Run ```tune run quantize --config config/mistral/quantization.yaml```
2. Run ```tune run chat.py --config config/mistral/inference_config.yaml```

Running this command loads the finetuned model and let's you have a conversation with it in the commandline.

## Hardware requirements
At least 26GB vRAM required for QLoRa Llama3 finetune with 4k context length. I ran the finetune on a RTX 3090.
When experimenting with other models, vRAM requirement might vary.


## FAQ
1. **I trained my model but it did not learn my writing style**
Try training for more than one epoch. You can change this in the ```qlora_train_config.yaml``` file.
2. **The preprocessing script does not work**
You probably need to adjust the regex pattern in ```preprocess.py```. The WhatsApp export format varies from region to region.
3. **I want to train a clone of on group chats**
The current setup does not support group chats. Hence do not export and save them into the ```data/raw_data``` directory. If you do want the model to simulate group chats, I think you have to adjust the preprocessing and ChatDataset of torchtune, such that they support more than 2 roles. I haven't tried this myself.

## Common Issues / Notes
* After training, adjust temperature and top_k parameters in the ```inference_config.yaml``` file. I found a temperature of 0.2 and top_k of 10000 to work well for me. 
* Finetuning works best with English chats. If your chats are in another language, you may need to adjust the preprocessing and training parameters accordingly.