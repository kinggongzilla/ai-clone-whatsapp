# Create an AI clone of yourself from your WhatsApp chats (using Mistral 7B)

## About
This repository lets you create an AI chatbot clone of yourself, using your WhatsApp chats as training data. The default model used is **Mistral-7B-Instruct-v0.2**. The code in this repository heavily builds upon llama-recipes (https://github.com/facebookresearch/llama-recipes), where you can find more examples on different things to do with llama models.

This repository includes code to:
* Preprocess exported WhatsApp chats into a suitable format for finetuning
* Finetune a model on your WhatsApp chats, using 4-bit quantized LoRa
* Chat with your finetuned AI clone, via a commandline interface

## Setup
1. Clone this repository
2. To install the required dependencies, run the following commands from inside the cloned repository:

```bash
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

## Obtaining and preprocessing your WhatsApp chats
To prepare your WhatsApp chats for training, follow these steps:

1. Export your WhatsApp chats as .txt files. This can be done directly in the WhatsApp app on your phone, for each chat individually. You can export just one .txt from a single chat or many .txt files from all your chats.
2. Copy the .txt files you exported into ```data/preprocessing/raw_chats/train```. If you want to have a validation set, copy the validation chats into ```data/preprocessing/raw_chats/validation```.
3. Run ```python data/preprocessing/preprocess.py```. This will convert your raw chats into a format suitable for training and save CSV files to ```data/preprocessing/processed_chats```

## Start finetune/training
Run ```python -m finetuning --dataset "custom_dataset" --custom_dataset.file "scripts/custom_dataset.py" --whatsapp_username "[insert whatsapp_username]"```. Where you replace```[insert whatsapp_username]``` with your name, as it appears in the exported .txt files from WhatsApp. This is necessary to assign the correct role of "assistant" and "user" for training.

This will first download the base model from Huggingface, if necessary, and then start a LoRa finetune with 4-bit quantization.

The config for the training can be set in ```configs/training.py```. In particular, you can enable evaluation on the validation set after each epoch by setting ```run_validation: bool=True``` 

## Chatting with your AI clone
After successful finetuning, run ```python3 commandline_chatbot.py --peft_model [insert checkpoint folder] --model_name mistralai/Mistral-7B-Instruct-v0.2```, where ```[insert checkpoint folder]``` should be replaced with the output directory you specified in ```configs/training.py```. Default is ```checkpoints```folder.

Running this command loads the finetuned model and let's you have a conversation with it in the commandline.

You can define your own system prompt by changing the ```start_prompt_english``` prompt text in the ```commandline_chatbot.py``` file.

## Hardware requirements
At least 22GB vRAM required for Mistral 7B finetune. I ran the finetune on a RTX 3090.
When experimenting with other models, vRAM requirement might vary.
vRAM requirements could probably be significantly reduced with some optimizations (maybe unsloth.ai).

## Common Issues / Notes
* If your custom_dataset.py script is not finishing or throws a recursion error, you may need to increase the maximum number of recursions in ```train.py```. Please edit ```sys.setrecursionlimit(50000)```.
* The system language when exporting chats should probably be English. Some data cleaning steps depend on this.
* Finetuning works best with English chats. If your chats are in another language, you may need to adjust the preprocessing and training parameters accordingly.
* This code should also work with other models than Mistral 7B, but I haven’t tried it myself. If you want to experiment with different models, you can find them here.
* In my finetunes, I have not exported group chats from WhatsApp. I am unsure if this would also work or cause any errors.
