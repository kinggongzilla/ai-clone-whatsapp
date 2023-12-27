# Create your AI Clone from your WhatsApp chats with Mistral 7B

## About
This repository lets you create an AI chatbot clone of yourself, using your WhatsApp chats as training data. The default model used is **Mistral-7B-Instruct-v0.2**. The code in this repository heavily builds upon llama-recipes (https://github.com/facebookresearch/llama-recipes), where you can find more examples on different things to do with llama models.

## Hardware requirements
At least 22GB vRAM required for Mistral 7B finetune. I ran the finetune on a RTX 3090.
When experimenting with other models, vRAM requirement might vary.
vRAM requirements could probably be significantly reduced with some optimizations.

## Setup
1. Clone this repository
2. To install the required dependencies, run the following commands from inside the cloned repository:

```bash
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

## Obtaining and preprocessing your WhatsApp chats
To prepare your WhatsApp chats for training, follow these steps:

1. Export your WhatsApp chats as .txt files. This can be done directly in the WhatsApp app on your phone, for each chat individually.
2. Copy the .txt files you exported into ```data/preprocessing/raw_chats/train```. If you want to have a validation set, copy the validation chats into ```data/preprocessing/raw_chats/validation```.
3. Run ```python data/preprocessing/preprocess.py```. This will convert your raw chats into a format suitable for training and save CSV files to ```data/preprocessing/processed_chats```

# Start training
Run ```python -m finetuning --dataset "custom_dataset" --custom_dataset.file "scripts/custom_dataset.py" --whatsapp_username [insert whatsapp_username]```. Where you replace```[insert whatsapp_username]``` with your name, as it appears in the exported .txt files.

This will first download the model from Huggingface and then start a LoRa finetune with 4-bit quantization.

# Chatting with your AI clone
Run ```python3 commandline_chatbot.py --peft_model [insert checkpoint folder] --model_name mistralai/Mistral-7B-Instruct-v0.2```, where ```[insert checkpoint folder]``` should be replaced with the output directory specified in ```configs/training.py```. Default is ````checkpoints```folder.


## Common Issues / Notes
* If your custom_dataset.py script is not finishing or throws a recursion error, you may need to increase the maximum number of recursions in ```train.py```. Please edit ```sys.setrecursionlimit(50000)```.
* The system language when exporting chats should probably be English. Some data cleaning steps depend on this.
* Finetuning works best with English chats. If your chats are in another language, you may need to adjust the preprocessing and training parameters accordingly.
* This code should also work with other models than Mistral 7B, but I haven’t tried it myself. If you want to experiment with different models, you can find them here.