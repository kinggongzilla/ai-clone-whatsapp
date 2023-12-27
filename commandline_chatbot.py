#command to run this:
#python3 commandline_chatbot.py --peft_model checkpoints --model_name mistralai/Mistral-7B-Instruct-v0.2

import fire

import torch
from transformers import AutoTokenizer

from inference.model_utils import load_model, load_peft_model

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=True,
    max_new_tokens =300, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=10, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.05, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.3, # [optional] The value used to modulate the next token probabilities.
    top_k: int=20, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.1, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    max_padding_length: int=0, # the max padding length to be used with tokenizer padding the prompts.
    **kwargs
):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    #this is the starting/system prompt for the chatbot
    start_prompt_english = [
        {"role": "user", "content": "Your name is AI-Bot. An AI chatbot trained on the WhatsApp chats of a magnificent human."},
        {"role": "assistant", "content": "I will follow these commands. I will now start the conversation."},
        ]
    
    # default to german
    prompt = start_prompt_english.copy()

    # Use a while loop to keep asking the user for input
    while True:
        # Prompt the user to enter some text or type 'quit' to exit
        text = input("")
        # Check if the user typed 'quit'
        if text == 'quit':
            # Break out of the loop
            break
        else:
            # Append the text to the list
            prompt.append({
                "role": "user",
                "content": text
            })
            tokenizer_input = tokenizer.apply_chat_template(conversation=prompt, tokenize=False)
            batch = tokenizer(tokenizer_input, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs 
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #print text that comes after the last [/INST]
            reply = output_text.split('[/INST]')[-1]
            print(reply)
            prompt.append({
                "role": "assistant",
                "content": reply
            })

if __name__ == "__main__":
    fire.Fire(main)
