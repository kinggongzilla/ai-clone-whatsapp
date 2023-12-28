# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

#python3 -m finetuning --dataset "custom_dataset" --custom_dataset.file "scripts/custom_dataset.py" --whatsapp_username "David Hauser"

import fire
from train import main

if __name__ == "__main__":
    fire.Fire(main)