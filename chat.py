# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig

from torch import nn

from torchtune import config, utils
from torchtune.data._types import Message
from torchtune.data import MistralChatFormat

logger = utils.get_logger("DEBUG")


class ChatRecipe:
    """
    Recipe for chatting via commandline interface.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For a general tutorial for inference with torchtune, please see the
    tutorial at: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def generate(self, cfg: DictConfig, chat_history: List[Message]):

        if cfg.checkpointer.model_type == "MISTRAL":
            chat_history = MistralChatFormat.format(chat_history)

        tokens, _ = self._tokenizer.tokenize_messages(
            chat_history, max_seq_len=cfg.max_new_tokens
        )
        #remove automatically appendend end of text token 128001
        if tokens[-1] == 128001:
            tokens = tokens[:-1] 
        model_input = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                utils.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = utils.generate(
                model=self._model,
                prompt=model_input,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                # eos_id=self._tokenizer.eos_id,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        generated_tokens = utils.generate(
            model=self._model,
            prompt=model_input,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            # eos_id=self._tokenizer.eos_id if cfg.checkpointer.model_type == "MISTRAL" else self._tokenizer.eot_id,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )

        if cfg.checkpointer.model_type == "MISTRAL":
            decoded_tokens = self._tokenizer.decode(generated_tokens)
        else:
            decoded_tokens = self._tokenizer.decode(generated_tokens, truncate_at_eos=False)

        return decoded_tokens
    
    def remove_eot_id(self, s: str):
        eot_id = "<|eot_id|>"
        if s.endswith(eot_id):
            return s[:-len(eot_id)]
        return s


    def chat(self, cfg: DictConfig):
        if cfg.checkpointer.model_type == "MISTRAL":
            chat_history = []
        else:
            chat_history = [Message(
                role="system",
                content=cfg.prompt,
            )]
        # Use a while loop to keep asking the user for input
        while True:
            prompt = input("Enter your prompt: ")
            chat_history.append(Message(role="user", content=prompt))
            output = self.generate(cfg=cfg, chat_history=chat_history)
            if cfg.checkpointer.model_type == "MISTRAL":
                output = output.split('[/INST ')[-1]
            else:
                output = output.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1]
            output = self.remove_eot_id(output)
            print(output)
            chat_history.append(Message(role="assistant", content=output))


@config.parse
def main(cfg: DictConfig) -> None:
    #flush memory
    torch.cuda.empty_cache()
    config.log_config(recipe_name="ChatRecipe", cfg=cfg)
    recipe = ChatRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.chat(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
