from torchtune.data import  ChatFormat, Llama2ChatFormat, Message

from torchtune.datasets import ChatDataset
from torchtune.modules.tokenizers import Tokenizer


def custom_dataset(
    tokenizer: Tokenizer,
    source: str = "data/preprocessed",
    train_on_input: bool = False,
) -> ChatDataset:
    """
    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.

    Returns:
        InstructDataset: dataset configured with source data and template


    Example:
        >>> dataset = custom_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(dataset, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        train_on_input=train_on_input,
        split="train",
    )
