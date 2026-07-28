# Adapted from LLaVA, Copyright 2023 Haotian Liu, Apache-2.0.
"""Minimal multimodal tokenization helper adapted from LLaVA."""

import torch

from .constants import IMAGE_TOKEN_INDEX


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    """Replace each ``<image>`` placeholder with the image-token sentinel."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(chunks, separator):
        paired = zip(chunks, [separator] * len(chunks))
        return [element for pair in paired for element in pair][:-1]

    input_ids = []
    offset = 0
    if (
        prompt_chunks
        and prompt_chunks[0]
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    separator = [image_token_index] * (offset + 1)
    for chunk in insert_separator(prompt_chunks, separator):
        input_ids.extend(chunk[offset:])

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")
