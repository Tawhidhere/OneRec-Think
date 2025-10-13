#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_special_tokens(max_range: int = 256) -> list[str]:
    special_tokens: list[str] = []

    special_tokens.append("<|sid_begin|>")
    special_tokens.append("<|sid_end|>")

    for prefix in ["s_a", "s_b", "s_c", "s_d"]:
        for idx in range(max_range):
            special_tokens.append(f"<{prefix}_{idx}>")

    return special_tokens


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return ((value + multiple - 1) // multiple) * multiple


def expand_vocabulary(
    base_model_dir: Path,
    save_dir: Path,
) -> None:
    print(f"Loading model config from: {base_model_dir}")
    config = AutoConfig.from_pretrained(base_model_dir)

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    if device == "cuda":
        model = model.to("cuda")
    device_for_encoding = device

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    new_tokens = get_special_tokens(max_range=256)
    print(f"Preparing to add {len(new_tokens)} special tokens.")

    tokens_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}, replace_additional_special_tokens=False
    )
    print(f"Successfully added {tokens_added} tokens.")

    updated_vocab_size = len(tokenizer)
    target_vocab_size = round_up_to_multiple(updated_vocab_size, 256)

    print(
        f"Current vocab size: {updated_vocab_size}, adjusting to: {target_vocab_size} (nearest 256 multiple)."
    )
    model.resize_token_embeddings(target_vocab_size)

    config.vocab_size = target_vocab_size

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving expanded model to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)

    sample_text = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    sample_ids = tokenizer.encode(sample_text, return_tensors="pt").to(device_for_encoding)
    print(f"Sample tokens encoded shape: {sample_ids.shape}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    base_model_dir = base_dir / "Qwen3-1-7B"
    save_dir = base_dir / "Qwen3-1-7B-expand"

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    expand_vocabulary(
        base_model_dir=base_model_dir,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
