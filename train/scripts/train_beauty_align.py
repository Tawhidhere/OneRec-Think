#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import TrainableTokensConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


@dataclass
class BeautyScriptArguments:
    model_dir: str = "../basemodel/Qwen3-1-7B-expand"
    train_data_path: str = "../data/training_data_train.parquet"
    val_data_path: str = "../data/training_data_val.parquet"


def prepare_dataset(data_path, sample_size=None, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")

    if sample_size is not None and len(data_pq) > sample_size:
        if local_rank == 0:
            print(f"Sampling {sample_size} samples from {len(data_pq)} total samples")
        data_pq = data_pq.head(sample_size)

    texts = data_pq['description'].tolist()
    if local_rank == 0:
        print(f"Total texts: {len(texts)}")

        print("\nFirst 3 text examples (full text):")
        for i, text in enumerate(texts[:3]):
            print(f"  [{i}]: {text}")
    
    dataset_dict = {
        'text': texts
    }
    return Dataset.from_dict(dataset_dict)

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        padding='longest',
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return tokenized

def get_special_tokens():
    special_tokens = []

    special_tokens.append('<|sid_begin|>')
    special_tokens.append('<|sid_end|>')

    max_range = 256
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(max_range):
            special_tokens.append(f'<{prefix}_{i}>')
    
    return special_tokens

if __name__ == "__main__":
    parser = HfArgumentParser((BeautyScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]

    model_dir = Path(script_args.model_dir).resolve()
    train_data_path = Path(script_args.train_data_path).resolve()
    val_data_path = Path(script_args.val_data_path).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    if not val_data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")

    if training_args.local_rank == 0:
        print(f"Debug: eval_strategy = {training_args.eval_strategy}")
        print(f"Debug: save_strategy = {training_args.save_strategy}")
        print(f"Debug: metric_for_best_model = {training_args.metric_for_best_model}")
        print(f"Debug: greater_is_better = {training_args.greater_is_better}")
        print(f"Debug: load_best_model_at_end = {training_args.load_best_model_at_end}")
        print(f"Debug: early stopping patience = 2")
        print(f"Using model_dir: {model_dir}")
        print(f"Training data path: {train_data_path}")
        print(f"Validation data path: {val_data_path}")

    print(f"Loading model from: {model_dir}")

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    special_tokens = get_special_tokens()
    print(f"Total special tokens: {len(special_tokens)}")

    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)

    valid_special_token_ids = []
    valid_special_tokens = []
    for i, token_id in enumerate(tokenized_special_tokens):
        if token_id != tokenizer.unk_token_id:
            valid_special_token_ids.append(token_id)
            valid_special_tokens.append(special_tokens[i])
    
    print(f"Valid special tokens: {len(valid_special_token_ids)}")
    print(f"First 10 valid special tokens: {valid_special_tokens[:10]}")
    print(f"Training token IDs range: {min(valid_special_token_ids)} to {max(valid_special_token_ids)}")

    lora_config = TrainableTokensConfig(
        token_indices=valid_special_token_ids,
        target_modules=["embed_tokens"],
        init_weights=True
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if training_args.local_rank == 0:
        print("\\nTrainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")

    if training_args.local_rank == 0:
        print("\\nLoading training dataset...")

    train_dataset = prepare_dataset(train_data_path, local_rank=training_args.local_rank)
    if training_args.local_rank == 0:
        print(f"Loaded raw train dataset, total samples: {len(train_dataset)}")

    if training_args.local_rank == 0:
        print("\\nLoading validation dataset...")
    val_dataset = prepare_dataset(val_data_path, local_rank=training_args.local_rank)
    if training_args.local_rank == 0:
        print(f"Loaded raw validation dataset, total samples: {len(val_dataset)}")

    if training_args.local_rank == 0:
        print("Tokenizing training dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    if training_args.local_rank == 0:
        print(f"Tokenized train dataset, total samples: {len(train_dataset)}")

    if training_args.local_rank == 0:
        print("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )
    if training_args.local_rank == 0:
        print(f"Tokenized validation dataset, total samples: {len(val_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if training_args.local_rank == 0:
        print(f"\\nTrainer eval_strategy: {trainer.args.eval_strategy}")
        print(f"Trainer has eval_dataset: {trainer.eval_dataset is not None}")
        print(f"Eval dataset size: {len(trainer.eval_dataset) if trainer.eval_dataset else 0}")
    
    if training_args.local_rank == 0:
        print("\\nStarting training...")
    trainer.train()

    if training_args.local_rank == 0:
        print("\\nFinal evaluation...")
    result = trainer.evaluate()
    if training_args.local_rank == 0:
        print("Final evaluation result:")
        print(result)

    if training_args.local_rank == 0:
        print("\\nSaving model...")
    output_dir = training_args.output_dir
    trainer.save_model(output_dir)
    if training_args.local_rank == 0:
        print(f"Model saved to: {output_dir}")
        print("Training completed!")
