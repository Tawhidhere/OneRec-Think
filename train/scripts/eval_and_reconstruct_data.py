#!/usr/bin/env python3

from transformers import AutoTokenizer, HfArgumentParser
import pandas as pd
from vllm import LLM
from vllm.sampling_params import SamplingParams, BeamSearchParams
from dataclasses import dataclass, field
from typing import Optional
import re
from tqdm import tqdm
import math
import os
from pathlib import Path

@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to the trained model checkpoint"}
    )
    epoch: int = field(
        metadata={"help": "Epoch of the trained model"}
    )
    matchine: int = field(
        metadata={"help": "Number of machines to use for train"}
    )
    gpus: int = field(
        metadata={"help": "Number of GPUs to use for train"}
    )
    bs_per_gpu: int = field(
        metadata={"help": "Batch size per GPU"}
    )
    data_path: str = field(
        metadata={"help": "Path to the training data to be evaluated"}
    )
    config_name: str = field(
        metadata={"help": "Name of the config"}
    )
    sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to process from the dataset for debugging"}
    )
    tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "Number of GPUs to use for tensor parallelism"}
    )

def extract_sid(text: str) -> Optional[str]:
    match = re.search(r"(<\|sid_begin\|>(?:<s_[a-d]_\d+>){4}<\|sid_end\|>)", text)
    if match:
        return match.group(1)
    return None

def main():
    parser = HfArgumentParser((EvalArguments,))
    eval_args = parser.parse_args_into_dataclasses()[0]
    model_name_or_path = eval_args.model_name_or_path
    print(f"Loading model from: {model_name_or_path} using vLLM")
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=eval_args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    # Sampling parameters for generating the <think> block
    generate_n = 2
    sampling_params_think = SamplingParams(
        n=generate_n,  # Generate two versions
        temperature=0.8,
        top_p=0.95,
        top_k=200,
        max_tokens=100,  # Enough for <think> block
        stop=["<|sid_begin|>"]
    )
    beam_sampling_params = BeamSearchParams(
        beam_width=5,
        max_tokens=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading dataset from: {eval_args.data_path}")
    data_pq = pd.read_parquet(eval_args.data_path)

    if eval_args.sample_size:
        print(f"Sampling {eval_args.sample_size} records for evaluation.")
        data_pq = data_pq.head(eval_args.sample_size)

    
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."

    prompts = []
    print("Preparing initial prompts...")
    for _, row in data_pq.iterrows():
        prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{row['description']}<|im_end|>
<|im_start|>assistant
"""
        prompts.append(prompt)

    print("Step 1: Generating <think> blocks (n=2 per prompt)...")
    think_outputs = llm.generate(prompts, sampling_params_think)

    sid_prompts = []
    print("Preparing prompts for SID generation...")
    for i, output in enumerate(think_outputs):
        original_prompt = output.prompt
        for completion in output.outputs:
            # The prompt for SID generation includes the generated <think> block
            # and the stop token. vLLM doesn't include the stop token in the output.
            new_prompt = original_prompt + completion.text + "<|sid_begin|>"
            sid_prompts.append({'prompt': new_prompt})
    # print("sid_prompts: ", sid_prompts)

    print("Step 2: Generating SIDs with beam search (n=5 per prompt)...")
    sid_outputs = llm.beam_search(prompts=sid_prompts,
        params=beam_sampling_params)
    
    reconstructed_data = []
    match_count = 0
    total_count = len(data_pq)

    print("Reconstructing data...")
    for i in tqdm(range(total_count), desc="Processing results"):
        row = data_pq.iloc[i]
        ground_truth_sid = extract_sid(row['groundtruth'])

        # Collect all 10 SID predictions for this sample
        all_predicted_sids = []
        
        # Collect SIDs from all generated beams
        for j in range(generate_n):
            output = sid_outputs[i * generate_n + j]
            for completion in output.sequences:
                tokens = completion.tokens[-4:]
                text = tokenizer.decode(tokens)
                predicted_sid_text = "<|sid_begin|>" + text + '<|sid_end|>'
                extracted_sid = extract_sid(predicted_sid_text)
                if extracted_sid:
                    all_predicted_sids.append(extracted_sid)
                if(i<10):
                    print(f'For index {i}, {j}-th cot is {completion.text}, decoded text is {text}')
        # print("--------------------------------")
        modified_row = row.to_dict()

        if ground_truth_sid and ground_truth_sid in all_predicted_sids:
            match_count += 1
            if(i<10):
                print(f"Match: text in index {i} is {think_outputs[i].prompt}, ground truth sid is {ground_truth_sid}, predicted sid is {all_predicted_sids}")
        else:
            modified_row['title'] = None
            modified_row['categories'] = None
            if(i<10):
                print(f"Match fail: text in index {i} is {think_outputs[i].prompt}, ground truth sid is {ground_truth_sid}, predicted sid is {all_predicted_sids}")

        reconstructed_data.append(modified_row)

    print(f"\nEvaluation finished.")
    print(f"Total rows processed: {total_count}")
    print(f"Matching rows: {match_count}")
    print(f"Match rate: {match_count / total_count:.2%}")

    new_df = pd.DataFrame(reconstructed_data)
    config_name = eval_args.config_name
    repeat_train_output_path = f'./results/{config_name}'
    os.makedirs(Path(repeat_train_output_path)/f"epoch_{eval_args.epoch}", exist_ok=True)
    output_path = f'{repeat_train_output_path}/epoch_{eval_args.epoch}/reconstructed_data.parquet'
    print(f"Saving reconstructed data to: {output_path}")
    new_df.to_parquet(output_path, index=False)
    print("Reconstruction complete.")


if __name__ == "__main__":
    main()
