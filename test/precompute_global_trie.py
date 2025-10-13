#!/usr/bin/env python3

import pandas as pd
import pickle
import argparse
import os
import re
from transformers import AutoTokenizer


def extract_all_sids_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    matches = re.findall(sid_pattern, text)
    return matches


def extract_sid_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    match = re.search(sid_pattern, text)
    if match:
        return match.group(0)
    return text.strip()


def build_global_trie(test_parquet_file, model_path, output_file):
    print(f"Loading test data from: {test_parquet_file}")
    df = pd.read_parquet(test_parquet_file)
    print(f"Total samples in test set: {len(df)}")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Extracting all SIDs from test set (description + groundtruth)...")
    valid_sids = set()

    for _, row in df.iterrows():
        description_sids = extract_all_sids_from_text(row['description'])
        for sid in description_sids:
            if sid and '<|sid_begin|>' in sid and '<|sid_end|>' in sid:
                valid_sids.add(sid)

        groundtruth_sid = extract_sid_from_text(row['groundtruth'])
        if groundtruth_sid and '<|sid_begin|>' in groundtruth_sid and '<|sid_end|>' in groundtruth_sid:
            valid_sids.add(groundtruth_sid)
    
    print(f"Found {len(valid_sids)} unique valid SIDs in test set")

    print("Converting SIDs to token sequences...")
    sid_token_sequences = []
    for sid in valid_sids:
        tokens = tokenizer.encode(sid, add_special_tokens=False)
        sid_token_sequences.append(tokens)
    
    print(f"Converted {len(sid_token_sequences)} SIDs to token sequences")

    from collections import defaultdict
    exact_trie = defaultdict(lambda: defaultdict(set))

    max_length = max(len(seq) for seq in sid_token_sequences) if sid_token_sequences else 0
    print(f"Maximum SID token length: {max_length}")

    for seq in sid_token_sequences:
        for pos in range(len(seq)):
            current_token = seq[pos]
            if pos + 1 < len(seq):
                next_token = seq[pos + 1]
                exact_trie[pos][current_token].add(next_token)
            else:
                eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                exact_trie[pos][current_token].add(eos_id)

    final_exact_trie = {}
    for pos in exact_trie:
        final_exact_trie[pos] = {}
        for token_id in exact_trie[pos]:
            final_exact_trie[pos][token_id] = list(exact_trie[pos][token_id])
    
    print(f"Built exact trie tree:")
    print(f"  Total unique SIDs: {len(valid_sids)}")
    print(f"  Search space size: {len(valid_sids):,} (exact match only)")
    print(f"  Trie depth: {max_length}")
    
    for pos in range(min(6, max_length)):
        num_tokens = len(final_exact_trie.get(pos, {}))
        print(f"  Position {pos}: {num_tokens} possible tokens")

    trie_data = {
        'exact_trie': final_exact_trie,
        'valid_sids': valid_sids,
        'valid_sid_tokens': sid_token_sequences,
        'tokenizer_name': model_path,
        'total_samples': len(df),
        'search_space_size': len(valid_sids),
        'max_length': max_length,
        'trie_type': 'exact'
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(trie_data, f)
    
    print(f"Exact trie saved to: {output_file}")
    return trie_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute global trie for parallel evaluation")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Test parquet file")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for tokenizer")
    parser.add_argument("--output_file", type=str, default="./global_trie.pkl", help="Output pickle file")
    
    args = parser.parse_args()
    build_global_trie(args.test_parquet_file, args.model_path, args.output_file)