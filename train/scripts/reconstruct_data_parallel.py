#!/usr/bin/env python3
"""
Parallel data reconstruction by pre-sharding the dataset
Each GPU processes a separate parquet file shard
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_gpu(args_tuple):
    """Run eval_and_reconstruct_data.py on a single GPU with a pre-sharded dataset"""
    gpu_id, model_path, shard_path, config_name, epoch, bs_per_gpu, output_file, log_file = args_tuple

    # Set environment for this process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['TOKENIZERS_PARALLELISM'] = 'false'

    cmd = [
        'python3', '-u', './scripts/eval_and_reconstruct_data.py',
        '--model_name_or_path', model_path,
        '--data_path', shard_path,
        '--config_name', f"{config_name}_shard{gpu_id}",
        '--epoch', str(epoch),
        '--matchine', '1',
        '--gpus', '1',
        '--bs_per_gpu', str(bs_per_gpu),
        '--tensor_parallel_size', '1',
    ]

    print(f"[GPU {gpu_id}] Starting with shard: {shard_path}")

    try:
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=True
            )
        print(f"[GPU {gpu_id}] Completed successfully")
        return gpu_id, True, None, output_file
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}"
        print(f"[GPU {gpu_id}] Failed: {error_msg}")
        return gpu_id, False, error_msg, None
    except Exception as e:
        error_msg = str(e)
        print(f"[GPU {gpu_id}] Exception: {error_msg}")
        return gpu_id, False, error_msg, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('data_path', type=str, help='Path to original training data')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('epoch', type=int, help='Current epoch number')
    parser.add_argument('num_gpus', type=int, help='Number of GPUs to use')
    parser.add_argument('bs_per_gpu', type=int, help='Batch size per GPU')
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Parallel Data Reconstruction")
    print(f"=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"GPUs: {args.num_gpus}")
    print(f"=" * 60)

    # Load and shard the dataset
    print(f"\nLoading dataset from: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    total_samples = len(df)
    print(f"Total samples: {total_samples}")

    samples_per_gpu = math.ceil(total_samples / args.num_gpus)
    print(f"Samples per GPU: {samples_per_gpu}")

    # Create temp directory for shards
    temp_dir = Path(f"./results/{args.config_name}/epoch_{args.epoch}/temp_shards")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create data shards
    print(f"\nCreating {args.num_gpus} data shards...")
    shard_paths = []
    for i in range(args.num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = min((i + 1) * samples_per_gpu, total_samples)

        shard_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        shard_path = temp_dir / f"input_shard_{i}.parquet"
        shard_df.to_parquet(shard_path, index=False)
        shard_paths.append(str(shard_path))

        print(f"  Shard {i}: rows {start_idx}-{end_idx} ({len(shard_df)} samples) -> {shard_path.name}")

    # Prepare tasks for all GPUs
    tasks = []
    for i in range(args.num_gpus):
        output_file = f"./results/{args.config_name}_shard{i}/epoch_{args.epoch}/reconstructed_data.parquet"
        log_file = str(temp_dir / f"log_gpu_{i}.txt")

        tasks.append((
            i,  # gpu_id
            args.model_path,
            shard_paths[i],
            args.config_name,
            args.epoch,
            args.bs_per_gpu,
            output_file,
            log_file
        ))

    # Launch all tasks in parallel
    print(f"\nLaunching {args.num_gpus} parallel GPU processes...")
    print(f"This may take several minutes depending on dataset size...\n")

    failed_gpus = []
    output_files = []

    with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {executor.submit(run_single_gpu, task): task[0] for task in tasks}

        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                gpu_id, success, error, output_file = future.result()
                if success:
                    output_files.append((gpu_id, output_file))
                else:
                    failed_gpus.append(gpu_id)
                    print(f"[GPU {gpu_id}] Check log: {temp_dir}/log_gpu_{gpu_id}.txt")
            except Exception as e:
                print(f"[GPU {gpu_id}] Unexpected error: {e}")
                failed_gpus.append(gpu_id)

    # Check results
    if failed_gpus:
        print(f"\n❌ Failed GPUs: {failed_gpus}")
        print(f"Check logs in {temp_dir}/")
        sys.exit(1)

    print(f"\n✅ All {args.num_gpus} processes completed successfully!")
    print("\nMerging reconstructed shard files...")

    # Collect all reconstructed files
    output_files.sort(key=lambda x: x[0])  # Sort by GPU ID
    all_data = []

    for gpu_id, output_file in output_files:
        output_path = Path(output_file)
        if output_path.exists():
            shard_df = pd.read_parquet(output_path)
            all_data.append(shard_df)
            print(f"  GPU {gpu_id}: {len(shard_df)} rows from {output_path}")
        else:
            print(f"  ⚠️  GPU {gpu_id}: Output file not found: {output_path}")

    if not all_data:
        print("❌ No output files found!")
        sys.exit(1)

    # Merge all shards
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal merged rows: {len(merged_df)}")

    # Save final reconstructed data
    final_output_path = f"./results/{args.config_name}/epoch_{args.epoch}/reconstructed_data.parquet"
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    merged_df.to_parquet(final_output_path, index=False)
    print(f"✅ Saved merged data to: {final_output_path}")

    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)

    # Remove shard output directories
    for gpu_id in range(args.num_gpus):
        shard_dir = Path(f"./results/{args.config_name}_shard{gpu_id}")
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

    print("✅ Parallel reconstruction complete!")

if __name__ == "__main__":
    main()
