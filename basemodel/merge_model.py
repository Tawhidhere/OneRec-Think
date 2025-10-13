#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

def merge_and_save_models():
    base_model_path = './Qwen3-1-7B-expand'
    lora_model_path = '../train/results/beauty_align/checkpoint-5250'
    output_path = './merged_beauty_model_1-1'
    
    print("="*80)
    print("MERGING TWO MODELS INTO SINGLE DIRECTORY")
    print("="*80)
    
    if os.path.exists(output_path):
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    try:
        print(f"\n1. Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Base model loaded successfully")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        
        if os.path.exists(lora_model_path):
            print(f"\n2. Loading and merging stage 2 model from: {lora_model_path}")
            stage2_model = PeftModel.from_pretrained(base_model, lora_model_path)
            final_merged_model = stage2_model.merge_and_unload()
            print(f"   Stage 2 model merged successfully")
            
            del base_model, stage2_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"\n2. Stage 2 model path does not exist, using base model")
            final_merged_model = base_model
        
        print(f"\n3. Saving final merged model to: {output_path}")
        final_merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"   ✓ Model saved successfully!")
        
        print(f"\n4. Verifying saved model...")
        saved_files = os.listdir(output_path)
        print(f"   Saved files: {saved_files}")
        
        test_model = AutoModelForCausalLM.from_pretrained(output_path)
        test_tokenizer = AutoTokenizer.from_pretrained(output_path)
        print(f"   ✓ Model verification successful!")
        print(f"   ✓ Model parameters: {test_model.num_parameters():,}")
        print(f"   ✓ Tokenizer vocab size: {test_tokenizer.vocab_size}")
        
        del test_model, test_tokenizer, final_merged_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n" + "="*80)
        print(f"MODEL MERGE COMPLETED SUCCESSFULLY!")
        print(f"Merged model saved to: {output_path}")
        print("="*80)
        
        return output_path
        
    except Exception as e:
        print(f"\nError during model merging: {e}")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        raise

if __name__ == "__main__":
    merge_and_save_models()