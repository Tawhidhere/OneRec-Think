# OneRec-Think

The emergence of large language models (LLMs) has transformed recommendation paradigms from conventional matching to generative frameworks. Although prior research has successfully formulated recommendations as end-to-end generative tasks, these methods typically function as direct predictors without incorporating explicit reasoning mechanisms.

To bridge this gap, we propose **OneRec-Think**, a unified framework that seamlessly integrates dialogue, reasoning, and personalized recommendation. By generating high-quality reasoning paths, our model not only improves recommendation precision but also maintains its native conversational ability.

![OneRec-Think pipeline](png/OneRec-Think.png)

The framework consists of three components:
1.  **Itemic Alignment**, which projects itemic tokens into the LLM's textual space to establish semantic grounding.
2.  **Reasoning Activation**, which constructs simple yet useful chain-of-thought (CoT) fine-tuning examples to stimulate reasoning capabilities within the recommendation context.
3.  **Reasoning Enhancement**, where we design a recommendation-specific reward function that accounts for the multi-validity nature of user preferences.

We validate our model's effectiveness on multiple public datasets, with its deployment on an industrial-scale short-video platform yielding a further online gain of **0.159% in APP Stay Time**. Additionally, we conduct extensive case studies that provide qualitative evidence for the role of reasoning in recommendation.


## Getting Started

Run the environment setup script before proceeding:
```bash
bash setup_conda_env.sh
```

### 1. Obtain the Base Model
- **Download Qwen3-1.7B from Hugging Face**  
```bash
cd basemodel
python3 download_basemodel.py
```
The model is saved under `basemodel/Qwen3-1-7B/`.

- **Extend the vocabulary to support SID tokens**  
```bash
python3 expand_vocab.py
```
The script reads `basemodel/Qwen3-1-7B/` and writes the extended model to `basemodel/Qwen3-1-7B-expand/`.

### 2. Generate Alignment Training Data
```bash
cd data
python3 generate_training_data.py
```
The script consumes `data/sequential_data_processed.txt` and `data/Beauty.pretrain.json`, producing train/validation/test parquet files (`training_data_train.parquet`, `training_data_val.parquet`, `training_data_test.parquet`) for the alignment stage.

### 3. Run Itemic Alignment Fine-tuning and Merge
- **Launch the alignment stage**
```bash
cd train
bash run_training_stage1.sh
```
This launches the LoRA-based alignment training with the parquet files generated above; adjust the script variables as needed for your environment.

- **Merge the best LoRA checkpoint into the expanded base model**
```bash
cd basemodel
python3 merge_model.py
```
Edit `lora_model_path` inside `basemodel/merge_model.py` so it targets the checkpoint you want to merge. The script combines the LoRA weights with `basemodel/Qwen3-1-7B-expand/` and saves the full model to `basemodel/merged_beauty_model_1-1/`.

### 4. Prepare Recommendation Training Corpora
- **Generate SID-only recommendation data**
```bash
cd data
python3 generate_sid_prediction_data.py
python3 generate_RA_data.py
```
These scripts consume the sequential data and Beauty metadata, producing `training_prediction_sid_data_{train,val,test}.parquet` for recommendation training and `training_RA_{train,val,test}.parquet` for the reasoning activation stage.

### 5. Run the Combined Training Pipeline (Recommendation + CoT)
```bash
cd train
bash run_training_stage2.sh
```
This helper first executes the recommendation training (via `scripts/run_training_rec.sh`), waits for it to finish, captures the latest `checkpoint-*` under `results/beauty_sid_rec/`, and then launches the reasoning activation training (via `scripts/run_training_RA.sh`) with that checkpoint. After it completes, you will have both the recommendation checkpoints and the CoT-enhanced checkpoints under `results/ReasoningActivation/`.

### 6. (Optional) Train the Recommendation Model Only
```bash
cd train
bash scripts/run_training_rec.sh
```
Ensure `MODEL_DIR` points to `basemodel/merged_beauty_model_1-1/` (or your merged output) and that the train/val parquet paths reference the freshly generated SID prediction files. The script writes checkpoints to `train/results/beauty_sid_rec/`. (Skip this step if you already ran the combined pipeline above.)

### 7. (Optional) Train the Reasoning Activation (CoT) Model Separately
- **Manual two-step execution**
  1. Identify the best recommendation checkpoint (e.g., `train/results/beauty_sid_rec/checkpoint-XXXX`).
  2. Pass that directory to the RA trainer:
     ```bash
     cd train
     bash scripts/run_training_RA.sh /path/to/beauty_sid_rec/checkpoint-XXXX
     ```

### 8. Evaluate the Models
- **Direct recommendation model (no CoT)**
```bash
cd test
bash eval_parallel_8gpu.sh
```
Update `MERGED_MODEL_PATH` and `TEST_PARQUET` to target the checkpoint produced by the recommendation training (either from the combined pipeline or the standalone run).

- **CoT-enhanced reasoning model**
```bash
cd test
bash eval_parallel_8gpu_cot.sh
```
Point `MERGED_MODEL_PATH` to the directory output by the reasoning activation training (typically under `train/results/ReasoningActivation/epoch_*/`; generated automatically when running the combined pipeline). This script evaluates the CoT-first, then recommendation pipeline.
