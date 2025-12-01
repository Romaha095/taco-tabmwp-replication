# 📘 TABMWP Training Pipeline

This repository provides a full workflow for downloading, preprocessing, and training a two-stage TAPEX-based model on the **TABMWP** dataset:

- **Stage 1:** Chain-of-Thought (CoT) generation.
- **Stage 2:** Final answer prediction.

---

## 1. Environment Setup

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate.bat
```

---

## 2. Installation & Data Preparation

### 2.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 2.2 Download raw TABMWP data

```bash
python src/data/download_tabmwp.py
```

This downloads

- `problems_train.json`
- `problems_dev.json`
- `problems_test.json`

into

```text
data/raw/tabmwp/
```

### 2.3 Create the processed dataset

Normalize and merge the raw data:

```bash
python src/data/prepare_tabmwp.py
```

Output:

```text
data/processed/tabmwp.jsonl
```

### 2.4 Build Stage 1 dataset (CoT supervision)

```bash
python src/data/build_stage1.py
```

Output:

```text
data/stage1/
```

### 2.5 Build Stage 2 dataset (answer supervision)

```bash
python src/data/build_stage2.py
```

Output:

```text
data/stage2/
```

---

## 3. Model Training

Training is controlled via JSON config files in `configs/`.  
The configs specify the TAPEX backbone (`model_name`), learning rate, batch size, number of epochs, output directory, etc.

In our experiments we mainly used:

- `microsoft/tapex-base`
- `microsoft/tapex-large`
- `microsoft/tapex-base-finetuned-wtq`
- `microsoft/tapex-large-finetuned-wtq`

with the following number of epochs:

- **Stage 1:** base – 20 epochs, large – 25 epochs.
- **Stage 2:** base – 15 epochs, large – 20 epochs.

### 3.1 Stage 1 – Chain-of-Thought model

```bash
python train_stage1.py --config configs/stage1_tapex_base.json
```

Change the config file to switch between different TAPEX variants or hyperparameters.

### 3.2 Stage 2 – Answer model

```bash
python train_stage2.py --config configs/stage2_tapex_base.json
```

Again, only the config file changes between base/large and WTQ-finetuned variants.  
The Stage 2 architecture itself is the same; the difference between “with calculator” and “without calculator” appears only when we build the Stage 2 dataset from Stage 1 outputs (see below), not in the training script.

---

## 4. Experiments

This section describes how to reproduce the experiments we ran for the project.

We use two evaluation settings:

1. **Pretrained (QT → A)** – directly evaluate logical TAPEX models on TABMWP in *Question + Table → Answer* format.
2. **Finetuned (TaCo)** – full two-stage pipeline: Stage 1 CoT generation, optional external calculator, Stage 2 answer model.

Paths such as checkpoints and result filenames can be adapted to your own setup; the commands below show a concrete, reproducible layout.

---

### 4.1 Pretrained (QT → A format)

Script: `src/eval/eval_pretrained_tapex_qta.py`.

This evaluates one or more pretrained TAPEX models directly on `data/raw/tabmwp/problems_test.json` and saves JSON files with predictions and detailed metrics.

#### Evaluate all four logical TAPEX models

```bash
python -m src.eval.eval_pretrained_tapex_qta \
  --data_file data/raw/tabmwp/problems_test.json \
  --output_dir results/pretrained_qta \
  --models tapex-base tapex-large tapex-base-finetuned-wtq tapex-large-finetuned-wtq \
  --split test \
  --batch_size 8
```

#### Example: evaluate a single model

```bash
python -m src.eval.eval_pretrained_tapex_qta \
  --data_file data/raw/tabmwp/problems_test.json \
  --output_dir results/pretrained_qta \
  --models tapex-large-finetuned-wtq \
  --split test \
  --batch_size 8
```

This will create files such as

```text
results/pretrained_qta/tapex-base_qt2a_test.json
results/pretrained_qta/tapex-large_qt2a_test.json
...
```

each containing the per-problem predictions needed for the official `evaluate_acc` metrics.

---

### 4.2 Finetuned (TaCo pipeline)

In the TaCo setting we always follow the same two-step procedure on the TABMWP **test** split:

1. **Build a Stage 2 dataset from Stage 1 outputs**  
   (with or without the external calculator).
2. **Evaluate a trained Stage 2 checkpoint** on that dataset.

#### 4.2.1 Step 1 – Build Stage 2 dataset from Stage 1 outputs

Script: `src/data/build_stage2_from_stage1.py`.

We always read problems from

```text
data/raw/tabmwp/problems_test.json
```

and call the script as a module:

```bash
python -m src.data.build_stage2_from_stage1 \
  --stage1_checkpoint_dir <STAGE1_CHECKPOINT_DIR> \
  --data_file data/raw/tabmwp/problems_test.json \
  --output_dir <STAGE2_DATASET_DIR> \
  --batch_size 8 [--use_calculator]
```

- Add `--use_calculator` to enable the external arithmetic calculator when post-processing Stage 1 CoTs.
- Omit `--use_calculator` to build a dataset without the calculator.

Once the Stage 2 dataset is built, you can evaluate a Stage 2 checkpoint with:

```bash
python -m src.eval.eval_answers \
  --checkpoint_dir <STAGE2_CHECKPOINT_DIR> \
  --dataset_path <STAGE2_DATASET_DIR> \
  --split test \
  --data_file data/raw/tabmwp/problems_test.json \
  --output_file <RESULT_JSON_PATH> \
  --batch_size 8
```

- `<STAGE2_CHECKPOINT_DIR>` is the directory with the trained Stage 2 model.
- `<STAGE2_DATASET_DIR>` must match the `--output_dir` used when calling `build_stage2_from_stage1`.
- `<RESULT_JSON_PATH>` is an arbitrary JSON file where predictions and correctness flags will be stored.
