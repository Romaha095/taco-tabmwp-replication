# 📘 TABMWP Training Pipeline
This repository provides a full workflow for downloading, preprocessing, and training a two-stage TAPEX-based model on the **TABMWP** dataset:

- **Stage 1:** Chain-of-Thought (CoT) generation
- **Stage 2:** Final answer prediction

---

# 🛠️ Environment Setup

## For Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

## For Windows
```bash
python -m venv venv
venv\Scripts\activate.bat
```

---

# 📦 Installation & Data Preparation

## 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2️⃣ Download raw TABMWP data
Run the downloader script:

```bash
python src/data/download_tabmwp.py
```

This will automatically fetch:

- problems_train.json
- problems_dev.json
- problems_test.json

Files will be saved to:

```
data/raw/tabmwp/
```

---

## 3️⃣ Create the processed dataset
Normalize and merge the raw dataset:

```bash
python src/data/prepare_tabmwp.py
```

Output:

```
data/processed/tabmwp.jsonl
```

---

## 4️⃣ Build Stage 1 dataset (Chain-of-Thought)
```bash
python src/data/build_stage1.py
```

Output:

```
data/stage1/
```

---

## 5️⃣ Build Stage 2 dataset (Answer Model)
```bash
python src/data/build_stage2.py
```

Output:

```
data/stage2/
```

---

# 🏋️ Model Training

## Stage 1 – Chain-of-Thought Model
```bash
python train_stage1.py --config configs/stage1_tapex_large.json
```

## Stage 2 – Answer Model
```bash
python train_stage2.py --config configs/stage2_tapex_large.json
```

