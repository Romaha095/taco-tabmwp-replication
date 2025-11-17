Inštalácia a príprava dát 
1️⃣ Nainštalujte závislosti
pip install -r requirements.txt

2️⃣ Stiahnite surové dáta TABMWP

Stiahnite súbory:

problems_train.json

problems_dev.json

problems_test.json

a vložte ich sem:

data/raw/tabmwp/

3️⃣ Vytvorte spracovaný dataset
py src/data/prepare_tabmwp.py


Tento skript vytvorí:

data/processed/tabmwp.jsonl

4️⃣ Dataset pre Stage 1 (Chain-of-Thought model)
py src/data/build_stage1.py


Výstup:

data/stage1/

5️⃣ Dataset pre Stage 2 (Answer model)
py src/data/build_stage2.py


Výstup:

data/stage2/

6️⃣ Tréning modelov
Stage 1 – CoT model
python train_stage1.py --config configs/stage1_tapex_large.json

Stage 2 – Answer model
python train_stage2.py --config configs/stage2_tapex_large.json