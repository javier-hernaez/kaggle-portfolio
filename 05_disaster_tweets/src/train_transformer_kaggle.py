"""
Pipeline de Fine-Tuning de Transformers para Kaggle / Colab (GPU).
Modelo recomendado: microsoft/deberta-v3-small o microsoft/deberta-v3-base.

Este script está listo para ser ejecutado directamente en un Notebook de Kaggle
con acelerador GPU (T4 o P100 gratuito) para alcanzar puntuaciones competitivas (>0.83 F1).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Instalar dependencias si corre en entorno Kaggle/Colab
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
except ImportError:
    print("[!] Este script requiere 'torch', 'transformers', 'datasets' y acelerador GPU.")
    print("    Ideal para subir a Kaggle Notebooks o Google Colab.")

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
NFOLDS = 5
SEED = 42

def prepare_data(df, is_train=True):
    """Combina keyword y texto para que el Transformer preste atención a ambos campos."""
    texts = []
    for _, row in df.iterrows():
        kw = str(row['keyword']) if pd.notna(row['keyword']) else ""
        text = str(row['text'])
        if kw and kw != "missing":
            full = f"Keyword: {kw}. Tweet: {text}"
        else:
            full = f"Tweet: {text}"
        texts.append(full)
    df['full_text'] = texts
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits[:, 1]))
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.2, 0.8, 0.02):
        preds = (probs >= th).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th
    return {"f1": best_f1, "best_threshold": best_th}

def run_deberta_pipeline(data_dir="./data", output_dir="./output_deberta"):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_df = prepare_data(train_df, is_train=True)
    test_df = prepare_data(test_df, is_train=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    
    test_dataset = Dataset.from_pandas(test_df[['full_text']])
    def tokenize_fn(batch):
        return tokenizer(batch['full_text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    
    test_tokenized = test_dataset.map(tokenize_fn, batched=True)
    oof_probs = np.zeros(len(train_df))
    test_probs = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target']), 1):
        print(f"\n--- Entrenando Fold {fold}/{NFOLDS} ---")
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]

        train_ds = Dataset.from_pandas(train_fold[['full_text', 'target']]).rename_column('target', 'label').map(tokenize_fn, batched=True)
        val_ds = Dataset.from_pandas(val_fold[['full_text', 'target']]).rename_column('target', 'label').map(tokenize_fn, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/fold_{fold}",
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        val_preds = trainer.predict(val_ds)
        val_p = 1 / (1 + np.exp(-val_preds.predictions[:, 1]))
        oof_probs[val_idx] = val_p

        test_p = trainer.predict(test_tokenized)
        test_probs += (1 / (1 + np.exp(-test_p.predictions[:, 1]))) / NFOLDS

    # Umbral final
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.2, 0.8, 0.01):
        sc = f1_score(train_df['target'], (oof_probs >= th).astype(int))
        if sc > best_f1:
            best_f1, best_th = sc, th

    print(f"\n[OK] DeBERTa-v3 OOF F1-Score: {best_f1:.5f} (Umbral: {best_th:.3f})")
    
    sub = pd.DataFrame({'id': test_df['id'], 'target': (test_probs >= best_th).astype(int)})
    sub.to_csv("submission_deberta.csv", index=False)
    print("[OK] Generado 'submission_deberta.csv'!")

if __name__ == "__main__":
    if "torch" in sys.modules and torch.cuda.is_available():
        run_deberta_pipeline()
    else:
        print("[!] Para ejecutar DeBERTa-v3 se recomienda GPU en Kaggle Notebooks.")
