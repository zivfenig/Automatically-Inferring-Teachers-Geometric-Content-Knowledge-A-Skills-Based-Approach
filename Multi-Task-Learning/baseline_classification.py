# ============================================================================
# Gemma 3 4B Baseline - Van Hiele Level Classification (No Skills)
# ============================================================================
# Clean classification task without auxiliary indicator predictions
# ============================================================================

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from typing import Dict
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

import torch.nn as nn

# ============================================================================
# IMPORT CONFIGURATION
# ============================================================================
from config_baseline import *

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB\n")

# ============================================================================
# LOAD DATA
# ============================================================================

train_df = pd.read_csv(FOLD_TRAIN_CSV)
test_df = pd.read_csv(FOLD_TEST_CSV)

print(f"\n=== Data Loading ===")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].astype(int)

# Split train into train/validation
train_df, val_df = train_test_split(
    train_df,
    test_size=VALIDATION_SPLIT,
    random_state=SEED,
    stratify=train_df[LABEL_COL]
)
print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}\n")

# Map labels 1..5 -> 0..4
unique_levels = sorted(train_df[LABEL_COL].unique())
level_to_id = {lvl: i for i, lvl in enumerate(unique_levels)}
id_to_level = {i: lvl for lvl, i in level_to_id.items()}
num_labels = len(level_to_id)

train_df["label"] = train_df[LABEL_COL].map(level_to_id)
val_df["label"] = val_df[LABEL_COL].map(level_to_id)
test_df["label"] = test_df[LABEL_COL].map(level_to_id)

print(f"✅ Van Hiele Levels: {unique_levels}")
print(f"   Mapping: {level_to_id}\n")

# ============================================================================
# BUILD TEXT INPUTS (CLEAN: Q+A ONLY)
# ============================================================================

def build_text(row):
    q = str(row[QUESTION_COL]).strip()
    a = str(row[ANSWER_COL]).strip()
    prompt = (
        "אתה מומחה בניתוח הנמקה גיאומטרית בהתאם לתורת ואן היל.\n\n"
        f"שאלה:\n{q}\n\n"
        f"תשובה:\n{a}"
    )
    return prompt

print("📝 Building text inputs...")
tqdm.pandas(desc="Building text")
train_df["text"] = train_df.progress_apply(build_text, axis=1)
val_df["text"] = val_df.progress_apply(build_text, axis=1)
test_df["text"] = test_df.progress_apply(build_text, axis=1)
print("✅ Text inputs built\n")

# ============================================================================
# TOKENIZATION
# ============================================================================

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = MAX_SEQUENCE_LENGTH

def tokenize_batch(batch):
    tokens = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokens["labels"] = batch["labels"]
    return tokens

train_ds = Dataset.from_pandas(train_df[["text", "label"]])
val_ds = Dataset.from_pandas(val_df[["text", "label"]])
test_ds = Dataset.from_pandas(test_df[["text", "label"]])

train_ds = train_ds.map(tokenize_batch, batched=True, desc="Tokenizing train")
val_ds = val_ds.map(tokenize_batch, batched=True, desc="Tokenizing validation")
test_ds = test_ds.map(tokenize_batch, batched=True, desc="Tokenizing test")

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"✅ Tokenized: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}\n")

# ============================================================================
# BASELINE MODEL: CLASSIFICATION ONLY
# ============================================================================

class Gemma4BForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int, lora_r: int):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Classification head
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use last token hidden state
        last_hidden = outputs.hidden_states[-1]
        last_token_hidden = last_hidden[:, -1, :]
        
        logits = self.classifier(last_token_hidden)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return type('Output', (), {'loss': loss, 'logits': logits})()

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    preds_levels = [id_to_level[int(p)] for p in preds]
    labels_levels = [id_to_level[int(l)] for l in labels]
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    mae = mean_absolute_error(labels_levels, preds_levels)
    qwk = cohen_kappa_score(labels_levels, preds_levels, weights="quadratic")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mae": mae,
        "qwk": qwk,
    }

# ============================================================================
# TRAINING
# ============================================================================

print("🏗️  Creating baseline model...")
model = Gemma4BForSequenceClassification(BASE_MODEL, num_labels=num_labels, lora_r=LORA_RANK)
model.to(device)
print("✅ Model created\n")

training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    save_total_limit=3,
    seed=SEED,
    save_safetensors=SAVE_SAFETENSORS,
    bf16=USE_BF16,
    max_grad_norm=MAX_GRAD_NORM,
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    dataloader_pin_memory=DATALOADER_PIN_MEMORY,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("="*80)
print("🚀 TRAINING: Baseline Van Hiele Classification (No Skills)")
print("="*80)
print(f"Fold ID: {FOLD_ID}")
print(f"Model: {BASE_MODEL}")
print(f"Task: Van Hiele Level Classification ({num_labels} classes)")
print(f"Input: Question + Answer only")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"LoRA Dropout: {LORA_DROPOUT}")
print(f"Head Dropout: {HEAD_DROPOUT}")
print(f"Output Folder: {FOLD_OUTPUT_DIR}")
print("="*80 + "\n")

resume_from_checkpoint = get_last_checkpoint(str(CHECKPOINTS_DIR))
if resume_from_checkpoint:
    print(f"✅ Resuming from checkpoint: {resume_from_checkpoint}\n")

train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\n=== Saving Model ===")
model.base_model.save_pretrained(str(MODEL_DIR))
torch.save(model.classifier.state_dict(), MODEL_DIR / "classifier.pt")
tokenizer.save_pretrained(str(MODEL_DIR))

level_to_id_json = {str(int(k)): int(v) for k, v in level_to_id.items()}
id_to_level_json = {str(int(k)): int(v) for k, v in id_to_level.items()}

label_map_path = MODEL_DIR / "label_mapping.json"
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "level_to_id": level_to_id_json,
            "id_to_level": id_to_level_json,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

config_info = {
    "model_name": "Gemma4BForSequenceClassification",
    "base_model": BASE_MODEL,
    "model_size": "4B",
    "lora_rank": LORA_RANK,
    "lora_dropout": LORA_DROPOUT,
    "head_dropout": HEAD_DROPOUT,
    "task": "Van Hiele Level Classification",
    "input_type": "Question + Answer only",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "max_seq_length": MAX_SEQUENCE_LENGTH,
    "num_epochs": NUM_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "validation_split": VALIDATION_SPLIT,
    "fold_id": FOLD_ID,
}

config_path = MODEL_DIR / "model_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_info, f, ensure_ascii=False, indent=2)

print(f"✅ Saved to: {MODEL_DIR}\n")

# ============================================================================
# EVALUATE
# ============================================================================

print(f"=== Validation Metrics (Fold {FOLD_ID}) ===")
val_metrics = trainer.evaluate(val_ds)
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

val_metrics_path = PREDICTIONS_DIR / "val_metrics.json"
with open(val_metrics_path, "w", encoding="utf-8") as f:
    json.dump(val_metrics, f, indent=2)

print(f"\n=== Test Metrics (Fold {FOLD_ID}) ===")
test_metrics = trainer.evaluate(test_ds)
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

test_metrics_path = PREDICTIONS_DIR / "test_metrics.json"
with open(test_metrics_path, "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)

# ============================================================================
# PREDICTIONS
# ============================================================================

print("\n=== Generating Predictions ===")
pred_output = trainer.predict(test_ds)
logits = pred_output.predictions
pred_ids = np.argmax(logits, axis=-1)
gold_ids = pred_output.label_ids

pred_levels = [id_to_level[int(p)] for p in pred_ids]
gold_levels = [id_to_level[int(g)] for g in gold_ids]

test_df["pred_level"] = pred_levels
test_df["gold_level"] = gold_levels

pred_csv = PREDICTIONS_DIR / f"fold_{FOLD_ID}_predictions.csv"
test_df.to_csv(pred_csv, index=False, encoding="utf-8")
print(f"✅ Predictions saved to: {pred_csv}")