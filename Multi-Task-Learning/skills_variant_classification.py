# ============================================================================
# Gemma 3 4B Skills Variant - Van Hiele + Skills Multi-Task Learning
# ============================================================================
# Multi-task learning with indicator attention mechanism
# ============================================================================

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# ============================================================================
# CONFIG: PATHS & HYPERPARAMETERS
# ============================================================================

BASE_MODEL = "google/gemma-3-4b-it"
FOLD_ID = 5
VALIDATION_SPLIT = 0.15
INDICATOR_LOSS_WEIGHT = 0.5
INDICATOR_EMB_DIM = 512

# Embedding model for initializing indicator embeddings
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# Relative paths from repo
DATA_BASE = Path(__file__).parent / ".." / "Data-and-preprocess"
FOLDS_DIR = DATA_BASE / "HE_Van_Hiele_Dataset" / "folds"
FOLD_TRAIN_CSV = FOLDS_DIR / f"fold_{FOLD_ID}_train.csv"
FOLD_TEST_CSV = FOLDS_DIR / f"fold_{FOLD_ID}_test.csv"

# Output directory
RESULTS_BASE = Path(__file__).parent / "results" / "skills_variant"
FOLD_OUTPUT_DIR = RESULTS_BASE / f"fold_{FOLD_ID}"
CHECKPOINTS_DIR = FOLD_OUTPUT_DIR / "checkpoints"
MODEL_DIR = FOLD_OUTPUT_DIR / "model"
PREDICTIONS_DIR = FOLD_OUTPUT_DIR / "predictions"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Column names
QUESTION_COL = "question"
ANSWER_COL = "answer"
LABEL_COL = "final_decision"
INDICATORS_COL = "final_indicators"

# Hyperparameters (consistent across variants)
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4
LORA_DROPOUT = 0.05
HEAD_DROPOUT = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB\n")

# ============================================================================
# LOAD INDICATORS DICTIONARY
# ============================================================================

print("📚 Loading indicator definitions...")
sys.path.insert(0, str(DATA_BASE / "HE_Skills_dictionary"))
from indicators_dictionary import indicators_dict

print(f"✅ Loaded {len(indicators_dict)} indicator definitions\n")

# ============================================================================
# LOAD & ENCODE DEFINITIONS WITH E5
# ============================================================================

print("🔄 Initializing E5 embedding model for definition encoding...")
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
embedding_model.eval()

def encode_text(text):
    """Encode text using E5 model with mean pooling"""
    with torch.no_grad():
        inputs = embedding_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"]
        mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = (embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1)
        mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings[0].cpu().numpy()

print("📝 Encoding indicator definitions...")
indicator_codes_ordered = sorted(indicators_dict.keys())
definition_embeddings_dict = {}
for code in tqdm(indicator_codes_ordered, desc="Encoding definitions"):
    definition_text = indicators_dict[code]
    emb = encode_text(definition_text)
    definition_embeddings_dict[code] = emb

print(f"✅ Encoded {len(definition_embeddings_dict)} indicator definitions ({len(definition_embeddings_dict[indicator_codes_ordered[0]])} dims)\n")

# Free memory
del embedding_model
del embedding_tokenizer
torch.cuda.empty_cache()

# ============================================================================
# LOAD DATA
# ============================================================================

train_df = pd.read_csv(FOLD_TRAIN_CSV)
test_df = pd.read_csv(FOLD_TEST_CSV)

print(f"=== Data Loading ===")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].astype(int)

# Split train into train/validation
train_df, val_df = train_test_split(
    train_df,
    test_size=VALIDATION_SPLIT,
    random_state=42,
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
# PREPARE INDICATORS
# ============================================================================

print("🏷️  Preparing indicators...")

def parse_indicators(indicator_str):
    if pd.isna(indicator_str):
        return []
    return [ind.strip() for ind in str(indicator_str).split(",")]

def create_indicator_labels(ind_list, indicator_to_id, num_indicators):
    labels = np.zeros(num_indicators, dtype=np.float32)
    for ind in ind_list:
        if ind in indicator_to_id:
            labels[indicator_to_id[ind]] = 1.0
    return labels

train_df["indicators_list"] = train_df[INDICATORS_COL].apply(parse_indicators)
val_df["indicators_list"] = val_df[INDICATORS_COL].apply(parse_indicators)
test_df["indicators_list"] = test_df[INDICATORS_COL].apply(parse_indicators)

all_indicators = set()
for ind_list in train_df["indicators_list"]:
    all_indicators.update(ind_list)

indicators_vocab = sorted(list(all_indicators))
indicator_to_id = {ind: i for i, ind in enumerate(indicators_vocab)}
id_to_indicator = {i: ind for ind, i in indicator_to_id.items()}
num_indicators = len(indicators_vocab)

print(f"✅ Found {num_indicators} unique indicators\n")

train_df["indicator_labels"] = train_df["indicators_list"].apply(
    lambda x: create_indicator_labels(x, indicator_to_id, num_indicators)
)
val_df["indicator_labels"] = val_df["indicators_list"].apply(
    lambda x: create_indicator_labels(x, indicator_to_id, num_indicators)
)
test_df["indicator_labels"] = test_df["indicators_list"].apply(
    lambda x: create_indicator_labels(x, indicator_to_id, num_indicators)
)

# ============================================================================
# BUILD TEXT INPUTS
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

max_length = 2048

train_data = {
    "text": train_df["text"].tolist(),
    "labels": train_df["label"].tolist(),
    "indicator_labels": train_df["indicator_labels"].tolist(),
}
val_data = {
    "text": val_df["text"].tolist(),
    "labels": val_df["label"].tolist(),
    "indicator_labels": val_df["indicator_labels"].tolist(),
}
test_data = {
    "text": test_df["text"].tolist(),
    "labels": test_df["label"].tolist(),
    "indicator_labels": test_df["indicator_labels"].tolist(),
}

train_ds = Dataset.from_dict(train_data)
val_ds = Dataset.from_dict(val_data)
test_ds = Dataset.from_dict(test_data)

def tokenize_with_indicators(batch):
    tokens = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokens["labels"] = batch["labels"]
    tokens["indicator_labels"] = batch["indicator_labels"]
    return tokens

print("⏳ Tokenizing...")
train_ds = train_ds.map(tokenize_with_indicators, batched=True, desc="Tokenizing train")
val_ds = val_ds.map(tokenize_with_indicators, batched=True, desc="Tokenizing val")
test_ds = test_ds.map(tokenize_with_indicators, batched=True, desc="Tokenizing test")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "indicator_labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "indicator_labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "indicator_labels"])

print(f"✅ Tokenized: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}\n")

# ============================================================================
# SKILLS VARIANT MODEL: MULTI-TASK WITH INDICATOR ATTENTION
# ============================================================================

class Gemma4BMultiTaskWithIndicatorAttention(nn.Module):
    """
    Multi-task model with indicator embeddings and attention mechanism.
    
    Tasks:
    - Primary: Van Hiele level classification
    - Auxiliary: Skills/indicator multi-label prediction
    """
    
    def __init__(self, model_name: str, num_labels: int, num_indicators: int, 
                 definition_embeddings_dict: dict, indicator_codes_ordered: list, 
                 indicator_to_id: dict, lora_r: int = 16, ind_emb_dim: int = 512):
        super().__init__()
        
        # Load base model with LoRA
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        hidden_size = self.base_model.config.hidden_size
        
        # Indicator embeddings (initialized from definition embeddings)
        self.ind_embed = nn.Embedding(num_indicators, ind_emb_dim)
        
        # Initialize embedding weights from pre-encoded definitions
        if definition_embeddings_dict:
            embedding_weights = []
            for code in indicator_codes_ordered:
                if code in definition_embeddings_dict:
                    emb = torch.tensor(definition_embeddings_dict[code], dtype=torch.float32)
                    if emb.shape[0] != ind_emb_dim:
                        # Project from E5 (768) to ind_emb_dim (512)
                        proj = nn.Linear(emb.shape[0], ind_emb_dim)
                        emb = proj(emb.unsqueeze(0)).squeeze(0)
                    embedding_weights.append(emb)
            if embedding_weights:
                embedding_weights = torch.stack(embedding_weights)
                self.ind_embed.weight.data = embedding_weights
        
        # Attention: Query-based weighting of indicators
        self.attention_query = nn.Linear(hidden_size, ind_emb_dim)
        self.attention_proj = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(ind_emb_dim, ind_emb_dim),
        )
        
        # Projection from indicator embeddings
        self.definition_proj = nn.Linear(ind_emb_dim, hidden_size)
        
        # Classification heads
        self.level_classifier = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(hidden_size * 2, num_labels),
        )
        
        self.indicator_classifier = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(hidden_size + ind_emb_dim, num_indicators),
        )

    def forward(self, input_ids, attention_mask=None, labels=None, indicator_labels=None):
        # Get encoder output
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        last_hidden = outputs.hidden_states[-1]
        last_token_hidden = last_hidden[:, -1, :]
        
        # Indicator attention
        query = self.attention_query(last_token_hidden)
        ind_ids = torch.arange(self.ind_embed.weight.shape[0], device=last_token_hidden.device)
        ind_embeddings = self.ind_embed(ind_ids)
        
        # Attention weights
        attn_logits = torch.matmul(query, ind_embeddings.T)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        # Weighted indicator representation
        ind_context = torch.matmul(attn_weights, ind_embeddings)
        ind_context = self.attention_proj(ind_context)
        
        # Project indicator context to hidden space
        ind_hidden = self.definition_proj(ind_context)
        
        # Level classification
        level_input = torch.cat([last_token_hidden, ind_hidden], dim=-1)
        level_logits = self.level_classifier(level_input)
        
        # Indicator classification
        indicator_input = torch.cat([last_token_hidden, ind_context], dim=-1)
        indicator_logits = self.indicator_classifier(indicator_input)
        
        loss = None
        if labels is not None:
            level_loss = nn.CrossEntropyLoss()(level_logits, labels)
            indicator_loss = nn.BCEWithLogitsLoss()(indicator_logits, indicator_labels)
            loss = level_loss + INDICATOR_LOSS_WEIGHT * indicator_loss
        
        return type('Output', (), {
            'loss': loss,
            'level_logits': level_logits,
            'indicator_logits': indicator_logits,
        })()

# ============================================================================
# CUSTOM TRAINER
# ============================================================================

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        indicator_labels = inputs.pop("indicator_labels")
        outputs = model(**inputs, indicator_labels=indicator_labels)
        
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if isinstance(logits, tuple):
        level_logits = logits[0]
    else:
        level_logits = logits
    
    preds = np.argmax(level_logits, axis=-1)
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

class MinEpochEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, min_epochs=0, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.epoch < self.min_epochs:
            return
        super().on_evaluate(args, state, control, metrics, **kwargs)

# ============================================================================
# TRAINING
# ============================================================================

print("🏗️  Creating skills variant model...")
model = Gemma4BMultiTaskWithIndicatorAttention(
    model_name=BASE_MODEL,
    num_labels=num_labels,
    num_indicators=num_indicators,
    definition_embeddings_dict=definition_embeddings_dict,
    indicator_codes_ordered=indicator_codes_ordered,
    indicator_to_id=indicator_to_id,
    lora_r=16,
    ind_emb_dim=INDICATOR_EMB_DIM,
)
model.to(device)
print("✅ Model created\n")

early_stopping = MinEpochEarlyStoppingCallback(
    min_epochs=0,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.0,
)

training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    save_total_limit=3,
    seed=42,
    save_safetensors=False,
    bf16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    lr_scheduler_type="cosine",
    warmup_steps=100,
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("="*80)
print("🚀 TRAINING: Skills Variant with Indicator Attention")
print("="*80)
print(f"Fold ID: {FOLD_ID}")
print(f"Model: {BASE_MODEL}")
print(f"Main Task: Van Hiele Level Classification ({num_labels} classes)")
print(f"Auxiliary Task: Skills Indicators ({num_indicators} multi-label)")
print(f"Input: Question + Answer")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"LoRA Dropout: {LORA_DROPOUT}")
print(f"Head Dropout: {HEAD_DROPOUT}")
print(f"Indicator Loss Weight: {INDICATOR_LOSS_WEIGHT}")
print(f"Indicator Embedding Dim: {INDICATOR_EMB_DIM}")
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
torch.save(model.level_classifier.state_dict(), MODEL_DIR / "level_classifier.pt")
torch.save(model.indicator_classifier.state_dict(), MODEL_DIR / "indicator_classifier.pt")
torch.save(model.ind_embed.state_dict(), MODEL_DIR / "ind_embed.pt")
torch.save(model.attention_query.state_dict(), MODEL_DIR / "attention_query.pt")
torch.save(model.attention_proj.state_dict(), MODEL_DIR / "attention_proj.pt")
torch.save(model.definition_proj.state_dict(), MODEL_DIR / "definition_proj.pt")
tokenizer.save_pretrained(str(MODEL_DIR))

level_to_id_json = {str(int(k)): int(v) for k, v in level_to_id.items()}
id_to_level_json = {str(int(k)): int(v) for k, v in id_to_level.items()}
indicator_to_id_json = {str(k): int(v) for k, v in indicator_to_id.items()}
id_to_indicator_json = {str(int(k)): str(v) for k, v in id_to_indicator.items()}

label_map_path = MODEL_DIR / "label_mapping.json"
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "level_to_id": level_to_id_json,
            "id_to_level": id_to_level_json,
            "indicator_to_id": indicator_to_id_json,
            "id_to_indicator": id_to_indicator_json,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

config_info = {
    "model_name": "Gemma4BMultiTaskWithIndicatorAttention",
    "base_model": BASE_MODEL,
    "model_size": "4B",
    "lora_rank": 16,
    "lora_dropout": LORA_DROPOUT,
    "head_dropout": HEAD_DROPOUT,
    "main_task": "Van Hiele Level Classification",
    "auxiliary_task": "Indicators Multi-label Prediction",
    "indicator_embedding_dim": INDICATOR_EMB_DIM,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "attention_type": "query-based indicator weighting",
    "indicator_loss_weight": INDICATOR_LOSS_WEIGHT,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": 2,
    "gradient_accumulation_steps": 2,
    "max_seq_length": max_length,
    "num_epochs": NUM_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "validation_split": VALIDATION_SPLIT,
    "fold_id": FOLD_ID,
    "num_levels": num_labels,
    "num_indicators": num_indicators,
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