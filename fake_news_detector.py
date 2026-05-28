"""
Fake News Detector using BERT
===============================
Uses a pre-trained BERT model fine-tuned on the Fake News dataset
to classify news articles as REAL or FAKE.

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

RANDOM_SEED   = 42
MAX_LEN       = 512
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
MODEL_NAME    = "bert-base-uncased"
OUTPUT_DIR    = "saved_model"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── 1. Load & Preprocess Dataset ─────────────────────────────────────────────

def load_data(fake_path: str = "Fake.csv", true_path: str = "True.csv") -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path, on_bad_lines='skip', engine='python')
    true_df = pd.read_csv(true_path, on_bad_lines='skip', engine='python')

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].str.strip()
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Dataset loaded: {len(df)} articles")
    print(f"  Fake : {(df['label'] == 0).sum()}")
    print(f"  Real : {(df['label'] == 1).sum()}")
    return df[["content", "label"]]


def explore_data(df: pd.DataFrame) -> None:
    print("\n── Data Overview ──────────────────────────────")
    print(df.head(3).to_string())
    print(f"\nAvg content length: {df['content'].str.split().apply(len).mean():.0f} words")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["label"].value_counts().plot(kind="bar", ax=ax, color=["#E63946", "#457B9D"])
    ax.set_xticklabels(["Fake", "Real"], rotation=0)
    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=150)
    plt.show()
    print("Saved: label_distribution.png")


# ─── 2. PyTorch Dataset ────────────────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── 3. Model ─────────────────────────────────────────────────────────────────

def build_model() -> tuple:
    print(f"\nLoading BERT model: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(DEVICE)
    return tokenizer, model


# ─── 4. Training Loop ─────────────────────────────────────────────────────────

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(data_loader), correct / total


def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            logits  = outputs.logits

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(data_loader), correct / total, all_preds, all_labels


# ─── 5. Visualisation Helpers ─────────────────────────────────────────────────

def plot_training_history(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved: training_history.png")


def plot_confusion_matrix(labels, preds) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


# ─── 6. Inference ─────────────────────────────────────────────────────────────

def predict(text: str, model, tokenizer, device) -> dict:
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    label      = "REAL" if probs[1] > probs[0] else "FAKE"
    confidence = float(max(probs))

    return {
        "label":       label,
        "confidence":  round(confidence * 100, 2),
        "prob_fake":   round(float(probs[0]) * 100, 2),
        "prob_real":   round(float(probs[1]) * 100, 2),
    }


# ─── 7. Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_data("Fake.csv", "True.csv")
    explore_data(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"\nSplits — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    tokenizer, model = build_model()

    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    val_dataset   = FakeNewsDataset(X_val,   y_val,   tokenizer)
    test_dataset  = FakeNewsDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*50}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✔ Best model saved (val_acc={val_acc:.4f})")

    plot_training_history(history)

    print("\n── Test Set Evaluation ────────────────────────────")
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, DEVICE)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Fake", "Real"]))
    plot_confusion_matrix(test_labels, test_preds)

    print("\n── Demo Inference ─────────────────────────────────")
    sample_texts = [
        "Scientists confirm that drinking bleach cures all diseases instantly.",
        "NASA announces successful launch of its new Artemis moon mission.",
    ]
    for text in sample_texts:
        result = predict(text, model, tokenizer, DEVICE)
        print(f"\nArticle  : {text[:80]}...")
        print(f"Prediction: {result['label']} ({result['confidence']}% confident)")
        print(f"  Prob Fake: {result['prob_fake']}% | Prob Real: {result['prob_real']}%")


if __name__ == "__main__":
    main()
