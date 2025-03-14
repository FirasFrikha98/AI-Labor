import os
import json
import torch
import torch
torch.set_num_threads(40)
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_scheduler
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dirs = {
    "first": "../training_data/3x/first",
    "middle": "../training_data/3x/middle",
    "last": "../training_data/3x/last",
}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TweetDataset(Dataset):
    def __init__(self, data, label_map, max_length=128):
        self.data = data
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = tokenizer(
            item["tweet"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        rationale_tokens = [0] * len(encoded["input_ids"].squeeze(0))

        rationale_words = (
            " ".join(item["rationale"]).split()
            if isinstance(item["rationale"], list)
            else item["rationale"].split()
        )

        for word in rationale_words:
            for idx, token in enumerate(tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))):
                if word.lower() in token.lower():
                    rationale_tokens[idx] = 1

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label_map[item["label"]], dtype=torch.long),
            "rationale": torch.tensor(rationale_tokens, dtype=torch.float),
        }


class BERT2BERT(nn.Module):
    def __init__(self, num_classes, rationale_weight=0.5):
        super(BERT2BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.rationale_decoder = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.rationale_weight = rationale_weight

    def forward(self, input_ids, attention_mask, labels=None, rationale_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.pooler_output
        token_embeddings = outputs.last_hidden_state

        class_logits = self.classifier(cls_output)
        rationale_logits = self.rationale_decoder(token_embeddings).squeeze(-1)
        rationale_probs = self.sigmoid(rationale_logits)

        loss = None
        if labels is not None and rationale_labels is not None:
            classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
            rationale_loss = nn.BCELoss()(rationale_probs, rationale_labels)
            loss = classification_loss + self.rationale_weight * rationale_loss

        return class_logits, rationale_probs, loss


def evaluate_model(model, val_loader, device, label_map):
    all_preds, all_labels = [], []
    all_rationale_preds, all_rationale_labels = [], []

    model.eval()
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            rationale_labels = batch["rationale"].to(device)

            class_logits, rationale_probs, _ = model(input_ids, attention_mask)

            preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_rationale_preds.append(rationale_probs.cpu().numpy())
            all_rationale_labels.append(rationale_labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluation progress: Batch {batch_idx + 1}/{len(val_loader)}")

    all_rationale_preds = np.concatenate(all_rationale_preds, axis=0)
    all_rationale_labels = np.concatenate(all_rationale_labels, axis=0)

    precision, recall, thresholds = precision_recall_curve(
        all_rationale_labels.flatten(), all_rationale_preds.flatten()
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    rationale_preds = (all_rationale_preds > optimal_threshold).astype(int)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    rationale_f1 = f1_score(
        all_rationale_labels.flatten(),
        rationale_preds.flatten(),
        average="macro"
    )

    cm = confusion_matrix(all_labels, all_preds)
    cm_labels = list(label_map.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Evaluation completed.")
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "rationale_f1": rationale_f1,
        "confusion_matrix": cm
    }

device = torch.device("cpu")
results = []

for split_name, data_dir in data_dirs.items():
    print(f"\nProcessing split: {split_name}")

    with open(os.path.join(data_dir, "training.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, "validation.json"), "r") as f:
        val_data = json.load(f)
    train_data = [entry for entry in train_data if isinstance(entry["tweet"], str) and entry["tweet"].strip()]
    val_data = [entry for entry in val_data if isinstance(entry["tweet"], str) and entry["tweet"].strip()]

    print(f"Filtered training samples: {len(train_data)}")
    print(f"Filtered validation samples: {len(val_data)}")

    all_labels = {item["label"] for item in train_data + val_data}
    label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
    print(f"Label map: {label_map}")

    train_dataset = TweetDataset(train_data, label_map)
    val_dataset = TweetDataset(val_data, label_map)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8, pin_memory=True)



    model = BERT2BERT(num_classes=len(label_map), rationale_weight=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 5)

    print("Starting training...")
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            rationale_labels = batch["rationale"].to(device)

            optimizer.zero_grad()
            _, _, loss = model(input_ids, attention_mask, labels, rationale_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    print(f"Evaluating model for split: {split_name}")
    metrics = evaluate_model(model, val_loader, device, label_map)
    results.append(metrics)

avg_results = {key: np.mean([res[key] for res in results if key in res]) for key in results[0]}
print("\n=== Cross-Validation Results ===")
for metric, value in avg_results.items():
    if metric != "confusion_matrix":  
        print(f"{metric.capitalize()}: {value:.4f}")
