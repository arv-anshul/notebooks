# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scikit-learn",
#     "torch",
#     "tqdm",
#     "transformers",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

"""
Finetune BERT model for sentiment analysis on Reddit comments.

- Loaded data using pandas.
- Tokenized and converted to a custom `torch.utils.data.Dataset` class.
- Used `problem_type='single_label_classification'` in model config.
- Trained model using with manual for-loop approach.

> This is working fine but there is a better way to do this, by using `transformers.Trainer` class.
"""

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    get_scheduler,
)

CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 212  # word length of 98%tile comments' is ~200
EPOCHS = 3
BATCH_SIZE = 42
LABELS = ("negative", "neutral", "positive")
MODEL_DIR = Path("./model_output")

# Get device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
tokenizer = BertTokenizerFast.from_pretrained(
    CHECKPOINT,
    clean_up_tokenization_spaces=True,
    strip_accents=True,
    tokenize_chinese_chars=False,
)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    _punc = r"[\"'-\+\/\-\&\(\)\[\]\{\}]"  # custom punctuations to remove from comments
    df = df.assign(
        comment=lambda x: x.comment.str.strip().str.replace(_punc, "", regex=True),
    )
    df = df[df["comment"].str.split(n=5).str.len() > 3]
    return df


# Load the dataset
df = (
    pd.read_csv(
        "https://raw.githubusercontent.com/campusx-team/Text-Datasets/refs/heads/main/Reddit_Data.csv",
    )
    .rename(columns={"clean_comment": "comment", "category": "labels"})
    .assign(
        labels=lambda x: x["labels"].map({-1: 0, 0: 1, 1: 2}),
    )
    .dropna()
    .pipe(preprocess_df)
    .sample(100)
)


class CustomDataset(Dataset):
    """Used to read the updated dataframe and tokenize the text."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: BertTokenizerFast,
        max_len: int,
    ):
        # reset_index to make __getitem__ work
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data["comment"][index])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            max_length=self.max_len,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(
                self.data["labels"][index],
                dtype=torch.int,
            ),
        }

    def __len__(self):
        return len(self.data)


# Prepare datasets
train_dataset, test_dataset = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["labels"],
)
train_dataset, val_dataset = train_test_split(
    train_dataset,
    test_size=0.2,
    random_state=42,
    stratify=train_dataset["labels"],
)

train_dataset = CustomDataset(train_dataset, tokenizer, MAX_LENGTH)
val_dataset = CustomDataset(val_dataset, tokenizer, MAX_LENGTH)
test_dataset = CustomDataset(test_dataset, tokenizer, MAX_LENGTH)

# Prepare data loaders
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=MAX_LENGTH,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    shuffle=True,
    drop_last=True,
    pin_memory=torch.cuda.is_available(),
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    shuffle=False,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    shuffle=False,
)

# Load BERT model
model_config = BertConfig(
    problem_type="single_label_classification",
    max_position_embeddings=MAX_LENGTH,
    label2id=dict(zip(LABELS, range(len(LABELS)), strict=True)),
    id2label=dict(enumerate(LABELS)),
    # this verify whether the dataset has the same number of labels
    num_labels=len(df["labels"].unique()),
)
model = BertForSequenceClassification.from_pretrained(
    CHECKPOINT,
    config=model_config,
    ignore_mismatched_sizes=True,
)
model = model.to(device)  # type: ignore

# Set up optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * EPOCHS,
)
loss_fn = torch.nn.CrossEntropyLoss().to(device)


def train_model(model: PreTrainedModel, train_dataloader: DataLoader):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for b in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in b.items()}
        outputs = model(**batch)
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    print(
        f"Training Loss: {(train_loss / len(train_dataloader)):.4f}, "
        f"Training Accuracy: {(correct / total):.4f}",
    )


def validate_model(model: PreTrainedModel, val_dataloader: DataLoader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for b in tqdm(val_dataloader):
            batch = {k: v.to(device) for k, v in b.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])

            val_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    print(
        f"Validation Loss: {(val_loss / len(val_dataloader)):.4f}, "
        f"Validation Accuracy: {(correct / total):.4f}",
    )


def training_loop(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """Training loop to train and validate the model"""
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        train_model(model, train_dataloader)
        validate_model(model, val_dataloader)


def test_model(model: PreTrainedModel, test_dataloader: DataLoader):
    """Testing the model"""
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for b in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in b.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    test_accuracy = correct / total
    print(
        f"Test loss: {test_loss / len(test_dataloader):.4f}, Accuracy: {test_accuracy:.4f}",
    )


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"Model saved to {MODEL_DIR}")


training_loop(model, train_dataloader, val_dataloader)
