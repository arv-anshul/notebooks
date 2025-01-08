# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "polars",
#     "scikit-learn",
#     "tqdm",
#     "transformers[torch]",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

"""
Finetune BERT model for sentiment analysis on Reddit comments.

- Loaded data using polars.
- Tokenized and converted to `datasets.Dataset` class.
- Used `problem_type='multi_label_classification'` in model config.
- Trained model using `transformers.Trainer` class.

> Is it possible to finetune the model with `problem_type='single_label_classification'` and then
> make predictions with `problem_type='multi_label_classification'`?
>
> If this is the case then it is easier to finetune the model with `problem_type='single_label_classification'`
> and use it further for prediction.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

BATCH_SIZE = 16
CHECKPOINT = "bert-base-uncased"
EPOCHS = 3
LABELS = ("negative", "neutral", "positive")
MAX_LENGTH = 212  # word length of 98%tile comments' is ~200
MODEL_DIR = Path("./model_output")
SEED = 42

# set seed
torch.manual_seed(SEED)

# Get device
DEVICE = (
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


def preprocess_df(df: pl.LazyFrame) -> pl.LazyFrame:
    _punc = r"[\"'-\+\/\-\&\(\)\[\]\{\}]"  # custom punctuations to remove from comments
    df = df.with_columns(
        pl.col("comment").str.replace_all(_punc, ""),
    ).filter(
        pl.col("comment").str.split(" ").list.len().gt(3),
    )
    return df


def make_labels_col_multi_labels(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns(
        labels=pl.col("sentiment").replace_strict(
            {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]},
            return_dtype=pl.List(pl.Float32),
        ),
    )
    return df


# Load the dataset
df = (
    pl.scan_csv(
        "https://raw.githubusercontent.com/campusx-team/Text-Datasets/refs/heads/main/Reddit_Data.csv",
    )
    .rename({"clean_comment": "comment", "category": "sentiment"})
    .drop_nulls()
    .pipe(preprocess_df)
    .pipe(make_labels_col_multi_labels)
    .collect(engine=("gpu" if DEVICE == "cuda" else "cpu"))
    .sample(250, shuffle=True, seed=SEED)
)


# Prepare datasets
train_dataset, test_dataset = train_test_split(
    df,
    test_size=0.25,
    random_state=SEED,
    stratify=df["sentiment"],
)
train_dataset, val_dataset = train_test_split(
    train_dataset,
    test_size=0.2,
    random_state=SEED,
    stratify=train_dataset["sentiment"],
)


def tokenize_function(x):
    return tokenizer(
        x["comment"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


train_dataset = (
    Dataset.from_polars(train_dataset)
    .map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        drop_last_batch=True,
        remove_columns=["comment", "sentiment"],
    )
    .shuffle(seed=SEED)
)
val_dataset = Dataset.from_polars(val_dataset).map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    drop_last_batch=True,
    remove_columns=["comment", "sentiment"],
)
test_dataset = Dataset.from_polars(test_dataset).map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    drop_last_batch=True,
    remove_columns=["comment", "sentiment"],
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=MAX_LENGTH,
)


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    labels = np.argmax(labels, axis=-1)
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# Load BERT model
model_config = BertConfig(
    # NOTE: this is a multi-label classification
    problem_type="multi_label_classification",
    max_position_embeddings=MAX_LENGTH,
    label2id=dict(zip(LABELS, range(len(LABELS)), strict=True)),
    id2label=dict(enumerate(LABELS)),
    # this verify whether the dataset has the same number of labels
    num_labels=len(df["sentiment"].unique()),
)
model = BertForSequenceClassification.from_pretrained(
    CHECKPOINT,
    config=model_config,
    ignore_mismatched_sizes=True,
)
model = model.to(DEVICE)  # type: ignore

training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model and logs
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model checkpoint at the end of each epoch
    learning_rate=1e-5,  # Learning rate
    per_device_train_batch_size=BATCH_SIZE,  # Training batch size
    per_device_eval_batch_size=BATCH_SIZE,  # Evaluation batch size
    num_train_epochs=EPOCHS,  # Number of epochs
    weight_decay=0.01,  # Weight decay for optimizer
    logging_dir="./logs",  # Directory for logging
    logging_steps=10,  # Log every 10 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="f1",  # Use F1 score to evaluate the best model
    save_total_limit=2,  # Keep only 2 checkpoints
    fp16=torch.cuda.is_available(),  # Use FP16 if GPU supports it
    report_to="none",  # Avoid reporting to external services (e.g., WandB)
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)


def save_fintuned(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
    """Save the model and tokenizer"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved at {MODEL_DIR}")


def prediction(
    model: PreTrainedModel,
    *comments: str,
) -> torch.Tensor:
    """
    Predict sentiment of comments and return list of probabilities for all labels.
    """
    inputs: dict[str, torch.Tensor] = tokenizer(
        comments,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model.eval()  # set model to eval mode
    with torch.no_grad():  # disable gradient calculation
        outputs = model(**inputs)

    predictions = torch.softmax(outputs.logits, dim=-1)
    return predictions


# prediction(
#     model,
#     "this is bad for now.",  # 0: negative
#     "bad movie not a good direction",  # 0: negative
#     "i am now following you for pytorch playlist",  # 1: neutral
#     "be good boy and smile as much as you want, be happy.",  # 2: positive
# )
