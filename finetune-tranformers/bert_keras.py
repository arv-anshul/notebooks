# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "keras",
#     "pandas",
#     "scikit-learn",
#     "tensorflow",
#     "tf-keras",
#     "tqdm",
#     "transformers",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

"""
## Learn and Improve

- How to do prediction?
- A better approach for dataset creation. (maybe `return_tensor="tf"`)
- Create a new script where `transformers.TFTranier` class is used for finetunning.
"""

import pandas as pd
from datasets import Dataset
from keras.api.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tf_keras.optimizers import Adam
from transformers import (
    DataCollatorWithPadding,
    RobertaTokenizerFast,
    TFPreTrainedModel,
    TFRobertaForSequenceClassification,
)

CHECKPOINT = "roberta-base"
MAX_LENGTH = 212  # word length of 98%tile comments' is ~200
EPOCHS = 3
BATCH_SIZE = 16
SEED = 42


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
    .sample(100)
)

tokenizer = RobertaTokenizerFast.from_pretrained(CHECKPOINT)


def tokenize_function(x):
    return tokenizer(
        x["comment"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


# Prepare datasets
train_dataset, test_dataset = train_test_split(
    df,
    test_size=0.25,
    stratify=df["labels"],
    random_state=SEED,
)
train_dataset, val_dataset = train_test_split(
    train_dataset,
    test_size=0.2,
    stratify=train_dataset["labels"],
    random_state=SEED,
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=MAX_LENGTH,
)

train_dataset = (
    Dataset.from_pandas(
        train_dataset,
        preserve_index=False,
    )
    .map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        drop_last_batch=True,
        remove_columns=["comment"],
    )
    .to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=True,
    )
)
val_dataset = (
    Dataset.from_pandas(
        val_dataset,
        preserve_index=False,
    )
    .map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["comment"],
    )
    .to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )
)
test_dataset = (
    Dataset.from_pandas(
        test_dataset,
        preserve_index=False,
    )
    .map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["comment"],
    )
    .to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )
)


# Load BERT model
model: TFPreTrainedModel = TFRobertaForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(df["labels"].unique()),
)

if not isinstance(model, TFPreTrainedModel):
    raise TypeError("error while loading model")

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=model.hf_compute_loss,  # Hugging Face provides its own loss function
    metrics=["accuracy"],
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
)

# Train the model
history = model.fit(
    train_dataset,  # type: ignore
    validation_data=val_dataset,
    epochs=1,
    # callbacks=[early_stopping],
)

# Evaluate the model on the test set
# test_gen = DataGenerator(test_dataset["input_ids"], test_dataset["label"])
# results = model.evaluate(test_gen)
# print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
