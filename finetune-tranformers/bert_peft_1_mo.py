# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets==3.2.0",
#     "marimo",
#     "numpy==2.2.1",
#     "peft==0.14.0",
#     "polars==1.19.0",
#     "scikit-learn==1.6.0",
#     "torch==2.5.1",
#     "tqdm",
#     "transformers[torch]==4.47.1",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("## Finetune BERT model for sentiment analysis with **PEFT method LoRA**")


@app.cell
def _():
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

    return (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
        DataCollatorWithPadding,
        Dataset,
        EvalPrediction,
        Path,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
        accuracy_score,
        f1_score,
        np,
        pl,
        torch,
        train_test_split,
    )


@app.cell
def _(Path):
    BATCH_SIZE = 16
    CHECKPOINT = "bert-base-uncased"
    EPOCHS = 2
    LABELS = ("negative", "neutral", "positive")
    MAX_LENGTH = 212  # word length of 98%tile comments' is ~200
    MODEL_DIR = Path("./model_output")
    SEED = 42
    return (
        BATCH_SIZE,
        CHECKPOINT,
        EPOCHS,
        LABELS,
        MAX_LENGTH,
        MODEL_DIR,
        SEED,
    )


@app.cell
def _(SEED, torch):
    # set seed
    torch.manual_seed(SEED)


@app.cell
def _(torch):
    # Get device
    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(DEVICE)
    return (DEVICE,)


@app.cell
def _(BertTokenizerFast, CHECKPOINT):
    tokenizer = BertTokenizerFast.from_pretrained(
        CHECKPOINT,
        clean_up_tokenization_spaces=True,
        strip_accents=True,
        tokenize_chinese_chars=False,
    )
    return (tokenizer,)


@app.cell
def _(pl):
    def preprocess_df(df: pl.LazyFrame) -> pl.LazyFrame:
        _punc = (
            r"[\"'-\+\/\-\&\(\)\[\]\{\}]"  # custom punctuations to remove from comments
        )
        df = df.with_columns(
            pl.col("comment").str.replace_all(_punc, ""),
        ).filter(
            pl.col("comment").str.split(" ").list.len().gt(3),
        )
        return df

    return (preprocess_df,)


@app.cell
def _(SEED, pl, preprocess_df):
    # Load the dataset
    df = (
        pl.scan_csv(
            "https://raw.githubusercontent.com/campusx-team/Text-Datasets/refs/heads/main/Reddit_Data.csv",
        )
        .rename({"clean_comment": "comment", "category": "labels"})
        .with_columns(
            labels=pl.col("labels").replace_strict(
                {-1: 0, 0: 1, 1: 2},
                return_dtype=pl.UInt8,
            ),
        )
        .drop_nulls()
        .pipe(preprocess_df)
        .collect()
        # NOTE: check below line before proceeding
        .sample(200, shuffle=True, seed=SEED)
    )
    return (df,)


@app.cell
def _(MAX_LENGTH, tokenizer):
    def tokenize_function(x):
        return tokenizer(
            x["comment"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    return (tokenize_function,)


@app.cell
def _(BATCH_SIZE, Dataset, SEED, df, tokenize_function, train_test_split):
    # Prepare datasets
    train_dataset, test_dataset = train_test_split(
        df,
        test_size=0.25,
        random_state=SEED,
        stratify=df["labels"],
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        random_state=SEED,
        stratify=train_dataset["labels"],
    )

    # Convert into datasets.Dataset class for further process
    train_dataset = (
        Dataset.from_polars(train_dataset)
        .map(
            tokenize_function,
            batched=True,
            batch_size=BATCH_SIZE,
            drop_last_batch=True,
            remove_columns=["comment"],
        )
        .shuffle(seed=SEED)
    )
    val_dataset = Dataset.from_polars(val_dataset).map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        drop_last_batch=True,
        remove_columns=["comment"],
    )
    test_dataset = Dataset.from_polars(test_dataset).map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        drop_last_batch=True,
        remove_columns=["comment"],
    )
    return test_dataset, train_dataset, val_dataset


@app.cell
def _(test_dataset, train_dataset, val_dataset):
    print(train_dataset, val_dataset, test_dataset, sep="\n\n")


@app.cell
def _(DataCollatorWithPadding, tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return (data_collator,)


@app.cell
def _(EvalPrediction, accuracy_score, f1_score, np):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        # labels = np.argmax(labels, axis=-1)
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": accuracy, "f1": f1}

    return (compute_metrics,)


@app.cell
def _(
    BertConfig,
    BertForSequenceClassification,
    CHECKPOINT,
    LABELS,
    MAX_LENGTH,
    df,
):
    # Load BERT model
    model_config = BertConfig(
        # NOTE: this is a single-label classification
        problem_type="single_label_classification",
        max_position_embeddings=MAX_LENGTH,
        label2id=dict(zip(LABELS, range(len(LABELS)), strict=True)),
        id2label=dict(enumerate(LABELS)),
        # this verify whether the dataset has the same number of labels
        num_labels=len(df["labels"].unique()),
    )
    base_model = BertForSequenceClassification.from_pretrained(
        CHECKPOINT,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    # base_model = base_model.to(DEVICE)  # type: ignore
    return base_model, model_config


@app.cell
def _(base_model):
    # Configure base_model with LoRA using `peft` library
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=4,
        lora_alpha=32,
        lora_dropout=0.2,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return LoraConfig, TaskType, get_peft_model, lora_config, model


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    Trainer,
    TrainingArguments,
    compute_metrics,
    data_collator,
    model,
    torch,
    train_dataset,
    val_dataset,
):
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        # load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer, training_args


@app.cell
def _(trainer):
    # Train the model
    trainer.train()


@app.cell
def _(trainer):
    # Evaluate the model
    eval_results = trainer.evaluate()
    eval_results
    return (eval_results,)


@app.cell
def _(MODEL_DIR, PreTrainedModel):
    def save_fintuned(model: PreTrainedModel):
        """
        Save the model.

        Note
        ----

            Do not save `tokenizer` because you can load it with `bert-base-uncased` checkpoint and use it `max_length=212` argument.
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        print(f"Model saved at {MODEL_DIR}")

    return (save_fintuned,)


@app.cell
def _(DEVICE, MAX_LENGTH, PreTrainedModel, tokenizer, torch):
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

    return (prediction,)


@app.cell
def _(model, prediction):
    predictions = prediction(
        model,
        "this is bad for now.",  # 0: negative
        "bad movie not a good direction",  # 0: negative
        "i am now following you for pytorch playlist",  # 1: neutral
        "be good boy and smile as much as you want, be happy.",  # 2: positive
    )
    predictions
    return (predictions,)


@app.cell
def _(predictions, torch):
    torch.argmax(predictions, dim=-1)


if __name__ == "__main__":
    app.run()
