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
#     "transformers[torch]==4.48.0",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""## Finetune BERT model for sentiment analysis with **PEFT method LoRA**"""
    )


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
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EvalPrediction,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
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
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(DEVICE)
    return (DEVICE,)


@app.cell
def _(AutoTokenizer, CHECKPOINT):
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        clean_up_tokenization_spaces=True,
        strip_accents=True,
        tokenize_chinese_chars=False,
    )
    return (tokenizer,)


@app.cell
def _(MAX_LENGTH, tokenizer, torch):
    def encode_dataset(rows):
        rows = {
            "clean_comment": [i["clean_comment"] for i in rows],
            "category": [i["category"] for i in rows],
        }

        rows["clean_comment"] = [i.strip() for i in rows["clean_comment"]]

        if not all(i in [-1, 0, 1] for i in rows["category"]):
            raise ValueError(
                f"category column has unexpected value {rows['category']}."
            )

        tokenized_dict = tokenizer(
            rows["clean_comment"],
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "labels": torch.tensor([i + 1 for i in rows["category"]], dtype=torch.int),
            **tokenized_dict,
        }

    return (encode_dataset,)


@app.cell
def _():
    def filter_dataset(comments):
        return [c is not None and len(c.split(" ", maxsplit=4)) > 3 for c in comments]

    return (filter_dataset,)


@app.cell
def _(Dataset, filter_dataset):
    dataset: Dataset = (
        Dataset.from_csv(
            "https://raw.githubusercontent.com/campusx-team/Text-Datasets/refs/heads/main/Reddit_Data.csv",
            num_proc=4,
        )
        .filter(
            filter_dataset,
            input_columns=["clean_comment"],
            batched=True,
            num_proc=4,
        )
        .select(range(100))
    )
    dataset
    return (dataset,)


@app.cell
def _(Dataset, SEED):
    from datasets import DatasetDict

    def get_train_val_test_dataset(full_dataset: Dataset):
        train_test_dict = full_dataset.train_test_split(test_size=0.2, seed=SEED)
        train_val_dict = train_test_dict["train"].train_test_split(
            test_size=0.2, seed=SEED
        )
        return DatasetDict(
            train=train_val_dict["train"],
            validation=train_val_dict["test"],
            test=train_test_dict["test"],
        )

    return DatasetDict, get_train_val_test_dataset


@app.cell
def _(dataset, get_train_val_test_dataset):
    ds_dict = get_train_val_test_dataset(dataset)
    ds_dict
    return (ds_dict,)


@app.cell
def _(EvalPrediction, accuracy_score, f1_score, np):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": accuracy, "f1": f1}

    return (compute_metrics,)


@app.cell
def _(AutoModelForSequenceClassification, CHECKPOINT, LABELS, MAX_LENGTH):
    # Load BERT model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
        max_position_embeddings=MAX_LENGTH,
        label2id=dict(zip(LABELS, range(len(LABELS)), strict=True)),
        id2label=dict(enumerate(LABELS)),
        num_labels=len(LABELS),
    )
    return (base_model,)


@app.cell
def _(base_model):
    # Configure base_model with LoRA using `peft` library
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
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
    ds_dict,
    encode_dataset,
    model,
    torch,
):
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",  # don't save the model
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
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["validation"],
        data_collator=encode_dataset,
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
