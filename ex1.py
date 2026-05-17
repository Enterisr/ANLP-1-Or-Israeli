"""ANLP Exercise 1 — Fine-tuning on SST-2.

W&B sweep configs (recorded):
- Best:  epochs=5, batch_size=64, lr=5e-5
- Worst: epochs=5, batch_size=64, lr=5e-4
"""

import argparse
from pathlib import Path
import modal

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "google-bert/bert-base-uncased"
DEFAULT_OUTPUT_DIR = "./results"
PREDICTIONS_PATH = "predictions.txt"

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App(image=image)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help=(
            "Number of samples to be used during training or -1 if all training samples should be used. "
            "If n != -1, selects the first n samples in the training set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help=(
            "Number of samples to be used during validation or -1 if all validation samples should be used. "
            "If n != -1, selects the first n samples in the validation set."
        ),
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=-1,
        help=(
            "Number of samples to be used during prediction or -1 if all test samples should be used. "
            "If n != -1, selects the first n samples in the test set."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--do_train", action="store_true", help="Run training.")
    parser.add_argument("--do_predict", action="store_true", help="Run prediction and generate predictions.txt.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The model path to use when running prediction.",
    )
    return parser.parse_args()


def _take_first_n(dataset_split, max_samples: int):
    if max_samples is None or max_samples == -1:
        return dataset_split
    if max_samples < -1:
        raise ValueError("max_*_samples must be -1 or a non-negative integer")
    return dataset_split.select(range(min(len(dataset_split), max_samples)))


def _build_training_arguments(*, output_dir: str, args: argparse.Namespace) -> TrainingArguments:
    common_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )

    # Transformers changed kwarg names in some versions; keep compatibility.
    try:
        return TrainingArguments(
            **common_kwargs,
            eval_strategy="epoch",
            save_strategy="epoch",
        )
    except TypeError:
        return TrainingArguments(
            **common_kwargs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )


def main() -> None:
    args = _parse_args()

    if not args.do_train and not args.do_predict:
        raise SystemExit("Nothing to do: pass --do_train and/or --do_predict")
    if args.do_predict and not args.model_path and not args.do_train:
        raise SystemExit("--do_predict requires --model_path (or run with --do_train as well)")

    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = _take_first_n(tokenized["train"], args.max_train_samples)
    eval_dataset = _take_first_n(tokenized["validation"], args.max_eval_samples)
    predict_dataset = _take_first_n(tokenized["test"], args.max_predict_samples)

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        accuracy = float(np.mean(preds == labels))
        return {"accuracy": accuracy}

    output_dir = DEFAULT_OUTPUT_DIR
    trained_model_dir: str | None = None

    if args.do_train:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        training_args = _build_training_arguments(output_dir=output_dir, args=args)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        trained_model_dir = output_dir

    if args.do_predict:
        model_dir = args.model_path or trained_model_dir
        if not model_dir:
            raise SystemExit("Internal error: model_dir not set")

        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            raise SystemExit(f"model_path does not exist: {model_dir}")

        pred_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        pred_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        pred_data_collator = DataCollatorWithPadding(tokenizer=pred_tokenizer)

        pred_training_args = _build_training_arguments(output_dir=str(model_dir_path / "pred"), args=args)
        pred_trainer = Trainer(
            model=pred_model,
            args=pred_training_args,
            tokenizer=pred_tokenizer,
            data_collator=pred_data_collator,
        )

        test_results = pred_trainer.predict(predict_dataset)
        predictions = np.argmax(test_results.predictions, axis=-1).astype(int)
        np.savetxt(PREDICTIONS_PATH, predictions, fmt="%d")


if __name__ == "__main__":
    main()