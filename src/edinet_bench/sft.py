import argparse
import csv
import datasets
import json
import os
import pandas as pd
from datasets import ClassLabel
from enum import Enum
from functools import partial
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    set_seed,
)


class DatasetName(Enum):
    FRAUD_DETECTION = "fraud_detection"
    EARNINGS_FORECAST = "earnings_forecast"
    INDUSTRY_PREDICTION = "industry_prediction"


PROMPT = """
Please analyze the following information extracted from a Japanese company’s securities report for any signs of 
fraudulent activities. Please note that some data may be missing and represented as "-" due to parsing errors.
The report has been verified by a certified public accountant, and the numerical values are consistent and correct from 
a calculation perspective. Therefore, please focus your analysis on non-numerical inconsistencies or logical red flags 
that could suggest fraud.

{report}
"""


class CSVLogger(TrainerCallback):
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.path = os.path.join(dir, "log.csv")
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "eval_loss",
                "eval_accuracy",
                "eval_precision",
                "eval_recall",
                "eval_f1",
                "eval_auroc",
                "eval_mcc",
            ])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                state.epoch,
                metrics["eval_loss"],
                metrics["eval_accuracy"],
                metrics["eval_precision"],
                metrics["eval_recall"],
                metrics["eval_f1"],
                metrics["eval_auroc"],
                metrics["eval_mcc"],
            ])


def assign_industry_labels(dataset):
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset["train"]["industry"])
    class_names = list(label_encoder.classes_)

    def encode_industry(batch):
        return {"label": label_encoder.transform(batch["industry"])}

    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].map(encode_industry, batched=True)
        new_features = dataset[split_name].features.copy()
        new_features["label"] = ClassLabel(names=class_names)
        dataset[split_name] = dataset[split_name].cast(new_features)

    return dataset


def preprocess_text(example, sheets):
    return {
        "text": PROMPT.format(
            report="\n".join([f"{sheet}: {example[sheet]}" for sheet in sheets if sheet in example])
        ),
        "labels": example["label"],
    }


def split_dataset_by_edinet_code(
        dataset: datasets.Dataset,
        test_size: float = 0.2,
        seed: int = 42,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """
    EDINETコード単位でtrain/testに分割する。
    同じEDINETコードを持つデータは必ず同じsplitに属するようにする。
    """
    df = pd.DataFrame(dataset)

    # EDINETコードのユニークなリストを取得してtrain/testに分割
    unique_codes = df["edinet_code"].unique()
    train_codes, test_codes = train_test_split(
        unique_codes, test_size=test_size, random_state=seed
    )

    # EDINETコードに基づいてフィルタリング
    dataset_train = datasets.Dataset.from_pandas(df[df["edinet_code"].isin(train_codes)].reset_index(drop=True))
    dataset_test = datasets.Dataset.from_pandas(df[df["edinet_code"].isin(test_codes)].reset_index(drop=True))

    return dataset_train, dataset_test


def split_fraud_detection_dataset(dataset, train_ratio):
    dataset_trainval = dataset["train"]
    dataset_test = dataset["test"]
    dataset_train, dataset_val = split_dataset_by_edinet_code(dataset_trainval, 1 - train_ratio)
    return dataset_train, dataset_val, dataset_test


def split_earnings_forecast_dataset(dataset, train_year_cutoff):
    dataset_trainval = dataset["train"]
    dataset_test = dataset["test"]
    
    def is_train(example):
        meta = json.loads(example["meta"])
        return int(meta["当事業年度開始日"].split("-")[0]) < train_year_cutoff

    dataset_train = dataset_trainval.filter(is_train)
    dataset_val = dataset_trainval.filter(lambda x: not is_train(x))
    return dataset_train, dataset_val, dataset_test


def split_industry_prediction_dataset(dataset, train_ratio):
    dataset = dataset["train"]
    dataset = dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label",
    )
    dataset_trainval, dataset_test = dataset["train"], dataset["test"]
    dataset_trainval = dataset_trainval.train_test_split(
        test_size=1 - train_ratio,
        seed=42,
        stratify_by_column="label",
    )
    dataset_train, dataset_val = dataset_trainval["train"], dataset_trainval["test"]
    return dataset_train, dataset_val, dataset_test


def compute_metrics(pred, is_binary):
    labels = pred.label_ids
    pred_class = pred.predictions.argmax(-1)
    pred_prob = softmax(pred.predictions, axis=1)

    average_type = "binary" if is_binary else "macro"

    accuracy = accuracy_score(labels, pred_class)
    precision = precision_score(labels, pred_class, average=average_type)
    recall = recall_score(labels, pred_class, average=average_type)
    f1 = f1_score(labels, pred_class, average=average_type)
    mcc = matthews_corrcoef(labels, pred_class)
    auroc = roc_auc_score(labels, pred_prob[:, 1]) if is_binary else float("nan")

    return {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_mcc": mcc,
        "eval_auroc": auroc,
    }


def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.dataset_name == DatasetName.INDUSTRY_PREDICTION:
        num_classes = 16
        compute_metrics_p = partial(compute_metrics, is_binary=False)
    else:
        num_classes = 2
        compute_metrics_p = partial(compute_metrics, is_binary=True)

    if args.checkpoint_dir:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_classes)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.dataset_name == DatasetName.FRAUD_DETECTION:
        dataset = datasets.load_dataset("SakanaAI/EDINET-Bench", "fraud_detection")
        dataset = dataset.map(partial(preprocess_text, sheets=args.sheets))
        dataset_train, dataset_val, dataset_test = split_fraud_detection_dataset(dataset, args.train_ratio)
    elif args.dataset_name == DatasetName.EARNINGS_FORECAST:
        dataset = datasets.load_dataset("SakanaAI/EDINET-Bench", "earnings_forecast")
        dataset = dataset.map(partial(preprocess_text, sheets=args.sheets))
        dataset_train, dataset_val, dataset_test = split_earnings_forecast_dataset(dataset, args.train_year_cutoff)
    else:
        assert args.dataset_name == DatasetName.INDUSTRY_PREDICTION
        # dataset = datasets.load_dataset("SakanaAI/EDINET-Bench", "industry_prediction")
        dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_json(os.path.join(args.industry_prediction_rebuttal_dir, "train.json")),
            "test": datasets.Dataset.from_json(os.path.join(args.industry_prediction_rebuttal_dir, "test.json")),
        })
        dataset = assign_industry_labels(dataset)
        dataset = dataset.map(partial(preprocess_text, sheets=args.sheets))
        dataset_train, dataset_val, dataset_test = split_industry_prediction_dataset(dataset, args.train_ratio)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_seq_length)

    dataset_train = dataset_train.map(tokenize_function)
    dataset_val = dataset_val.map(tokenize_function)
    dataset_test = dataset_test.map(tokenize_function)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.results_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.train_steps,
        logging_strategy="no",
        eval_strategy="steps",
        eval_steps=args.steps_per_val,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="best",
        report_to="none",
        bf16=True,
        bf16_full_eval=True,
    )
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics_p,
        callbacks=[CSVLogger(args.results_dir)],
    )
    if not args.test_only:
        trainer.train()
    
    test_metrics = trainer.evaluate(dataset_test)
    
    with open(os.path.join(args.results_dir, "test_metrics.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        keys = list(test_metrics.keys())
        writer.writerow(keys)
        writer.writerow([test_metrics[key] for key in keys])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", choices=[
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
    ])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--industry_prediction_rebuttal_dir", type=str)
    parser.add_argument("--dataset_name", type=DatasetName, required=True)
    parser.add_argument("--sheets", type=str, nargs="+", default=["summary", "bs", "pl", "cf", "text"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_year_cutoff", type=int, default=2018)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--steps_per_val", type=int, default=1000)
    parser.add_argument("--test_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())