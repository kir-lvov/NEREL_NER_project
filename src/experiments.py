"""Experiments for hyperparameter tuning."""

import logging
import time

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from src.data_utils import tokenize_and_align_labels
from src.model_utils import create_model

logger = logging.getLogger(__name__)


def run_experiment(config, nerel, tokenizer, label2id, id2label, label_list):
    """Run experiment with given config."""
    logger.info("=" * 60)
    logger.info(f"Experiment: {config['name']}")
    logger.info(
        f"  max_length={config['max_len']}, lr={config['learning_rate']}, dropout={config['dropout']}"
    )
    logger.info("=" * 60)

    tokenized = {}
    for split in ["train", "dev"]:
        tokenized[split] = nerel[split].map(
            lambda x: tokenize_and_align_labels(
                x, tokenizer, label2id, max_len=config["max_len"]
            ),
            batched=True,
            remove_columns=nerel[split].column_names,
        )

    model, _ = create_model(
        model_name="DeepPavlov/rubert-base-cased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        dropout=config["dropout"],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=2)
        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, p.label_ids):
            true = []
            pred = []
            for ptag, lab in zip(pred_seq, label_seq):
                if lab != -100:
                    pred.append(id2label[ptag])
                    true.append(id2label[lab])
            if true:
                true_labels.append(true)
                pred_labels.append(pred)

        return {
            "f1": f1_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "accuracy": accuracy_score(true_labels, pred_labels),
        }

    training_args = TrainingArguments(
        output_dir=f"./models/exp_{config['name'].replace(' ', '_')[:30]}",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=False,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Training...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start

    test_tok = nerel["test"].map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, label2id, max_len=config["max_len"]
        ),
        batched=True,
        remove_columns=nerel["test"].column_names,
    )

    pred = trainer.predict(test_tok)
    y_pred = []
    y_true = []
    for pred_seq, label_seq in zip(np.argmax(pred.predictions, axis=2), pred.label_ids):
        y_pred.append(
            [id2label[p] for p, lab in zip(pred_seq, label_seq) if lab != -100]
        )
        y_true.append([id2label[lab] for lab in label_seq if lab != -100])

    result = {
        "config": config,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "train_time": train_time,
    }

    logger.info(f"RESULT: F1 = {result['f1']:.4f}")
    return result


def compare_experiments(results):
    """Compare experiment results."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Config':<40} {'F1':<8} {'Prec':<8} {'Rec':<8}")
    logger.info("-" * 65)

    for r in results:
        name = r["config"]["name"][:38]
        logger.info(
            f"{name:<40} {r['f1']:.4f}   {r['precision']:.4f}   {r['recall']:.4f}"
        )

    best = max(results, key=lambda x: x["f1"])
    logger.info(f"BEST: {best['config']['name']} (F1 = {best['f1']:.4f})")
