"""Model creation and training utilities."""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def create_model(
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dropout: float = 0.15,
) -> Tuple[AutoModelForTokenClassification, torch.device]:
    """
    Load a pretrained transformer model for token classification.

    Returns:
        model: The loaded model.
        device: torch device (cuda or cpu).
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Model loaded on {device}, dropout = {dropout}")
    return model, device


def setup_trainer(
    model: AutoModelForTokenClassification,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    eval_dataset,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_dir: str = "./nerel_rubert_model",
) -> Trainer:
    """Configure and return a Hugging Face Trainer."""
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
            "accuracy": accuracy_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2.5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        save_total_limit=1,
    )

    logger.info("Trainer configured with batch_size=2, lr=2.5e-5, epochs=2")
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
