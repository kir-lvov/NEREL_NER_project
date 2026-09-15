"""Inference utilities for extracting entities from text."""

import logging

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def extract_entities(
    text: str,
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    id2label: dict,
    device: str = "cpu",
) -> list:
    """
    Extract named entities from raw text.

    Returns:
        List of tuples (start_char, end_char, entity_type, entity_text).
    """
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding.pop("offset_mapping")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=2)

    predictions = predictions[0].cpu().numpy()
    offset_mapping = offset_mapping[0].cpu().numpy()

    entities = []
    current_entity = None

    for i, (pred_idx, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        if start == 0 and end == 0:
            continue
        label = id2label.get(pred_idx, "O")

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = (int(start), int(end), label[2:], text[start:end])
        elif label.startswith("I-") and current_entity:
            s, _, t, _ = current_entity
            current_entity = (s, int(end), t, text[s:end])
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)
    return entities


class NERExtractor:
    """Load model once and use for multiple predictions."""

    def __init__(self, model_path: str = "./best_ner_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"Loaded NER model from {model_path}")

    def extract(self, text: str) -> list:
        """Extract entities using the loaded model."""
        return extract_entities(
            text, self.model, self.tokenizer, self.id2label, self.device
        )
