"""Data loading and preprocessing utilities for NEREL dataset."""

import logging
from typing import Dict, List, Tuple

from datasets import DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


id_to_label_name: Dict[int, str] = {
    0: "O",
    1: "B-AGE",
    2: "I-AGE",
    3: "B-AWARD",
    4: "I-AWARD",
    5: "B-CITY",
    6: "I-CITY",
    7: "B-COUNTRY",
    8: "I-COUNTRY",
    9: "B-CRIME",
    10: "I-CRIME",
    11: "B-DATE",
    12: "I-DATE",
    13: "B-DISEASE",
    14: "I-DISEASE",
    15: "B-DISTRICT",
    16: "I-DISTRICT",
    17: "B-EVENT",
    18: "I-EVENT",
    19: "B-FACILITY",
    20: "I-FACILITY",
    21: "B-FAMILY",
    22: "I-FAMILY",
    23: "B-IDEOLOGY",
    24: "I-IDEOLOGY",
    25: "B-LANGUAGE",
    26: "I-LAW",
    27: "B-LAW",
    28: "B-LOCATION",
    29: "I-LOCATION",
    30: "B-MONEY",
    31: "I-MONEY",
    32: "B-NATIONALITY",
    33: "I-NATIONALITY",
    34: "B-NUMBER",
    35: "I-NUMBER",
    36: "B-ORDINAL",
    37: "I-ORDINAL",
    38: "B-ORGANIZATION",
    39: "I-ORGANIZATION",
    40: "B-PENALTY",
    41: "I-PENALTY",
    42: "B-PERCENT",
    43: "I-PERCENT",
    44: "B-PERSON",
    45: "I-PERSON",
    46: "I-PRODUCT",
    47: "B-PRODUCT",
    48: "B-PROFESSION",
    49: "I-PROFESSION",
    50: "B-RELIGION",
    51: "I-RELIGION",
    52: "B-STATE_OR_PROVINCE",
    53: "I-STATE_OR_PROVINCE",
    54: "B-TIME",
    55: "I-TIME",
    56: "B-WORK_OF_ART",
    57: "I-WORK_OF_ART",
}


def load_and_prepare_data() -> Tuple[
    DatasetDict, Dict[str, int], Dict[int, str], List[str]
]:
    """
    Load the NEREL Short dataset and create label mappings.

    Returns:
        nerel: Hugging Face DatasetDict with train/dev/test splits.
        label2id: Mapping from label string to integer ID.
        id2label: Mapping from integer ID to label string.
        label_list: Sorted list of unique labels including O.
    """
    from datasets import load_dataset

    nerel = load_dataset("surdan/nerel_short")
    label_set = set()

    for split in ["train", "test", "dev"]:
        for item in nerel[split]:
            for tag_id in item["ids"]:
                label_set.add(id_to_label_name.get(tag_id, "O"))

    label_list = ["O"] + sorted([label for label in label_set if label != "O"])
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    logger.info(
        f"Loaded {len(nerel['train'])} train, {len(nerel['dev'])} dev, {len(nerel['test'])} test examples"
    )
    logger.info(f"Found {len(label_list) - 1} entity types")
    return nerel, label2id, id2label, label_list


def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
    max_len: int = 128,
) -> Dict[str, List[int]]:
    """
    Tokenize raw text and align NER labels with subword tokens.

    Args:
        examples: Dictionary with "sequences" (list of word tokens) and "ids" (NER tags).
        tokenizer: Hugging Face tokenizer.
        label2id: Mapping from label to integer ID.
        max_len: Maximum sequence length after padding.

    Returns:
        Dictionary with input_ids, attention_mask, and labels for the model.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for i in range(len(examples["sequences"])):
        tokens = examples["sequences"][i]
        tag_ids = examples["ids"][i]

        input_ids = [tokenizer.cls_token_id]
        labels = [-100]

        for j, token in enumerate(tokens):
            tag_id = tag_ids[j] if j < len(tag_ids) else 0
            label_name = id_to_label_name.get(tag_id, "O")
            label_idx = label2id[label_name]

            sub_tokens = tokenizer.tokenize(token)
            sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)

            if not sub_ids:
                continue

            input_ids.extend(sub_ids)
            labels.append(label_idx)
            labels.extend([-100] * (len(sub_ids) - 1))

        input_ids.append(tokenizer.sep_token_id)
        labels.append(-100)

        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
        else:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        attention_mask = [
            1 if tid != tokenizer.pad_token_id else 0 for tid in input_ids
        ]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }
