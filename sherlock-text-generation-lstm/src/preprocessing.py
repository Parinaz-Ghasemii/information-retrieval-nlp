import os
import re
import json
import torch
import collections
from typing import Tuple, Dict, List
from src import utils
from src.config import ProjectConfig


def combine_and_load_raw_data(raw_dir: str, output_path: str) -> str:
    """
    Provided helper: combines all .txt files into a single corpus.
    """
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    combined_text = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(".txt") and file != "sherlock.txt":
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as infile:
                        combined_text.append(infile.read())
                except:
                    pass

    full_text = "\n".join(combined_text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return full_text


def clean_text(text: str) -> str:
    """
    Normalize text:
    - lowercase
    - keep basic punctuation
    - remove non-standard characters
    - normalize whitespace
    """
    text = text.lower()

    # Keep letters, numbers, and common punctuation
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\;\:\'\-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_vocab(tokens: List[str], max_size: int, min_freq: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary mappings.
    Must include:
    <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
    """
    counter = collections.Counter(tokens)

    # Special tokens (fixed IDs)
    token_to_id = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3
    }

    # Reserve space for special tokens
    max_size = max_size - len(token_to_id)

    # Sort by frequency then alphabetically (stable, deterministic)
    sorted_tokens = sorted(
        [(tok, freq) for tok, freq in counter.items() if freq >= min_freq],
        key=lambda x: (-x[1], x[0])
    )

    for token, _ in sorted_tokens[:max_size]:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    id_to_token = {idx: tok for tok, idx in token_to_id.items()}

    return token_to_id, id_to_token


def run_preprocessing_pipeline(config: ProjectConfig):
    logger = utils.get_logger("preprocessing")
    logger.info("Starting Preprocessing...")

    raw_dir = utils.RAW_DATA_DIR
    combined_path = os.path.join(raw_dir, "sherlock.txt")

    # 1. Load & clean text
    raw_text = combine_and_load_raw_data(raw_dir, combined_path)
    clean_corpus = clean_text(raw_text)

    # 2. Tokenize
    tokens = clean_corpus.split()

    # 3. Build vocabulary
    token_to_id, id_to_token = build_vocab(
        tokens,
        max_size=config.max_vocab_size,
        min_freq=2
    )

    # 4. Encode tokens
    encoded = [
        token_to_id.get(tok, token_to_id["<UNK>"])
        for tok in tokens
    ]

    # 5. Convert to tensor
    encoded_tensor = torch.tensor(encoded, dtype=torch.long)

    # 6. Save artifacts
    os.makedirs(os.path.dirname(utils.SEQUENCES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(utils.VOCAB_PATH), exist_ok=True)

    torch.save(encoded_tensor, utils.SEQUENCES_PATH)

    with open(utils.VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "token_to_id": token_to_id,
                "id_to_token": id_to_token
            },
            f,
            indent=2
        )

    logger.info(f"Vocabulary size: {len(token_to_id)}")
    logger.info(f"total tokens: {len(tokens)}")
    logger.info("Preprocessing pipeline finished.")
