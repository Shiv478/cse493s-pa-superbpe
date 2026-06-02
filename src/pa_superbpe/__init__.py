"""Fast research scaffold for parity-aware SuperBPE experiments."""

from pa_superbpe.evaluate import evaluate_tokenizer, gini, max_min_ratio
from pa_superbpe.tokenizer import BPETokenizer
from pa_superbpe.training import train_all_tokenizers

__all__ = [
    "BPETokenizer",
    "evaluate_tokenizer",
    "gini",
    "max_min_ratio",
    "train_all_tokenizers",
]

