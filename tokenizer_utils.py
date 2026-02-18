"""
tokenizer_utils.py
==================
Reusable BPE tokenizer + Trigram Language Model.
Extracted from the notebook so the FastAPI microservice can import them.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional


# ─── BPE Tokenizer ───────────────────────────────────────────────

class BPETokenizer:
    """
    Applies learned BPE merge rules to encode raw Urdu text
    into subword tokens.
    """

    def __init__(self, merges_path: str | Path):
        self.merges = self._load_merges(merges_path)

    @staticmethod
    def _load_merges(path: str | Path):
        merges = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return merges

    def encode_word(self, word: str) -> List[str]:
        """Break a single word into BPE subword tokens."""
        tokens = list(word)
        while True:
            merged = False
            for a, b in self.merges:
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == a and tokens[i + 1] == b:
                        tokens[i : i + 2] = [a + b]
                        merged = True
                    else:
                        i += 1
            if not merged:
                break
        return tokens

    def encode_text(self, text: str) -> List[str]:
        """Tokenize a full string (split on whitespace, then BPE each word, with \u2581 word boundary markers)."""
        WB = "\u2581"
        tokens = []
        for word in text.strip().split():
            tokens.append(WB)  # mark start of a new word
            tokens.extend(self.encode_word(word))
        return tokens


# ─── Trigram Language Model ───────────────────────────────────────

class TrigramLM:
    """
    Interpolated trigram language model.

    P(w3 | w1, w2) = λ1·P_uni(w3) + λ2·P_bi(w3|w2) + λ3·P_tri(w3|w1,w2)
    """

    def __init__(
        self,
        unigram_counts: dict,
        bigram_counts: dict,
        trigram_counts: dict,
        lambda1: float = 0.1,
        lambda2: float = 0.3,
        lambda3: float = 0.6,
        eot_token: str = "\uE002",
    ):
        self.unigram_counts = Counter(unigram_counts)
        self.bigram_counts = Counter(bigram_counts)
        self.trigram_counts = Counter(trigram_counts)
        self.total_unigrams = sum(self.unigram_counts.values())
        self.vocab = set(self.unigram_counts.keys())
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.eot_token = eot_token

        # Pre-compute denominators
        self.unigram_for_bigram: Counter = Counter()
        for (w1, _), c in self.bigram_counts.items():
            self.unigram_for_bigram[w1] += c

        self.bigram_for_trigram: Counter = Counter()
        for (w1, w2, _), c in self.trigram_counts.items():
            self.bigram_for_trigram[(w1, w2)] += c

    # ── Probabilities ─────────────────────────────────────────────

    def p_uni(self, w: str) -> float:
        return self.unigram_counts.get(w, 0) / self.total_unigrams

    def p_bi(self, w2: str, w1: str) -> float:
        denom = self.unigram_for_bigram.get(w1, 0)
        return self.bigram_counts.get((w1, w2), 0) / denom if denom else 0.0

    def p_tri(self, w3: str, w1: str, w2: str) -> float:
        denom = self.bigram_for_trigram.get((w1, w2), 0)
        return self.trigram_counts.get((w1, w2, w3), 0) / denom if denom else 0.0

    def prob(self, w3: str, w1: str, w2: str) -> float:
        return (
            self.lambda1 * self.p_uni(w3)
            + self.lambda2 * self.p_bi(w3, w2)
            + self.lambda3 * self.p_tri(w3, w1, w2)
        )

    # ── Sampling ──────────────────────────────────────────────────

    def sample_next(self, w1: str, w2: str, temperature: float = 1.0) -> str:
        probs = [(w, self.prob(w, w1, w2)) for w in self.vocab]
        probs = [(w, p) for w, p in probs if p > 0]
        if not probs:
            return random.choice(list(self.vocab))

        tokens, weights = zip(*probs)
        if temperature != 1.0:
            weights = [w ** (1.0 / temperature) for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(tokens, weights=weights, k=1)[0]

    # ── Generation ────────────────────────────────────────────────

    def generate(
        self,
        max_tokens: int = 300,
        temperature: float = 0.8,
        seed_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate tokens until EOT or max_tokens."""
        BOS = "<BOS>"

        if seed_tokens and len(seed_tokens) >= 2:
            w1, w2 = seed_tokens[-2], seed_tokens[-1]
            generated = list(seed_tokens)
        elif seed_tokens and len(seed_tokens) == 1:
            w1, w2 = BOS, seed_tokens[0]
            generated = list(seed_tokens)
        else:
            w1, w2 = BOS, BOS
            generated = []

        for _ in range(max_tokens):
            next_tok = self.sample_next(w1, w2, temperature=temperature)
            generated.append(next_tok)
            if next_tok == self.eot_token:
                break
            w1, w2 = w2, next_tok

        return generated

    @staticmethod
    def tokens_to_text(tokens: List[str], eot_token: str = "\uE002") -> str:
        """Join BPE tokens back into readable text. \u2581 tokens become spaces."""
        WB = "\u2581"
        # Special PUA tokens that should never appear in output
        SPECIAL_CHARS = {"\uE000", "\uE001", "\uE002"}  # EOS, EOP, EOT
        clean = [t for t in tokens if t not in (eot_token, "<BOS>")]
        parts = []
        for t in clean:
            if t == WB:
                parts.append(" ")
            else:
                # Strip any embedded special characters from within tokens
                stripped = t
                for sc in SPECIAL_CHARS:
                    stripped = stripped.replace(sc, "")
                if stripped:
                    parts.append(stripped)
        text = "".join(parts).strip()
        return " ".join(text.split())


# ─── Load model from JSON ────────────────────────────────────────

def load_model_from_json(model_path: str | Path) -> TrigramLM:
    """
    Load the saved trigram_model.json and reconstruct TrigramLM.
    The JSON uses '|||' as key separator for bigram/trigram counts.
    """
    with open(model_path, encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct count dicts with tuple keys
    unigram_counts = data["unigram_counts"]

    bigram_counts = {}
    for key, count in data["bigram_counts"].items():
        parts = key.split("|||")
        bigram_counts[(parts[0], parts[1])] = count

    trigram_counts = {}
    for key, count in data["trigram_counts"].items():
        parts = key.split("|||")
        trigram_counts[(parts[0], parts[1], parts[2])] = count

    lambdas = data["lambdas"]
    eot_token = data.get("eot_token", "\uE002")

    return TrigramLM(
        unigram_counts=unigram_counts,
        bigram_counts=bigram_counts,
        trigram_counts=trigram_counts,
        lambda1=lambdas[0],
        lambda2=lambdas[1],
        lambda3=lambdas[2],
        eot_token=eot_token,
    )
