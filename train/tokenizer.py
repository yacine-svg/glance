"""Byte-level BPE tokenizer, written from scratch.

Same idea as GPT-2's tokenizer: text is first cut into chunks (words, numbers,
punctuation, whitespace) by a regular expression, each chunk is turned into its
UTF-8 bytes, and the most frequent adjacent pair of tokens is merged into a new
token, again and again, until the vocabulary is full. Because it starts from
bytes, any text can be encoded: no "unknown token".

Training keeps pair counts up to date incrementally (only the words that
contain the merged pair are touched), which turns hours into seconds.

The pre-tokenisation pattern and the normalisation are mirrored exactly in
web/lib/tokenizer.ts so the browser encodes prompts the same way.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict

# Contractions, words with their leading space, short digit groups, punctuation
# runs, and whitespace. ASCII classes only, so Python `re` and JavaScript agree.
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PATTERN)

_REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", "\t": " ", "\r": "", "\f": " ", "\v": " ", "﻿": "",
}
_REPLACE_RE = re.compile("|".join(re.escape(k) for k in _REPLACE))


def normalize(text: str) -> str:
    """NFC, straight quotes, plain spaces. Mirrored in the browser."""
    text = unicodedata.normalize("NFC", text)
    return _REPLACE_RE.sub(lambda m: _REPLACE[m.group(0)], text)


def pretokenize(text: str) -> list[str]:
    return _PAT.findall(text)


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], eot: int):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.eot = eot
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.eot + 1

    # ---------- training ----------
    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, verbose: bool = True) -> "Tokenizer":
        n_merges = vocab_size - 256 - 1  # last id is <|endoftext|>
        counts = Counter(pretokenize(normalize(text)))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        merges: list[tuple[int, int]] = []
        for m in range(n_merges):
            if not pair_counts:
                break
            # most frequent pair; ties broken by the smaller pair for determinism
            best = max(pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))[0]
            if pair_counts[best] < 2:
                break
            new_id = 256 + m
            merges.append(best)
            for wi in list(where[best]):
                ids, f = words[wi], freqs[wi]
                # remove this word's old pairs
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                # merge every occurrence of `best` in the word
                out, i = [], 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
            where.pop(best, None)
            if verbose and (m + 1) % 500 == 0:
                tok = cls(merges, 256 + len(merges)).vocab[new_id]
                print(f"  merge {m + 1}/{n_merges}: {tok!r}")
        return cls(merges, 256 + len(merges))

    # ---------- encode / decode ----------
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            pair, new_id = self.merges[best_rank], 256 + best_rank
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(normalize(text)):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids if i != self.eot).decode("utf-8", errors="replace")

    # ---------- files ----------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte-bpe", "pattern": PATTERN, "eot": self.eot, "merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls([tuple(m) for m in d["merges"]], d["eot"])
