"""Дедупликация с точным Жаккаром и полным индексом словных шинглов.

Индекс исключает пары без общих шинглов, фильтр длин — заведомо слишком
разные. В отличие от вероятностного LSH, положительный порог проверяется
без пропусков и ложных срабатываний. num_perm сохранён для API заготовки.
"""
from collections import defaultdict
from typing import Sequence

from src.textnorm import normalize_text, shingles


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    seen = set()
    dupes = []
    for i, key in enumerate(keys):
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


class ShingleIndex:
    def __init__(self, threshold: float):
        if not 0 < threshold <= 1:
            raise ValueError("Jaccard threshold must be in (0, 1]")
        self.threshold = threshold
        self.postings = defaultdict(set)
        self.items = {}

    def add(self, key: int, tokens: set[str]):
        self.items[key] = tokens
        for token in tokens:
            self.postings[token].add(key)

    def query(self, tokens: set[str]) -> list[int]:
        candidates = set()
        for token in tokens:
            candidates.update(self.postings.get(token, ()))
        matches = []
        for key in sorted(candidates):
            other = self.items[key]
            if min(len(tokens), len(other)) < self.threshold * max(len(tokens), len(other)):
                continue
            intersection = len(tokens & other)
            union = len(tokens) + len(other) - intersection
            if union and intersection / union >= self.threshold:
                matches.append(key)
        return matches


def near_duplicates(texts: Sequence[str], shingle_words: int,
                    num_perm: int, threshold: float) -> list[int]:
    index = ShingleIndex(threshold)
    dupes = []
    for i, text in enumerate(texts):
        tokens = shingles(normalize_text(text), shingle_words)
        if index.query(tokens):
            dupes.append(i)
        else:
            index.add(i, tokens)
    return dupes


def cross_near_duplicates(left: Sequence[str], right: Sequence[str],
                          shingle_words: int, num_perm: int,
                          threshold: float) -> list[tuple[int, int]]:
    index = ShingleIndex(threshold)
    for i, text in enumerate(left):
        index.add(i, shingles(normalize_text(text), shingle_words))
    return [(i, j) for j, text in enumerate(right)
            for i in index.query(shingles(normalize_text(text), shingle_words))]
