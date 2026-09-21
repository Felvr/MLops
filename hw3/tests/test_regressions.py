"""Регрессии для исправлений, которые дополняют восемь проверок курса."""
import unittest

from src.dedup import ShingleIndex, cross_near_duplicates, near_duplicates
from src.pii import scrub
from src.split import group_split


class RegressionTests(unittest.TestCase):
    def test_jaccard_index_matches_bruteforce(self):
        sets = [{str(i) for i in range(10) if mask & (1 << i)}
                for mask in range(1, 1024, 7)]
        index = ShingleIndex(0.85)
        for i, tokens in enumerate(sets):
            index.add(i, tokens)
        for tokens in sets:
            expected = [i for i, other in enumerate(sets)
                        if len(tokens & other) / len(tokens | other) >= 0.85]
            self.assertEqual(index.query(tokens), expected)

    def test_mcq_options_reordering(self):
        question = "Вопрос: Какой метод применяется для обработки медицинских изображений? Варианты ответа: "
        a = question + "0. свёрточная сеть 1. линейная регрессия 2. метод ближайших соседей"
        b = question + "0. метод ближайших соседей 1. свёрточная сеть 2. линейная регрессия"
        self.assertEqual(near_duplicates([a, b], 4, 64, 0.85), [1])
        self.assertEqual(cross_near_duplicates([a], [b], 4, 64, 0.85), [(0, 0)])

    def test_group_assignments_survive_extension_and_reordering(self):
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        before = group_split(["A", "B", "C"], ratios, 42)
        after = group_split(["D", "C", "B", "A"], ratios, 42)
        self.assertEqual(before, {k: after[k] for k in before})

    def test_pii_russian_and_english(self):
        value, hits = scrub("Телефон +7 (999) 123-45-67, date of birth: 12.03.1975, "
                            "patient@example.org; phone (212) 555-0199")
        self.assertEqual(hits, {"phone": 2, "email": 1, "birth_date": 1})
        self.assertNotIn("1975", value)
        self.assertNotIn("example.org", value)


if __name__ == "__main__":
    unittest.main()
