import unittest

from pa_superbpe.evaluate import evaluate_tokenizer
from pa_superbpe.training import TrainConfig, train_all_tokenizers


class TrainingTests(unittest.TestCase):
    def test_train_all_tokenizers_on_tiny_corpus(self) -> None:
        texts = {
            "eng_Latn": ["token fairness helps languages", "fair token choices matter"],
            "spa_Latn": ["la equidad ayuda idiomas", "las decisiones importan"],
            "zho_Hans": ["公平的分词帮助语言", "分词选择很重要"],
        }
        results = train_all_tokenizers(texts, TrainConfig(stage1_merges=8, stage2_merges=8, min_frequency=1))
        self.assertEqual(set(results), {"bpe", "superbpe", "parity_bpe", "pa_superbpe"})
        for result in results.values():
            self.assertGreater(len(result.tokenizer.merges), 0)
            rows = evaluate_tokenizer(result.tokenizer, texts)
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()

