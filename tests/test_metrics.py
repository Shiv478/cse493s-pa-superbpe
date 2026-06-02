import unittest

from pa_superbpe.evaluate import gini, max_min_ratio


class MetricTests(unittest.TestCase):
    def test_gini_is_zero_for_equal_values(self) -> None:
        self.assertAlmostEqual(gini([1.0, 1.0, 1.0]), 0.0)

    def test_gini_increases_for_unequal_values(self) -> None:
        self.assertGreater(gini([1.0, 2.0, 8.0]), 0.0)

    def test_max_min_ratio(self) -> None:
        self.assertAlmostEqual(max_min_ratio([2.0, 4.0, 8.0]), 4.0)


if __name__ == "__main__":
    unittest.main()

