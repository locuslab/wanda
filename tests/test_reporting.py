import unittest

from lib.reporting import format_result_log


class ResultLogFormattingTests(unittest.TestCase):
    def test_result_log_includes_pruning_duration(self):
        result_log = format_result_log(
            method="wanda",
            sparsity_ratio=0.5,
            perplexity=6.42,
            pruning_time_seconds=183.42,
        )

        self.assertEqual(
            result_log,
            "method\tactual_sparsity\tppl_test\tpruning_time_seconds\n"
            "wanda\t0.5000\t6.4200\t183.42\n",
        )


if __name__ == "__main__":
    unittest.main()
