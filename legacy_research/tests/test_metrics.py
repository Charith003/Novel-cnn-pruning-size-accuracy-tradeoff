import unittest
import torch
from legacy_research.pruning.utils import count_parameters, count_flops, compression_stats, model_size_mb


class MetricTests(unittest.TestCase):
    def test_parameter_and_size_metrics(self):
        model = torch.nn.Linear(3, 2)
        self.assertEqual(count_parameters(model, trainable_only=False), 8)
        self.assertEqual(count_parameters(model), 8)
        # The legacy estimator rounds to 0.001 MB, so a tiny model rounds to 0.000.
        self.assertEqual(model_size_mb(model), round(8 * 4 / 1024 / 1024, 3))

    def test_compression_metrics(self):
        result = compression_stats(100, 50, 200, 100)
        self.assertEqual(result["compression_ratio"], 2)
        self.assertEqual(result["params_reduction_pct"], 50)
        self.assertEqual(result["flops_reduction_pct"], 50)

    def test_flops_count(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Flatten(), torch.nn.Linear(3600, 2))
        self.assertGreater(count_flops(model, input_shape=(3, 32, 32)), 0)


if __name__ == "__main__": unittest.main()
