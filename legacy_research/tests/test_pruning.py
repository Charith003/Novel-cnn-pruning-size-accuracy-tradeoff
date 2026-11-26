import unittest
import tempfile
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from legacy_research.models import build_model
from legacy_research.pruning.criteria import random_importance, l1_norm_importance, taylor_importance, fpgm_importance
from legacy_research.pruning.engine import iterative_prune_and_finetune, prune_filters
from legacy_research.pruning.criteria import _weiszfeld_geometric_median
from legacy_research.pruning.utils import count_parameters
from legacy_research.prune import load_pruned_checkpoint


class PruningCriterionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model = build_model("resnet56", 10)

    def _check(self, scores):
        self.assertTrue(scores)
        for value in scores.values():
            self.assertEqual(value.ndim, 1)
            self.assertTrue(torch.isfinite(value).all().item())

    def test_random_seeded(self):
        a, b = random_importance(self.model, seed=3), random_importance(self.model, seed=3)
        self._check(a)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        first = self.model.get_prunable_layers()[0]
        g = torch.Generator().manual_seed(3)
        self.assertTrue(torch.equal(a[first.name], torch.rand(first.conv.weight.shape[0], generator=g)))

    def test_l1_scores_match_filter_weight_norms(self):
        scores = l1_norm_importance(self.model)
        self._check(scores)
        for layer in self.model.get_prunable_layers():
            expected = layer.conv.weight.detach().flatten(1).abs().sum(1).cpu()
            self.assertTrue(torch.allclose(scores[layer.name], expected))

    def test_fpgm_matches_geometric_median_distances(self):
        scores = fpgm_importance(self.model)
        self._check(scores)
        layer = self.model.get_prunable_layers()[0]
        flat = layer.conv.weight.detach().cpu().flatten(1).float()
        median = _weiszfeld_geometric_median(flat)
        self.assertTrue(torch.allclose(scores[layer.name], torch.linalg.vector_norm(flat-median, dim=1)))

    def test_each_criterion_prunes_a_structural_model(self):
        for criterion in (random_importance, l1_norm_importance, taylor_importance, fpgm_importance):
            with self.subTest(criterion=criterion.__name__):
                model = build_model("resnet56", 10)
                loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([1, 2])), batch_size=2)
                scores = criterion(model, loader, n_batches=1) if criterion is taylor_importance else criterion(model)
                free_before = sum(layer.conv.out_channels for layer in model.get_prunable_layers() if not layer.skip_locked)
                reduced = prune_filters(model, scores, .01, "resnet56")
                free_after = sum(layer.conv.out_channels for layer in reduced.get_prunable_layers() if not layer.skip_locked)
                self.assertLess(count_parameters(reduced), count_parameters(model))
                self.assertEqual(free_before - free_after, int(free_before * .01))
                self.assertEqual(tuple(reduced(torch.randn(1, 3, 32, 32)).shape), (1, 10))

    def test_lowest_scored_filter_is_removed(self):
        model = build_model("resnet56", 10)
        scores = random_importance(model, seed=11)
        free_layers = [layer for layer in model.get_prunable_layers() if not layer.skip_locked]
        lowest_layer = min(free_layers, key=lambda layer: scores[layer.name].min().item())
        lowest_index = int(scores[lowest_layer.name].argmin().item())
        before = lowest_layer.conv.weight.detach().clone()
        total = sum(scores[layer.name].numel() for layer in free_layers)
        pruned = prune_filters(model, scores, 1.2 / total, "resnet56")
        after = next(layer for layer in pruned.get_prunable_layers() if layer.name == lowest_layer.name).conv.weight.detach()
        expected = torch.cat((before[:lowest_index], before[lowest_index + 1:]), dim=0)
        self.assertTrue(torch.equal(after, expected))

    def test_taylor_scores_and_backward_signal(self):
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([1, 2])), batch_size=2)
        x, y = next(iter(loader))
        reference = copy.deepcopy(self.model).train()
        reference.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(reference(x), y).backward()
        scores = taylor_importance(self.model, loader, n_batches=1)
        self._check(scores)
        layer = self.model.get_prunable_layers()[0]
        reference_layer = dict(reference.named_modules())[layer.name]
        expected = (reference_layer.weight.grad * reference_layer.weight).flatten(1).abs().sum(1).cpu()
        self.assertTrue(torch.allclose(scores[layer.name], expected, atol=1e-5))

    def test_taylor_empty_loader_fails(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)))
        with self.assertRaisesRegex(RuntimeError, "empty"): taylor_importance(self.model, loader)

    def test_prune_ratio_validation(self):
        with self.assertRaisesRegex(ValueError, "target_ratio"):
            prune_filters(self.model, random_importance(self.model), 1.0, "resnet56")

    def test_score_shape_validation(self):
        scores = random_importance(self.model)
        key = next(iter(scores))
        scores[key] = scores[key].unsqueeze(0)
        with self.assertRaisesRegex(ValueError, "1-D Tensor"):
            prune_filters(self.model, scores, .1, "resnet56")

    def test_prune_finetune_and_evaluate(self):
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([1, 2])
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        source = build_model("resnet56", 10)
        baseline_params = count_parameters(source)
        baseline_filters = sum(layer.conv.out_channels for layer in source.get_prunable_layers() if not layer.skip_locked)
        result, logs = iterative_prune_and_finetune(
            source, target_ratio=.02, criterion_name="random",
            calibration_loader=loader, train_loader=loader, val_loader=loader,
            step_ratio=.02, finetune_epochs=1, device="cpu", arch="resnet56", verbose=False)
        self.assertEqual(len(logs), 1)
        self.assertIn("finetune_loss", logs[0])
        self.assertLess(count_parameters(result), baseline_params)
        pruned_filters = sum(layer.conv.out_channels for layer in result.get_prunable_layers() if not layer.skip_locked)
        self.assertEqual(baseline_filters - pruned_filters, int(baseline_filters * .02))
        logits = result(x)
        self.assertEqual(tuple(logits.shape), (2, 10))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "pruned.pt"
            torch.save({"model_object": result}, checkpoint)
            reloaded = load_pruned_checkpoint(checkpoint)
            self.assertEqual(tuple(reloaded(x).shape), (2, 10))
            self.assertEqual(count_parameters(reloaded), count_parameters(result))
        result.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(logits, y).backward()
        self.assertTrue(any(p.grad is not None for p in result.parameters()))


if __name__ == "__main__": unittest.main()
