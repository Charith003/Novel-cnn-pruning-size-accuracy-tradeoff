import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
import yaml
from torch.utils.data import TensorDataset
from legacy_research.data.cifar import build_cifar10_datasets, build_cifar100_datasets, CIFAR10_MEAN, CIFAR10_STD
from legacy_research.data.custom import build_custom_datasets
from legacy_research.data.factory import get_dataloaders


class DataTests(unittest.TestCase):
    def test_cifar10_no_download_by_default(self):
        fake = TensorDataset(torch.zeros(20, 3, 32, 32), torch.zeros(20, dtype=torch.long))
        with patch("legacy_research.data.cifar.datasets.CIFAR10", return_value=fake) as ctor:
            split = build_cifar10_datasets(root="unused", val_size=4)
        self.assertEqual((len(split["train"]), len(split["val"]), split["num_classes"]), (16, 4, 10))
        self.assertFalse(ctor.call_args.kwargs["download"])
        normalization = ctor.call_args.kwargs["transform"].transforms[-1]
        self.assertEqual(normalization.mean, CIFAR10_MEAN)
        self.assertEqual(normalization.std, CIFAR10_STD)

    def test_cifar_download_requires_explicit_opt_in(self):
        fake = TensorDataset(torch.zeros(20, 3, 32, 32), torch.zeros(20, dtype=torch.long))
        with patch("legacy_research.data.cifar.datasets.CIFAR10", return_value=fake) as ctor:
            build_cifar10_datasets(root="unused", val_size=4, download=True)
        self.assertEqual(len(ctor.call_args_list), 3)
        self.assertTrue(all(call.kwargs["download"] for call in ctor.call_args_list))

    def test_cifar100_class_count_and_no_download(self):
        fake = TensorDataset(torch.zeros(20, 3, 32, 32), torch.zeros(20, dtype=torch.long))
        with patch("legacy_research.data.cifar.datasets.CIFAR100", return_value=fake) as ctor:
            split = build_cifar100_datasets(root="unused", val_size=4)
        self.assertEqual(split["num_classes"], 100)
        self.assertFalse(ctor.call_args.kwargs["download"])

    def test_missing_cifar_has_explicit_guidance(self):
        with patch("legacy_research.data.cifar.datasets.CIFAR10", side_effect=RuntimeError("not found")):
            with self.assertRaisesRegex(FileNotFoundError, "--download"):
                build_cifar10_datasets(root="missing-dataset")

    def test_nine_yaml_configs_parse_and_paths_are_module_local(self):
        module_root = Path(__file__).resolve().parents[1]
        paths = sorted((module_root / "configs").glob("*.yaml"))
        self.assertEqual(len(paths), 9)
        allowed_arch = {"resnet56", "vgg16", "mobilenetv2"}
        allowed_datasets = {"cifar10", "cifar100", "custom"}
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIn(cfg["arch"], allowed_arch, path.name)
            self.assertIn(cfg["dataset"], allowed_datasets, path.name)
            self.assertGreater(cfg["epochs"], 0)
            self.assertGreater(cfg["lr"], 0)
            self.assertGreater(cfg["momentum"], 0)
            self.assertGreaterEqual(cfg["weight_decay"], 0)
            self.assertEqual(cfg["lr_schedule"], "cosine")
            self.assertTrue(cfg["nesterov"])
            self.assertNotIn("criterion", cfg)  # source YAMLs are training configs
            self.assertNotIn("target_ratio", cfg)
            for key in ("data_root", "checkpoint_dir", "results_dir"):
                resolved = (module_root / cfg[key]).resolve()
                self.assertTrue(resolved.is_relative_to(module_root.resolve()), (path.name, key, resolved))

    def test_custom_imagefolder_factory_and_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, color in (("cats", (255, 0, 0)), ("dogs", (0, 255, 0))):
                folder = root / name; folder.mkdir()
                for i in range(5): Image.new("RGB", (12, 12), color).save(folder / f"{i}.png")
            split = build_custom_datasets(str(root), .2, .2, image_size=16, mean=(.5,)*3, std=(.5,)*3)
            self.assertEqual(split["classes"], ["cats", "dogs"])
            train, val, test, meta = get_dataloaders("custom", str(root), batch_size=2, num_workers=0,
                val_fraction=.2, custom_test_fraction=.2, image_size=16, custom_mean=(.5,)*3, custom_std=(.5,)*3)
            self.assertEqual(tuple(next(iter(train))[0].shape), (2, 3, 16, 16))
            self.assertEqual((len(val.dataset), len(test.dataset), meta["num_classes"]), (2, 2, 2))

    def test_unknown_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            get_dataloaders("unknown", batch_size=2, num_workers=0)


if __name__ == "__main__": unittest.main()
