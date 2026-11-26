import unittest
import torch
from legacy_research.models import build_model


class ModelTests(unittest.TestCase):
    def _check(self, arch):
        model = build_model(arch, num_classes=10).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 10))
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_resnet56_forward(self): self._check("resnet56")
    def test_vgg16_bn_forward(self): self._check("vgg16")
    def test_mobilenetv2_forward(self): self._check("mobilenetv2")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this environment")
    def test_resnet56_cuda_forward(self):
        model = build_model("resnet56", num_classes=10).cuda().eval()
        with torch.no_grad(): output = model(torch.randn(1, 3, 32, 32, device="cuda"))
        self.assertEqual(tuple(output.shape), (1, 10))


if __name__ == "__main__": unittest.main()
