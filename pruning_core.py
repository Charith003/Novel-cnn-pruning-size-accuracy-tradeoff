"""
AMFI pruning core: model, importance scoring, structured pruning, training and
experiment runner. Logic is taken from the original notebook and wrapped so the
Flask backend can run it as a background job.
"""
import os, io, copy, time, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

OUT_DIR = "amfi_outputs"
METHODS_ALL = ["Random", "L1", "Taylor", "FPGM", "AMFI"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- model
class SmallCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 128), num_classes=10):
        super().__init__()
        in_ch = 3
        self.convs, self.bns, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for out_ch in channels:
            self.convs.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.channels = list(channels)

    def forward(self, x):
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(F.relu(bn(conv(x))))
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- scoring
def minmax(x):
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def channel_activation_entropy(activation, bins=32):
    values = activation.detach().abs()
    out = []
    for c in range(values.shape[1]):
        v = values[:, c].reshape(-1)
        vmax = v.max()
        if vmax <= 1e-12:
            out.append(torch.tensor(0.0, device=values.device)); continue
        hist = torch.histc(v, bins=bins, min=0.0, max=float(vmax.item()))
        p = hist / (hist.sum() + 1e-8)
        out.append(-(p * torch.log(p + 1e-8)).sum())
    return torch.stack(out)


def calculate_importance(model, layer_index, method, calib_batch, device):
    model.eval()
    conv = model.convs[layer_index]
    captured = {}
    handle = conv.register_forward_hook(lambda m, i, o: captured.__setitem__("a", o))
    model.zero_grad(set_to_none=True)
    images, labels = calib_batch[0].to(device), calib_batch[1].to(device)
    F.cross_entropy(model(images), labels).backward()
    handle.remove()

    w = conv.weight.detach()
    l1 = w.abs().flatten(1).sum(1)
    taylor = (torch.zeros_like(l1) if conv.weight.grad is None
              else (w * conv.weight.grad.detach()).abs().flatten(1).sum(1))
    entropy = channel_activation_entropy(captured["a"])
    filt = F.normalize(w.flatten(1), p=2, dim=1)
    cos = torch.abs(filt @ filt.T); cos.fill_diagonal_(0)
    redundancy = cos.max(1).values

    if method == "Random":
        score = torch.rand_like(l1)
    elif method == "L1":
        score = l1
    elif method == "Taylor":
        score = taylor
    elif method == "FPGM":
        score = torch.cdist(filt, filt, p=2).sum(1)
    elif method == "AMFI":
        score = (0.25 * minmax(l1) + 0.35 * minmax(taylor)
                 + 0.20 * minmax(entropy) - 0.20 * minmax(redundancy))
    else:
        raise ValueError("Unknown pruning method")
    return score.detach()


# ---------------------------------------------------------------- pruning
def prune_one_layer(model, idx, method, ratio, calib_batch, device):
    score = calculate_importance(model, idx, method, calib_batch, device)
    old_conv, old_bn = model.convs[idx], model.bns[idx]
    n = old_conv.out_channels
    n_remove = min(int(round(n * ratio)), max(0, n - 8))
    if n_remove <= 0:
        return
    keep = n - n_remove
    k = torch.argsort(score, descending=True)[:keep].sort().values.to(old_conv.weight.device)

    new_conv = nn.Conv2d(old_conv.in_channels, keep, old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding,
                         bias=(old_conv.bias is not None)).to(old_conv.weight.device)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.data[k])
    model.convs[idx] = new_conv

    new_bn = nn.BatchNorm2d(keep, eps=old_bn.eps, momentum=old_bn.momentum).to(old_bn.weight.device)
    with torch.no_grad():
        new_bn.weight.copy_(old_bn.weight.data[k]); new_bn.bias.copy_(old_bn.bias.data[k])
        new_bn.running_mean.copy_(old_bn.running_mean[k]); new_bn.running_var.copy_(old_bn.running_var[k])
        new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
    model.bns[idx] = new_bn

    if idx < len(model.convs) - 1:
        nxt = model.convs[idx + 1]
        new_nxt = nn.Conv2d(keep, nxt.out_channels, nxt.kernel_size, stride=nxt.stride,
                            padding=nxt.padding, bias=(nxt.bias is not None)).to(nxt.weight.device)
        with torch.no_grad():
            new_nxt.weight.copy_(nxt.weight.data[:, k])
        model.convs[idx + 1] = new_nxt
    else:
        fc = model.classifier
        new_fc = nn.Linear(keep, fc.out_features).to(fc.weight.device)
        with torch.no_grad():
            new_fc.weight.copy_(fc.weight.data[:, k]); new_fc.bias.copy_(fc.bias.data)
        model.classifier = new_fc
    model.channels[idx] = keep


def prune_model(src, method, ratio, calib_batch, device):
    m = copy.deepcopy(src).to(device)
    for i in range(len(m.convs)):
        prune_one_layer(m, i, method, ratio, calib_batch, device)
    return m


# ---------------------------------------------------------------- metrics
def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def serialized_size_mb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


@torch.no_grad()
def measure_latency_ms(m, device, repeats=100):
    m.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    for _ in range(15): m(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats): m(x)
    if device.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / repeats


# ---------------------------------------------------------------- experiment runner
class Experiment:
    def __init__(self, cfg, log, set_progress, should_stop):
        self.cfg, self.log, self.set_progress, self.should_stop = cfg, log, set_progress, should_stop
        self.device = get_device()

    def _data(self):
        name = self.cfg["dataset"]
        if name == "CIFAR10":
            cls, mean, std, self.nc = datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10
        else:
            cls, mean, std, self.nc = datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100
        tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.log(f"Loading {name} (downloads on first run)...")
        train_ds = cls("./data", train=True, download=True, transform=tr)
        test_ds = cls("./data", train=False, download=True, transform=te)
        nw, pin = self.cfg.get("num_workers", 2), self.device.type == "cuda"
        bs = self.cfg.get("batch_size", 128)
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw, pin_memory=pin)
        self.calib = next(iter(DataLoader(Subset(train_ds, list(range(512))), batch_size=128, shuffle=False)))

    @torch.no_grad()
    def evaluate(self, m):
        m.eval(); c = t = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            c += m(x).argmax(1).eq(y).sum().item(); t += y.size(0)
        return 100.0 * c / t

    def fit(self, m, epochs, lr, tag):
        m.to(self.device)
        opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        crit = nn.CrossEntropyLoss()
        for ep in range(epochs):
            if self.should_stop(): raise InterruptedError("Stopped by user")
            m.train(); tl = tc = tn = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                out = m(x); loss = crit(out, y); loss.backward(); opt.step()
                tl += loss.item() * y.size(0); tc += out.argmax(1).eq(y).sum().item(); tn += y.size(0)
            sch.step()
            self.log(f"[{tag}] epoch {ep+1}/{epochs} loss {tl/tn:.4f} train {100*tc/tn:.2f}% test {self.evaluate(m):.2f}%")
        return m

    def _row(self, method, ratio, acc, pre, m):
        return {"Method": method, "Pruning_Ratio": ratio, "Accuracy_Percent": round(acc, 2),
                "Pre_Finetune_Accuracy": pre, "Parameters": count_parameters(m),
                "Model_Size_MB": serialized_size_mb(m), "Latency_ms": measure_latency_ms(m, self.device),
                "Channels": str(m.channels)}

    def run(self):
        c = self.cfg
        set_seed(c.get("seed", 42))
        os.makedirs(OUT_DIR, exist_ok=True)
        self.log(f"Device: {self.device}")
        self._data()
        total = 1 + len(c["ratios"]) * len(c["methods"]); done = 0

        base = SmallCNN(num_classes=self.nc).to(self.device)
        base = self.fit(base, c["base_epochs"], 0.08, "Baseline")
        torch.save(base.state_dict(), f"{OUT_DIR}/baseline_model.pth")
        rows = [self._row("Baseline", 0.0, self.evaluate(base), None, base)]
        done += 1; self.set_progress(100 * done / total)

        for ratio in c["ratios"]:
            for method in c["methods"]:
                self.log(f"=== {method} | {ratio:.0%} ===")
                pruned = prune_model(base, method, ratio, self.calib, self.device)
                pre = self.evaluate(pruned)
                self.log(f"Accuracy before fine-tuning: {pre:.2f}%")
                pruned = self.fit(pruned, c["finetune_epochs"], 0.01, f"{method} {int(ratio*100)}%")
                rows.append(self._row(method, ratio, self.evaluate(pruned), round(pre, 2), pruned))
                torch.save(pruned.state_dict(), f"{OUT_DIR}/{method.lower()}_{round(ratio*100)}pct.pth")
                done += 1; self.set_progress(100 * done / total)
                pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pruning_results.csv", index=False)
        self.log("Finished. Results saved.")
        return rows


# ---------------------------------------------------------------- inference
def load_model(channels, path, num_classes=10):
    m = SmallCNN(channels=tuple(channels), num_classes=num_classes)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def predict_image(model, pil_img):
    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                             transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    probs = F.softmax(model(tf(pil_img.convert("RGB")).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, 3)
    return [{"label": CIFAR10_CLASSES[i], "prob": round(float(p) * 100, 2)}
            for p, i in zip(top.values, top.indices)]
