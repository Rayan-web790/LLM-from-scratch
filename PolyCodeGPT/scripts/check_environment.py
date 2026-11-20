#!/usr/bin/env python
"""PolyCodeGPT — Environment Verification Script.

Verifies Python, PyTorch, CUDA, NumPy, and tokenizers installation.
Runs actual CUDA computation tests including matrix multiplication
and a small neural network forward/backward pass.

Returns a nonzero exit code if any required dependency or CUDA test fails.
"""

import sys
import os

# Track overall pass/fail
all_passed = True


def report(label, value=None, passed=True):
    global all_passed
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    if value is not None:
        print(f"  [{status}] {label}: {value}")
    else:
        print(f"  [{status}] {label}")
    return passed


def main():
    print("=" * 60)
    print("PolyCodeGPT — Environment Verification")
    print("=" * 60)

    # ── Python ──────────────────────────────────────────────
    print("\n[1] Python")
    report("Version", sys.version.split()[0])
    report("Executable", sys.executable)

    # ── PyTorch ─────────────────────────────────────────────
    print("\n[2] PyTorch")
    try:
        import torch

        report("Version", torch.__version__)
        report("CUDA build", torch.version.cuda)
        cuda_available = torch.cuda.is_available()
        report("CUDA available", cuda_available, passed=cuda_available)

        if not cuda_available:
            report("GPU name", "N/A", passed=False)
            report("Compute capability", "N/A", passed=False)
            report("GPU memory", "N/A", passed=False)
        else:
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            report("GPU name", gpu_name, passed=gpu_name == "NVIDIA GeForce RTX 5060 Laptop GPU")
            report("Compute capability", cc, passed=cc == (12, 0))
            report("GPU memory", f"{gpu_mem:.2f} GB")
            report("Architecture list", torch.cuda.get_arch_list())
    except Exception as e:
        report("PyTorch import", str(e), passed=False)
        torch = None

    # ── NumPy ───────────────────────────────────────────────
    print("\n[3] NumPy")
    try:
        import numpy as np

        report("Version", np.__version__)
        report("Import", "OK")
    except Exception as e:
        report("NumPy import", str(e), passed=False)
        np = None

    # ── tokenizers ──────────────────────────────────────────
    print("\n[4] tokenizers")
    try:
        import tokenizers

        report("Version", tokenizers.__version__)
        report("Import", "OK")
    except Exception as e:
        report("tokenizers import", str(e), passed=False)
        tokenizers = None

    # ── tqdm ────────────────────────────────────────────────
    print("\n[5] tqdm")
    try:
        import tqdm

        report("Version", tqdm.__version__)
        report("Import", "OK")
    except Exception as e:
        report("tqdm import", str(e), passed=False)

    # ── PyYAML ──────────────────────────────────────────────
    print("\n[6] PyYAML")
    try:
        import yaml

        report("Version", yaml.__version__)
        report("Import", "OK")
    except Exception as e:
        report("PyYAML import", str(e), passed=False)

    # ── CUDA Matrix Multiplication ──────────────────────────
    print("\n[7] CUDA Matrix Multiplication Test")
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            finite = torch.all(torch.isfinite(c)).item()
            report("Matrix mult 1024x1024", "finite" if finite else "non-finite", passed=finite)
        except Exception as e:
            report("Matrix multiplication", str(e), passed=False)
    else:
        report("Matrix multiplication", "SKIPPED (no CUDA)", passed=False)

    # ── Neural Network Forward/Backward ─────────────────────
    print("\n[8] Neural Network Forward/Backward Test")
    if torch is not None and torch.cuda.is_available():
        try:
            import torch.nn as nn

            class SmallNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(256, 512)
                    self.gelu = nn.GELU()
                    self.linear2 = nn.Linear(512, 128)

                def forward(self, x):
                    x = self.gelu(self.linear1(x))
                    x = self.linear2(x)
                    return x

            model = SmallNet().to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            inputs = torch.randn(32, 256, device="cuda")
            targets = torch.randn(32, 128, device="cuda")

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            params_on_cuda = all(p.device.type == "cuda" for p in model.parameters())
            report("Forward pass", f"loss={loss.item():.6f}")
            report("Backward pass", "OK")
            report("Optimizer step", "OK")
            report("Parameters on CUDA", params_on_cuda, passed=params_on_cuda)
        except Exception as e:
            report("Neural network test", str(e), passed=False)
    else:
        report("Neural network test", "SKIPPED (no CUDA)", passed=False)

    # ── NumPy Interoperability ──────────────────────────────
    print("\n[9] NumPy Interoperability Test")
    if torch is not None and np is not None:
        try:
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            tensor = torch.from_numpy(arr)
            back = tensor.numpy()
            match = np.allclose(arr, back)
            report("NumPy -> Tensor conversion", "OK", passed=match)
            report("Tensor -> NumPy conversion", "OK", passed=match)
        except Exception as e:
            report("NumPy interoperability", str(e), passed=False)
    else:
        report("NumPy interoperability", "SKIPPED (missing dep)", passed=False)

    # ── GPU Memory ──────────────────────────────────────────
    if torch is not None and torch.cuda.is_available():
        print("\n[10] GPU Memory")
        report("Allocated", f"{torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        report("Reserved", f"{torch.cuda.memory_reserved() / (1024**2):.2f} MB")
        report("Max allocated", f"{torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")

    # ── Final Result ────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
