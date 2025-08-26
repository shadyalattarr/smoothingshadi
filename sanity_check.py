#!/usr/bin/env python3
import sys
import os
import platform
import argparse
import importlib
import subprocess
from textwrap import indent

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{name:<12} ✅  version: {ver}")
        return mod
    except Exception as e:
        print(f"{name:<12} ❌  import failed: {e}")
        return None

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10)
        return out.strip()
    except Exception as e:
        return f"(failed to run: {e})"

def print_header(t):
    print("\n" + "="*len(t))
    print(t)
    print("="*len(t))

def main():
    parser = argparse.ArgumentParser(description="Environment sanity check")
    parser.add_argument("--train", action="store_true",
                        help="Run a tiny GPU training step if CUDA is available")
    args = parser.parse_args()

    # Avoid GUI backend issues when importing matplotlib
    os.environ.setdefault("MPLBACKEND", "Agg")

    print_header("Python & System")
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Working dir: {os.getcwd()}")

    print_header("Key Libraries")
    np = safe_import("numpy")
    scipy = safe_import("scipy")
    seaborn = safe_import("seaborn")
    matplotlib = safe_import("matplotlib")
    pandas = safe_import("pandas")
    statsmodels = safe_import("statsmodels")
    setGPU = safe_import("setGPU")  # optional helper package

    print_header("PyTorch / CUDA")
    torch = safe_import("torch")
    tv = None
    if torch:
        # Torch basics
        try:
            print(f"Torch version: {torch.__version__}")
            print(f"CUDA available (torch.cuda.is_available()): {torch.cuda.is_available()}")
            print(f"Built with CUDA (torch.version.cuda): {getattr(torch.version, 'cuda', None)}")
            try:
                cudnn_ver = torch.backends.cudnn.version()
            except Exception:
                cudnn_ver = None
            print(f"cuDNN version: {cudnn_ver}")

            # GPU listing
            try:
                n = torch.cuda.device_count()
                print(f"Detected GPU(s): {n}")
                for i in range(n):
                    print(f"  - [{i}] {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"GPU enumeration failed: {e}")

            # nvidia-smi (if present)
            print("\n`nvidia-smi` check:")
            smi = run_cmd("nvidia-smi")
            print(indent(smi, "  "))

            # TorchVision
            print_header("TorchVision")
            tv = safe_import("torchvision")
            if tv:
                print(f"torchvision compiled with CUDA ops: {torch.cuda.is_available()}")
                # Optionally test an op import
                try:
                    from torchvision.ops import nms  # noqa: F401
                    print("torchvision.ops.nms import ✅")
                except Exception as e:
                    print(f"torchvision.ops.nms import ❌  ({e})")

            # Optional setGPU behavior (only print effect; do not force environment changes mid-run)
            if setGPU:
                try:
                    # setGPU typically sets CUDA_VISIBLE_DEVICES on import; we won't reassign here.
                    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
                    print(f"setGPU present. Current CUDA_VISIBLE_DEVICES: {cvd}")
                except Exception as e:
                    print(f"setGPU check failed: {e}")

            # Quick CUDA compute test
            print_header("CUDA Compute Smoke Test")
            if torch.cuda.is_available():
                try:
                    device = torch.device("cuda:0")
                    a = torch.randn((1024, 1024), device=device)
                    b = torch.randn((1024, 1024), device=device)
                    c = a @ b  # matmul on GPU
                    torch.cuda.synchronize()
                    print("GPU tensor matmul ✅")
                except Exception as e:
                    print(f"GPU tensor matmul ❌  ({e})")
            else:
                print("CUDA not available — skipping GPU compute test.")

            # Optional tiny training step
            if args.train and torch.cuda.is_available():
                print_header("Tiny Training Step (GPU)")
                try:
                    import torch.nn as nn
                    import torch.optim as optim
                    device = torch.device("cuda:0")
                    model = nn.Sequential(nn.Linear(256, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 10)).to(device)
                    opt = optim.SGD(model.parameters(), lr=1e-2)
                    x = torch.randn(256, 256, device=device)
                    y = torch.randint(0, 10, (256,), device=device)
                    loss_fn = nn.CrossEntropyLoss()
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    print(f"Training step ✅  loss={loss.item():.4f}")
                except Exception as e:
                    print(f"Training step ❌  ({e})")
            elif args.train:
                print("`--train` requested but CUDA not available — skipping.")

        except Exception as e:
            print(f"Torch/CUDA check failed: {e}")

    print_header("Summary")
    def ok(name, mod):
        return f"{name:<12} {'✅' if mod else '❌'}"
    print(ok("numpy", np), ok("scipy", scipy), ok("seaborn", seaborn))
    print(ok("matplotlib", matplotlib), ok("pandas", pandas), ok("statsmodels", statsmodels))
    print(ok("setGPU", setGPU), ok("torch", torch), ok("torchvision", tv))

    print("\n✅ Sanity check complete.")

if __name__ == "__main__":
    main()
