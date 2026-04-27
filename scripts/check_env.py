import platform
import sys

import torch

print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    x = torch.randn(4, 4, device="mps")
    y = (x @ x.T).sum().item()
    print("mps smoke op ok, scalar:", y)
