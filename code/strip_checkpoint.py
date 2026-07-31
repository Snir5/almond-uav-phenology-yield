"""Strip optimizer state from a training checkpoint so the weights can be shared.

The training checkpoint stores the model weights alongside the Adam optimizer state,
which keeps two extra buffers per parameter and triples the file size. Inference needs
only the weights.

Usage:
    python strip_checkpoint.py <input.pth> <output.pth>
"""

import os
import sys

import torch


def strip(src, dst):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        raise SystemExit(f"{src} has no 'model_state_dict' key; keys are {list(ckpt)}")

    keep = {"model_state_dict": ckpt["model_state_dict"]}
    for k in ("epoch", "val_dice", "loss"):
        if k in ckpt:
            keep[k] = ckpt[k]

    torch.save(keep, dst)

    before = os.path.getsize(src) / 1048576
    after = os.path.getsize(dst) / 1048576
    print(f"{src}  {before:.1f} MB")
    print(f"{dst}  {after:.1f} MB")
    print(f"dropped {before - after:.1f} MB of optimizer state")

    # Confirm the result still loads and carries every weight tensor.
    check = torch.load(dst, map_location="cpu", weights_only=True)
    n_src = len(ckpt["model_state_dict"])
    n_dst = len(check["model_state_dict"])
    if n_src != n_dst:
        raise SystemExit(f"tensor count changed: {n_src} -> {n_dst}")
    print(f"verified: {n_dst} weight tensors intact, file reloads cleanly")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    strip(sys.argv[1], sys.argv[2])
