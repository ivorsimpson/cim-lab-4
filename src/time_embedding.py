"""Visualise the sinusoidal timestep embedding used by the diffusion UNet.

Usage:
    uv run src/time_embedding.py
"""

import argparse
from pathlib import Path

import torch
from diffusers.models.embeddings import get_timestep_embedding
from matplotlib import pyplot as plt

from models import ClassConditionedUnet

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dim", type=int, default=32, help="Size of the embedding")
parser.add_argument("--out", default="outputs")
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# Each timestep t (a single integer) becomes a vector of sines and cosines of different frequencies.
t = torch.arange(1000)
emb = get_timestep_embedding(t, args.dim, flip_sin_to_cos=True, downscale_freq_shift=0)  # (1000, dim)

# This is the function that the UNet in train.py applies to the timestep before anything else.
unet_emb = ClassConditionedUnet().model.time_proj(t)
if args.dim == unet_emb.shape[1]:
    print("Matches the UNet's own time_proj:", torch.allclose(emb, unet_emb, atol=1e-5))

# How similar are the embeddings of two timesteps? (cosine similarity)
unit = emb / emb.norm(dim=1, keepdim=True)
similarity = unit @ unit.T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
im1 = ax1.imshow(emb, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
ax1.set_xlabel("embedding dimension")
ax1.set_ylabel("timestep t")
ax1.set_title("Embedding of each timestep")
fig.colorbar(im1, ax=ax1)
im2 = ax2.imshow(similarity, cmap="viridis")
ax2.set_xlabel("timestep t")
ax2.set_ylabel("timestep t")
ax2.set_title("Cosine similarity between timesteps")
fig.colorbar(im2, ax=ax2)
fig.tight_layout()
fig.savefig(out / "time_embedding.png", dpi=150)
print(f"Saved time_embedding.png to {out}")
