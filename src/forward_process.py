"""Visualise the forward (noising) process on MNIST digits.

Usage:
    uv run src/forward_process.py
    uv run src/forward_process.py --timesteps 0 50 100 200 400 999 --num-digits 6
"""

import argparse
from pathlib import Path

import torch
import torchvision
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 100, 250, 500, 750, 999],
                    help="Timesteps (0-999) at which to show the corrupted images")
parser.add_argument("--num-digits", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs")
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)

dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
idx = torch.randperm(len(dataset))[: args.num_digits]
x0 = torch.stack([dataset[i][0] for i in idx]) * 2 - 1  # clean images in (-1, 1)

# The same scheduler as in train.py: it fixes how much noise is added at each timestep.
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# Use the same noise at every timestep so we see ONE image being gradually destroyed.
noise = torch.randn_like(x0)

fig, axes = plt.subplots(args.num_digits, len(args.timesteps),
                         figsize=(2 * len(args.timesteps), 2 * args.num_digits), squeeze=False)
for col, t in enumerate(args.timesteps):
    t_batch = torch.full((args.num_digits,), t, dtype=torch.long)
    xt = scheduler.add_noise(x0, noise, t_batch)

    # add_noise implements x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * noise
    abar = scheduler.alphas_cumprod[t]
    assert torch.allclose(xt, abar.sqrt() * x0 + (1 - abar).sqrt() * noise, atol=1e-5)

    for row in range(args.num_digits):
        ax = axes[row, col]
        ax.imshow(xt[row, 0].clip(-1, 1), cmap="Greys", vmin=-1, vmax=1)
        ax.axis("off")
        if row == 0:
            ax.set_title(f"t = {t}")
fig.tight_layout()
fig.savefig(out / "forward_process.png", dpi=150)
plt.close(fig)

# How much signal and how much noise is in x_t?
abar = scheduler.alphas_cumprod
plt.plot(abar.sqrt(), label=r"signal weight $\sqrt{\bar\alpha_t}$")
plt.plot((1 - abar).sqrt(), label=r"noise weight $\sqrt{1-\bar\alpha_t}$")
plt.xlabel("timestep t")
plt.legend()
plt.savefig(out / "noise_schedule.png", dpi=150)
plt.close()

print(f"The closed form x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) noise matches add_noise.")
print(f"Saved forward_process.png and noise_schedule.png to {out}")
