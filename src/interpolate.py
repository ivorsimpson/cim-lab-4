"""Interpolate between two initial noise samples and denoise each one with DDIM.

DDIM is deterministic (no noise is injected during sampling), so every initial
noise tensor maps to exactly one image. Moving smoothly through the noise space
therefore moves smoothly through image space.

Usage:
    uv run src/interpolate.py
    uv run src/interpolate.py --seed-a 1 --seed-b 2 --num-frames 12
    uv run src/interpolate.py --mode linear
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import DDIMScheduler, DDPMPipeline


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Straight line between z0 and z1."""
    return (1 - t) * z0 + t * z1


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation: move along the arc between z0 and z1."""
    a, b = z0.flatten(), z1.flatten()
    omega = torch.acos(torch.clamp(a @ b / (a.norm() * b.norm()), -1.0, 1.0))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * z0 + (torch.sin(t * omega) / so) * z1


@torch.no_grad()
def ddim_sample(pipe: DDPMPipeline, noise: torch.Tensor, steps: int) -> list:
    """Run the reverse process of the pipeline's scheduler from the given noise."""
    pipe.scheduler.set_timesteps(steps)
    x = noise
    for t in pipe.scheduler.timesteps:
        eps = pipe.unet(x, t).sample  # predicted noise
        x = pipe.scheduler.step(eps, t, x).prev_sample  # DDIM: eta=0, deterministic
    x = (x / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
    return pipe.numpy_to_pil(x.permute(0, 2, 3, 1).cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/ddpm-cifar10-32")
    parser.add_argument("--seed-a", type=int, default=0, help="Seed of the start noise")
    parser.add_argument("--seed-b", type=int, default=1, help="Seed of the end noise")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Number of points along the path (including both ends)")
    parser.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim",
                        help="Sampler used to denoise each point on the path")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--mode", choices=["spherical", "linear"], default="linear",
                        help="How to interpolate between the two noise samples")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    pipe = DDPMPipeline.from_pretrained(args.model).to(device)
    if args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    size = pipe.unet.config.sample_size
    shape = (1, pipe.unet.config.in_channels, size, size)
    z0 = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed_a))
    z1 = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed_b))

    interp = slerp if args.mode == "spherical" else lerp
    ts = torch.linspace(0, 1, args.num_frames).tolist()
    noise = torch.cat([interp(z0, z1, t) for t in ts]).to(device)
    print(f"Noise std along the path: {[round(n.std().item(), 2) for n in noise]}")

    images = ddim_sample(pipe, noise, args.steps)

    fig, axes = plt.subplots(1, args.num_frames,
                             figsize=(2 * args.num_frames, 2.4), squeeze=False)
    for ax, img, t in zip(axes[0], images, ts):
        ax.imshow(img)
        ax.set_title(f"t={t:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{args.mode}: seed {args.seed_a} -> seed {args.seed_b}")
    fig.tight_layout()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "interpolation.png", dpi=150)
    print(f"Saved interpolation to {output}")
    plt.show()


if __name__ == "__main__":
    main()
