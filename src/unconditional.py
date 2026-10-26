"""Unconditional image generation with a pre-trained diffusion model.

Usage:
    uv run src/unconditional.py
    uv run src/unconditional.py --model google/ddpm-celebahq-256 --num-images 4
    uv run src/unconditional.py --scheduler ddim --steps 50
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/ddpm-cifar10-32",
                        help="Unconditional model on the Hugging Face Hub")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampler used to run the reverse process")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Downloads the weights from the Hub the first time and caches them.
    pipe = DDPMPipeline.from_pretrained(args.model).to(device)

    # The scheduler defines how we step from noise to image. The trained network
    # (the UNet) stays the same, so we can swap in a faster sampler such as DDIM.
    if args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Fixing the seed fixes the initial noise, so results are reproducible.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    images = pipe(
        batch_size=args.num_images,
        num_inference_steps=args.steps,
        generator=generator,
    ).images  # list of PIL images

    # Plot the samples in a grid and save to disk.
    cols = min(4, args.num_images)
    rows = (args.num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
    fig.tight_layout()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "unconditional_samples.png", dpi=150)
    print(f"Saved {len(images)} images to {output}")
    plt.show()


if __name__ == "__main__":
    main()
