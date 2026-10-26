"""Class-conditional image generation with a pre-trained Diffusion Transformer (DiT).

The model was trained on ImageNet, so we can ask it for one of the 1000 ImageNet classes.

Usage:
    uv run src/conditional.py
    uv run src/conditional.py --labels "golden retriever" "volcano" --guidance-scale 8
    uv run src/conditional.py --list-labels
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/DiT-XL-2-256")
    parser.add_argument("--labels", nargs="+", default=["golden retriever", "white shark"],
                        help="ImageNet class names to generate")
    parser.add_argument("--num-per-label", type=int, default=2,
                        help="Number of images generated for each label")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="Classifier-free guidance strength (1.0 disables guidance)")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--list-labels", action="store_true",
                        help="Print the available class names and exit")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiTPipeline.from_pretrained(args.model, dtype=dtype).to(device)

    if args.list_labels:
        print("\n".join(sorted(pipe.labels)))
        return

    # DPM-Solver is a fast sampler, similar in spirit to DDIM: 25 steps are enough.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Turn the class names into the integer ids the model was trained with.
    class_ids = pipe.get_label_ids(args.labels)
    class_ids = [i for i in class_ids for _ in range(args.num_per_label)]

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    images = pipe(
        class_labels=class_ids,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
    ).images

    # One row per label.
    fig, axes = plt.subplots(len(args.labels), args.num_per_label,
                             figsize=(3 * args.num_per_label, 3.3 * len(args.labels)),
                             squeeze=False)
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.axis("off")
    for row, label in zip(axes, args.labels):
        row[0].set_title(label, fontsize=10, loc="left")
    fig.suptitle(f"guidance scale = {args.guidance_scale}")
    fig.tight_layout()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "conditional.png", dpi=150)
    print(f"Saved {len(images)} images to {output}")
    plt.show()


if __name__ == "__main__":
    main()
