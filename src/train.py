"""Train a small class-conditional diffusion model on MNIST.

Usage:
    uv run src/train.py
    uv run src/train.py --epochs 10 --out outputs/my_run
"""

import argparse
from pathlib import Path

import torch
import torchvision
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import ClassConditionedUnet

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="Passes through the training set")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--out", default="outputs/training")
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset
dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# View some examples
x, y = next(iter(train_dataloader))
print("Input shape:", x.shape)
print("Labels:", y[:8])
plt.imshow(torchvision.utils.make_grid(x[:8])[0], cmap="Greys")
plt.savefig(out / "data.png", dpi=150)
plt.close()

# The forward process: adds noise to an image on a fixed schedule (nothing is learned here)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# The model that learns to predict the noise, given the noisy image, the timestep and the class
net = ClassConditionedUnet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []

for epoch in range(args.epochs):
    for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
        x = x.to(device) * 2 - 1  # map the images to (-1, 1)
        y = y.to(device)

        # 1. Corrupt the clean images: pick a random timestep per image and add the matching amount of noise
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # 2. Ask the model to predict the noise that was added, given the timestep and the class label
        pred = net(noisy_x, timesteps, y)

        # 3. The loss is how close the prediction is to the true noise
        loss = loss_fn(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:]) / 100
    print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

torch.save(net.state_dict(), out / "net.pt")

# View the loss curve
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.savefig(out / "loss.png", dpi=150)
plt.close()

# Sampling: start from pure noise and denoise step by step (the reverse process).
# We ask for 8 examples of each digit 0-9.
net.eval()
noise_scheduler.set_timesteps(1000)
x = torch.randn(80, 1, 28, 28).to(device)
y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
    with torch.no_grad():
        residual = net(x, t, y)  # the predicted noise, given the class labels y
    x = noise_scheduler.step(residual, t, x).prev_sample  # remove a little bit of it

# Show the results
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap="Greys")
ax.axis("off")
plt.tight_layout()
plt.savefig(out / "conditional_samples.png", dpi=150)
plt.close()
print(f"Saved data.png, loss.png, conditional_samples.png and net.pt to {out}")
