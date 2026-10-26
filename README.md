# CIM Lab 4: Diffusion Models

## Introduction

Today we will learn about diffusion models. Diffusion models are a class of generative models that have superseded GANs as the state of the art in image generation, and they have wide utility in fields like inverse problems, where they are used as versatile priors for arbitrary domains.

## Setup

Install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Image generation with Diffusion Models

We will begin with simple image generation using a pre-trained model. Training and running diffusion models has become much easier with the advent of Hugging Face's [diffusers library](https://huggingface.co/docs/diffusers/index).

The script `src/unconditional.py` downloads a pre-trained model from the [Hugging Face Hub](https://huggingface.co/models?other=diffusers:DDPMPipeline), starts from pure Gaussian noise and iteratively denoises it to produce new images.

We will start with [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32), a small model trained on CIFAR-10 (32x32 images of ten classes such as cars, horses and ships). It runs in reasonable time on a laptop and is the default model, so this is the first command to run:

```bash
uv run src/unconditional.py
```

The first run downloads the weights, and the generated images are saved to `outputs/unconditional_samples.png`. Note that the model is *unconditional*: we cannot ask it for a particular class; it simply samples from the distribution it was trained on. The script has a few options you can play with:

```bash
# Different random noise gives different images
uv run src/unconditional.py --seed 42

# Generate more images
uv run src/unconditional.py --num-images 16
```

Run `uv run src/unconditional.py --help` to see all options.

### Additional model: CelebA-HQ

If you want to see what a larger model can do, [`google/ddpm-celebahq-256`](https://huggingface.co/google/ddpm-celebahq-256) was trained on 256x256 face images. It is much slower than the CIFAR-10 model, so this is optional, and a GPU helps (or try it with the faster sampler from the next section).

```bash
uv run src/unconditional.py --model google/ddpm-celebahq-256 --num-images 4
```

## Faster sampling with DDIM

The default DDPM sampler needs 1000 denoising steps, one network evaluation each, which is slow. [DDIM](https://arxiv.org/abs/2010.02502) (Denoising Diffusion Implicit Models) is a different way of stepping through the reverse process that can skip steps. It works with the *same* pre-trained network, so no retraining is needed: only the scheduler changes.

```bash
uv run src/unconditional.py --scheduler ddim --steps 50
```

This should be roughly 20x faster than the default run. Try different numbers of steps (e.g. `--steps 10`, `--steps 25`, `--steps 100`) with the same `--seed` and compare the images and the run time. How few steps can you use before the quality degrades?

## Interpolating in noise space

Because DDIM is deterministic, every initial noise tensor $x_T$ maps to exactly one image. This means the initial noise acts like a latent code: if we pick two noise samples and move between them, we can watch the generated image change.

The script `src/interpolate.py` draws two noise tensors (from `--seed-a` and `--seed-b`), builds a path between them, denoises every point on the path with DDIM, and plots the results in a row.

```bash
uv run src/interpolate.py
```

The image is saved to `outputs/interpolation.png`. You can choose the endpoints and the number of frames:

```bash
uv run src/interpolate.py --seed-a 3 --seed-b 7 --num-frames 12
```

There are two ways to interpolate between noise samples $z_0$ and $z_1$:

- **Linear (`--mode linear`)**: $z_t = (1-t)\,z_0 + t\,z_1$
- **Spherical (`--mode spherical`, default)**: $z_t = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,z_0 + \frac{\sin(t\Omega)}{\sin\Omega}\,z_1$, where $\Omega$ is the angle between $z_0$ and $z_1$

```bash
uv run src/interpolate.py --mode linear
```

Compare the two. The script also prints the standard deviation of the noise along the path. What do you notice about the images in the middle of the linear path, and how does this relate to the printed standard deviation? (Hint: the network was trained on noise with standard deviation 1. What happens to the norm of the average of two independent Gaussian vectors?)

**Try it with DDPM.** The script can also use the original DDPM sampler:

```bash
uv run src/interpolate.py --scheduler ddpm --steps 1000
```

This will *not* give a smooth path, even with spherical interpolation. Run it a few times, including twice with identical arguments. Why does it fail? Look at what the DDPM update does at each step compared to DDIM, and think about what it means to say that the initial noise "determines" the image.

## Conditional generation

So far the model has decided by itself what to generate. In conditional generation we also give the model a *condition* (here a class label), and it generates an image of that class. The script `src/conditional.py` uses [`facebook/DiT-XL-2-256`](https://huggingface.co/facebook/DiT-XL-2-256), a class-conditional model trained on the 1000 classes of ImageNet.

> **Note:** this model is much bigger than the CIFAR-10 one. The first run downloads about 3 GB of weights, and generation is slow on a CPU.

```bash
uv run src/conditional.py
```

This generates two images each of a golden retriever and a white shark and saves them to `outputs/conditional.png`. Choose your own classes with `--labels`, and use `--list-labels` to print all available class names:

```bash
uv run src/conditional.py --labels "golden retriever" "volcano" "espresso" --num-per-label 3
uv run src/conditional.py --list-labels
```

### Classifier-free guidance

The `--guidance-scale` option controls how strongly the model is pushed towards the requested class. During training the label is randomly dropped, so the same network learns both a conditional prediction $\epsilon_\theta(x_t, c)$ and an unconditional one $\epsilon_\theta(x_t, \varnothing)$. At sampling time we combine them as

$$\hat\epsilon = \epsilon_\theta(x_t, \varnothing) + w\,\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big),$$

where $w$ is the guidance scale. With $w = 1$ there is no guidance.

```bash
uv run src/conditional.py --guidance-scale 1
uv run src/conditional.py --guidance-scale 4
uv run src/conditional.py --guidance-scale 15
```

Keep the same `--seed` and compare. What happens to the quality, the diversity and the saturation of the images as the guidance scale grows? Why might the model produce *less* varied images when it is pushed harder towards the class?

## Training a diffusion model

So far we have used models that someone else trained. Now we will train our own small one, a class-conditional model that generates MNIST digits. A diffusion model has two components:

- a fixed *forward process* that progressively adds noise to an image on a fixed schedule (nothing is learned here)
- a learned *network* that is trained to undo this corruption, so that it can progressively denoise an image from pure noise to a data point

Before training, we look at the key ideas one at a time. The first run of each script downloads MNIST into `data/`, and all figures are saved to `outputs/`.

### 1. The forward process

```bash
uv run src/forward_process.py
```

This takes a few digits and corrupts them with the forward process at timesteps $t = 0, 100, 250, 500, 750, 999$ (saved to `outputs/forward_process.png`). The same noise is used at every timestep, so you see *one* image being gradually destroyed. Because the noise added at each step is Gaussian and sums of Gaussians are Gaussian, we can jump straight to any timestep $t$ in one step:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

The script checks this formula against the `add_noise` function from diffusers, and plots the signal weight $\sqrt{\bar\alpha_t}$ and noise weight $\sqrt{1-\bar\alpha_t}$ against $t$ in `outputs/noise_schedule.png`. Try other timesteps with `--timesteps`, e.g. `--timesteps 0 20 40 60 80 100`.

- At which timestep can you no longer recognise the digit? How much signal is left at that point?
- Is the noise schedule linear in $t$?

### 2. The network predicts the noise

The network is a UNet (`ClassConditionedUnet` in `src/models.py`). Given a noisy image $x_t$ and the timestep $t$, it is trained to predict the noise $\epsilon$ that was added. One training step (see `src/train.py`) is just three lines:

```python
noisy_x = noise_scheduler.add_noise(x, noise, timesteps)  # 1. corrupt a clean image
pred = net(noisy_x, timesteps, y)                          # 2. predict the noise
loss = loss_fn(pred, noise)                                # 3. compare with the true noise
```

At sampling time we start from pure noise and repeatedly ask the network "what noise is in this image?", remove a little bit of it, and repeat for 1000 steps (this is the reverse process you used in the previous sections).

- Why do you think we predict the noise rather than the clean image directly?
- The timestep is picked at random for each image in a batch. Why not train on one timestep at a time?

### 3. The time embedding

The same network has to denoise images at every noise level, so it needs to know what $t$ is. Passing the integer $t$ directly would work poorly, so it is first turned into a vector of sines and cosines at different frequencies (the same idea as positional encodings in transformers). Inside the UNet, this vector goes through a small MLP and is added to the features in every ResNet block.

```bash
uv run src/time_embedding.py
```

The left plot in `outputs/time_embedding.png` shows the embedding of each timestep; the right plot shows how similar the embeddings of two timesteps are.

- Why are the embeddings of nearby timesteps similar? Why do we need several frequencies?

### 4. The class embedding (optional)

To make the model conditional, `ClassConditionedUnet` maps the class label (0-9) to a learned vector of size 4 with `nn.Embedding`, copies it to every pixel, and appends it to the image as 4 extra input channels. The network can then use the label to decide which digit to produce. (This is one of the simplest ways to condition a model; the DiT model in the previous section also embeds the label, but injects it differently.)

- The class embedding is *learned*, while the time embedding is *fixed*. Why might that be?
- How would you make the model generate without a class (hint: classifier-free guidance)?

### 5. Train the model

```bash
uv run src/train.py --epochs 5
```

This trains the model for 5 passes through the 60000 MNIST images, then generates 8 examples of each digit starting from pure noise.

> **Note:** On a GPU each epoch should take no more than a couple of minutes, and the final sampling about one more minute. **A CPU will be considerably slower**. Start it, then read on or work on the questions while it runs.

The outputs are written to `outputs/training/`:

- `data.png`: some training examples
- `loss.png`: the training loss
- `conditional_samples.png`: generated digits, one row per class (0 to 9)
- `net.pt`: the trained weights

Look at `conditional_samples.png` after 1 epoch and after 5 epochs (`--epochs 1 --out outputs/one_epoch`). How does the quality change? Does the model respect the class label? (After 1 epoch you should see digit-like strokes that mostly ignore the row's label; after 5 epochs most rows contain recognisable digits of the right class, with a few messy samples left.)

## Questions to ponder for next week

- How could we add additional information into the reverse diffusion process to affect the outcome of the generation process?
