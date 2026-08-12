# Splatting

The CPU 3D Gaussian Splatting trainer: how it differs from the reference
implementation, and what it costs.

Back to [AGENTS.md](../AGENTS.md).

## Gaussian splatting on CPU

**The splat step is commented out of the step list in `pipeline.py`.** It costs more than
everything above it put together and most runs do not want it; uncomment the
`Build Gaussian Splat` line to put it back. What follows is what it does when you do.

It trains a 3DGS scene from the dense reconstruction without CUDA,
using the differentiable rasteriser in `libs/cpu_splatting.py` (plain PyTorch, autograd
for the backward pass). Output: `storage/datasets/<name>/splat/point_cloud.ply` in the
original 3DGS layout, plus `metrics.json` and photo/render comparisons. Delete `splat/`
to retrain, the other steps stay skipped.

How it differs from the reference implementation, and why:

- gaussians are initialised on the dense MVS points (position, colour, and scale from
  the local point spacing), so **no densification** is needed — only pruning of the
  transparent and oversized ones,
- colours are **view-independent** (SH degree 0): higher bands cost CPU time for
  view-dependent highlights,
- the rasteriser groups pixels in small tiles, each compositing its `capacity`
  closest gaussians in chunks, dropping tiles whose transmittance is spent. Pairs
  that cannot reach 1/255 anywhere in their tile are culled before ranking. Both
  bound the work to the real depth complexity — a fixed-capacity `(tiles, capacity,
  pixels)` tensor is what makes the whole pass vectorised and autograd friendly.

`--splat-warmup` trains the first half of the iterations on half resolution views,
where a step costs a quarter. Measured at 800 iterations against the same run without
it: banana 271s vs 412s with holdout 25.14 vs 25.13 dB, south-building 232s vs 340s
with holdout 17.99 vs 18.04 dB. A third of the time for holdout quality inside the
noise, so it is on by default.

`--splat-capacity` trades fidelity for speed: measured against a near-exact render,
64 is ~33 dB and 96 ~44 dB, at 1.4x the cost. `--splat-max-size` (training resolution)
drives cost linearly in pixels, and `--splat-max-gaussians` should follow it: 60k
gaussians already match a 120k render at 400 px.

### Measured dead ends

Keep these from being retried:

- **`--splat-device mps` is ~3x slower than CPU**, despite a single render plus
  backward being 2.5x faster in isolation (0.30s vs 0.73s). The compositing loop drops
  saturated tiles between chunks, which reads a tensor back to the host and stalls the
  Metal queue every chunk. Results are identical to the CPU ones (20.70 dB both), so
  the flag stays, but it only pays off if the tile culling stops syncing.
- **Decaying the position learning rate makes it worse**, unlike the reference
  implementation: 29.08 dB train / 22.66 holdout against 30.91 / 25.13 at 800
  iterations. That schedule spans 30k iterations; compressed into a couple of thousand
  it starves the positions long before they settle.
- **Coarse to fine depth in `cpu_mvs` is incompatible with this cost function.**
  Warping every pixel with its own depth breaks the 7x7 ZNCC window: the patch is no
  longer a coherent piece of the source, so the cost rises even where the depth is
  right. Starting the refinement from the 128 plane sweep result leaves the depth
  unchanged on 93% of the pixels yet drops coverage from 92% to 44%. Doing it properly
  needs per-pixel patch sampling (49 remaps per candidate, worse than 128 planes).
  Fewer planes is not free either: coverage goes 94.5% at 256 planes, 92.1% at 128,
  90.2% at 96, 86.4% at 64.

Quality with the defaults is in [README.md](../README.md). Object-scale scenes converge
well; large outdoor scenes need more gaussians and iterations than the defaults.
