# Play Colmap

## Running

```bash
python pipeline.py --dataset storage/datasets/home --reset
python relocation.py --dataset storage/datasets/home --image storage/inputs/relocation_home.jpg --output storage/relocations/home
```

## Dense reconstruction

The pipeline ends with a dense reconstruction that runs entirely on CPU (COLMAP's
`patch_match_stereo` is CUDA-only), producing `storage/datasets/<name>/dense/fused.ply`.
Depth maps come from the plane-sweep MVS in `libs/cpu_mvs.py`, the fusion from pycolmap.

```bash
python pipeline.py --dataset storage/datasets/home --dense-max-size 1024  # denser, ~2.5x slower
```

Roughly 1.4s per image at the default `--dense-max-size 640` on an M4 (10 cores).
See [AGENTS.md](AGENTS.md) for the details and the remaining tuning flags.

## Gaussian splatting

The pipeline ends by training a 3D Gaussian Splatting scene on CPU (no CUDA
rasteriser) from the dense point cloud, writing `storage/datasets/<name>/splat/`:
`point_cloud.ply` in the standard 3DGS layout, `metrics.json`, and photo/render
comparisons under `renders/`. Tune it with `--splat-iterations`, `--splat-max-size`,
`--splat-max-gaussians`, `--splat-warmup`.

To retrain it, delete `splat/` (or pass `--reset`) and run the pipeline again — every
other step is skipped since its output already exists:

```bash
rm -rf storage/datasets/home/splat
python pipeline.py --dataset storage/datasets/home --splat-iterations 4000
```

Half of the iterations run on half resolution views (`--splat-warmup`), which costs a
quarter per step: on an M4 (10 cores) an iteration is 0.15s during the warmup and
0.7-0.8s after it, at the default 400 px and 60k gaussians.

Measured with the defaults (2000 iterations, CPU), photometric quality on the
training views and on a 1-in-8 holdout:

| dataset | gaussians | time | train | holdout |
|---|---|---|---|---|
| banana (14 views) | 60k | 11 min | 32.05 dB / 0.900 ssim | 26.54 dB / 0.819 |
| south-building (128 views) | 59k | 11 min | 21.28 dB / 0.697 ssim | 20.31 dB / 0.694 |

Large outdoor scenes stay blurry at these defaults: raise `--splat-max-gaussians` and
`--splat-iterations` (cost grows sublinearly in gaussians, linearly in iterations).

## Util links

- https://github.com/ruili3/awesome-dust3r
- https://github.com/naver/dust3r [https://www.perplexity.ai/search/running-colmap-dense-on-a-macb-l3mZtelGQxmjZizgbo49XA#1]
- https://github.com/cvg/Hierarchical-Localization
- https://github.com/freddewitt/CorbeauSplat?tab=readme-ov-file
- https://github.com/OpsiClear/DepthDensifier