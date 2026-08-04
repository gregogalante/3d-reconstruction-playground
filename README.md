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

## Relocalisation

`relocation.py` estimates where a single photo was taken from, against an existing
reconstruction. It ranks the database images by descriptor votes, matches the query
against the best ones, and solves PnP on the resulting 2D-3D correspondences
(`libs/localizer.py`). Output next to the JSON: a `_overlay.jpg` with the query on one
side and the model reprojected from the estimated pose on the other, joined by one
line per inlier. The lines are parallel when the pose is right, and coloured green to
red by reprojection error. It is the only way to judge a pose without ground truth,
and the viewer opens it from the relocation list.

```bash
python relocation.py --dataset storage/datasets/home --image storage/inputs/query.jpg --output storage/relocations/home
```

Measured on queries built from dataset photos re-encoded at 75% scale, gamma 1.35 and
JPEG 70, under a neutral name so nothing can be recognised — the reconstruction pose
is then exact ground truth. Position error as a percentage of the scene radius:

| dataset | before | now | time |
|---|---|---|---|
| banana (14 views) | 33.2% / 2.80° | 0.03% / 0.009° | 2.0s |
| home (65 views) | 136.6% / 5.83° | 0.23% / 0.051° | 1.5s |
| over-office-2 (205 views) | 35.2% / 6.38° | 0.05% / 0.023° | 1.5s |
| south-building (128 views) | 20.1% / 4.43° | 0.01% / 0.009° | 6s |
| over-office-1 (416 views) | 32.8% / 5.96° | 0.03% / 0.017° | 2.0s |

Every query lands within 2% of the scene radius and 2°, none did before. See
[AGENTS.md](AGENTS.md) for what changed and what did not help.

## Util links

- https://github.com/ruili3/awesome-dust3r
- https://github.com/naver/dust3r [https://www.perplexity.ai/search/running-colmap-dense-on-a-macb-l3mZtelGQxmjZizgbo49XA#1]
- https://github.com/cvg/Hierarchical-Localization
- https://github.com/freddewitt/CorbeauSplat?tab=readme-ov-file
- https://github.com/OpsiClear/DepthDensifier