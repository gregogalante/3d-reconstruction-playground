# 3D Reconstruction Playground

A COLMAP playground that runs end to end on a Mac, with no CUDA anywhere: capture a
dataset with an Android phone, reconstruct it (sparse, then dense on CPU, optionally a
gaussian splat), localise single photos against the result, and look at all of it in a
browser.

```
capture_server.py + capture/   a phone fills storage/datasets/<name>/train/
pipeline.py                    train/ becomes a reconstruction
relocation.py                  where was this photo taken from?
viewer_server.py + viewer/     the result, in a browser
```

## Getting a dataset

`storage/` is empty in a fresh clone — every dataset is git-ignored, including the ones
named in the measurements further down, which are local captures. So the first step is
always to make one. A dataset is just a folder with a `train/` in it:

```
storage/datasets/<name>/train/    photos, or a single .mov/.mp4/.m4v
```

Three ways to fill it, in the order they are worth trying:

**From your phone.** Start the capture server, scan the QR it prints, and walk around
the subject while the page tells you where to go. Frames and ARCore poses land straight
in `train/`. Needs Chrome on Android with ARCore; see [docs/capture.md](docs/capture.md)
for why it serves HTTPS and what the guidance enforces.

```bash
python capture_server.py
```

**From photos you already have.** Drop them in and go. Shots have to overlap and come
from different places — a ring of photos around a thing, not a panorama from one spot.

```bash
mkdir -p storage/datasets/kitchen/train && cp ~/Pictures/kitchen/*.jpg storage/datasets/kitchen/train/
```

**From a video.** A `train/` holding a clip instead of photos is cut into as many even
windows as the cap allows, each giving up its sharpest frame, so the scene is covered
evenly and the frames motion blur ruined are skipped. Everything after that is identical.

```bash
mkdir -p storage/datasets/kitchen/train && cp ~/Movies/kitchen.mov storage/datasets/kitchen/train/
```

Public datasets work too — COLMAP's own `south-building` is what several numbers below
were measured on.

## Running the pipeline

```bash
python pipeline.py --dataset storage/datasets/kitchen --reset
```

Every step skips work already on disk, so re-running is cheap and `--reset` is how you
force a rebuild. In order: resize the photos into `images/`, extract features, match every
pair, build the sparse model and export it three ways (PLY, COLMAP text, and a
`transforms.json` for NeRF tooling), undistort into the dense workspace, sweep the depth
maps, fuse them into `dense/fused.ply`, and finish with the relocalisation check.

`IMAGE_MAX_ITEMS` in [pipeline.py](pipeline.py) caps how many photos of `train/` are
used, 300 by default. Matching is quadratic in their number, so a denser capture costs
much more than it adds; over the cap the photos are decimated uniformly (400 capped at
300 keeps three and skips one) instead of cutting the tail, which would leave a hole in
the scene. Set it to 0 to use them all.

If a capture comes back as several models instead of one, `MAPPER_MIN_INLIERS` and its
two neighbours are the reason more often than the scene is — see
[docs/reconstruction.md](docs/reconstruction.md).

## The two servers

Each is named after the page it serves: `<name>_server.py` serves `<name>/`.

```bash
python capture_server.py    # capture/ — https://<lan-ip>:8443, a QR of it in the terminal
python viewer_server.py     # viewer/  — http://localhost:8000
```

The viewer is a React app built by Vite; it is served from `viewer/dist`, so build it
once after cloning:

```bash
cd viewer && yarn install && yarn build
```

## Dense reconstruction

The dense step runs entirely on CPU (COLMAP's `patch_match_stereo` is CUDA-only),
producing `storage/datasets/<name>/dense/fused.ply`. Depth maps come from the
plane-sweep MVS in `libs/cpu_mvs.py`, the fusion from pycolmap.

```bash
python pipeline.py --dataset storage/datasets/kitchen --dense-max-size 1024  # denser, ~2.5x slower
```

Roughly 1.4s per image at the default `--dense-max-size 640` on an M4 (10 cores).
See [docs/reconstruction.md](docs/reconstruction.md) for the details and the remaining
tuning flags.

## Gaussian splatting

There is a CPU 3D Gaussian Splatting trainer (no CUDA rasteriser) that turns the dense
cloud into `storage/datasets/<name>/splat/`: `point_cloud.ply` in the standard 3DGS
layout, `metrics.json`, and photo/render comparisons under `renders/`.

**It is commented out of the step list in [pipeline.py](pipeline.py)** — it costs more
than the rest of the pipeline put together, and most runs do not want it. Uncomment the
`Build Gaussian Splat` line to put it back, and tune it with `--splat-iterations`,
`--splat-max-size`, `--splat-max-gaussians`, `--splat-warmup`.

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
python relocation.py --dataset storage/datasets/kitchen --image ~/Pictures/query.jpg --output storage/relocations/kitchen
```

Measured on queries built from dataset photos re-encoded at 75% scale, gamma 1.35 and
JPEG 70, under a neutral name so nothing can be recognised — the reconstruction pose
is then exact ground truth. Position error as a percentage of the scene radius:

| dataset | before | now | time |
|---|---|---|---|
| banana (14 views) | 33.2% / 2.80° | 0.03% / 0.009° | 2.0s |
| banana (65 views) | 136.6% / 5.83° | 0.23% / 0.051° | 1.5s |
| an office, 205 views | 35.2% / 6.38° | 0.05% / 0.023° | 1.5s |
| south-building (128 views) | 20.1% / 4.43° | 0.01% / 0.009° | 6s |
| the same office, 416 views | 32.8% / 5.96° | 0.03% / 0.017° | 2.0s |

Every query lands within 2% of the scene radius and 2°, none did before. See
[docs/relocalisation.md](docs/relocalisation.md) for what changed and what did not help.

The viewer does the same thing without the command line: pick a dataset, hit **Locate a
photo** in the Relocations panel, and the upload comes back as a red frustum in the
scene with its overlay opened. A photo the dataset does not contain is rejected rather
than placed somewhere plausible.

### How good is this model at being relocalised against?

The last pipeline step answers that without you having to ask: it hides a spread of the
dataset's own images, one at a time, localises each against what is left, and pushes
until it fails. The report lands in `storage/datasets/<name>/relocation.json`.

It reports **margins, not accuracy**, because accuracy does not discriminate — every
healthy dataset places a held-out photo within a hundredth of a percent. What varies is
how far from the mapped path a query can be before it stops registering at all, so each
holdout is pushed away (hiding its nearest views) and degraded (resolution, blur, JPEG)
until the pose leaves tolerance. `--relocation-stride 0` skips the whole check.

## A note on the numbers above

Two are named because the name means something. `south-building` is COLMAP's own public
dataset, so those rows can be reproduced from scratch; `banana` is a 14 photo capture of
an object, still on the machine these were measured on but not in the repo, since
`.gitignore` keeps all of `storage/` out of it.

Everything else is described rather than named — an office, a flat, the phone captures of
white kitchen walls — because those folders have since been deleted and a name nobody can
obtain is not a reference. The measurements stay so the reasoning behind a default can be
checked, not so it can be re-run on a fresh clone.

## Where the details are

[AGENTS.md](AGENTS.md) is the map: environment, layout and conventions, pointing at one
file per domain in [docs/](docs) — [capture](docs/capture.md),
[reconstruction](docs/reconstruction.md), [relocalisation](docs/relocalisation.md),
[splatting](docs/splatting.md). Each records what was measured, including what failed.
