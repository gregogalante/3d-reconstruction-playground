# AGENTS.md

Agent-facing notes for this repo. Project overview and run commands: see [README.md](README.md).

## Purpose

COLMAP playground: SfM + dense reconstruction from image sets, plus single-image
relocalisation against a reconstruction, served to a small web UI.

## Environment

- Python via Conda (`base` env). No system Python.
- `pycolmap` 3.13 (no CUDA on macOS), `opencv-python`, `numpy`, `scipy`, `open3d`.
- Do **not** import `torch` together with `pycolmap` in the same process: both link
  their own `libomp` and the process aborts (`OMP: Error #15`). `pipeline.py`
  (pycolmap) therefore launches `libs/splat_trainer.py` (torch) as a subprocess
  instead of importing it, and the trainer reads the COLMAP model through
  `libs/read_write_model.py`.

## Commands

```bash
python pipeline.py --dataset storage/datasets/home --reset   # sfm + dense + splat
python relocation.py --dataset storage/datasets/home --image storage/inputs/relocation_home.jpg --output storage/relocations/home
python server.py
```

## Layout

- `pipeline.py` — SfM + dense pipeline, one function per step, all steps skip work
  that already exists on disk (`--reset` to rebuild).
- `libs/cpu_mvs.py` — CPU multi-view stereo (see below).
- `libs/cpu_splatting.py` — CPU gaussian splatting: rasteriser, training, PLY export.
- `libs/splat_trainer.py` — runnable trainer (`python -m libs.splat_trainer`), launched
  by the pipeline's splat step. See below.
- `libs/console.py` — the coloured `print_*` helpers shared with the subprocess.
- `libs/ply.py` — numpy PLY vertex I/O, torch free so `server.py` can use it too.
- `libs/read_write_model.py` — COLMAP model text/binary I/O (upstream script).
- `relocation.py` — localise a query image against an existing reconstruction.
- `server.py` + `ui/` — FastAPI backend and viewer. `/api/datasets/<name>/clouds`
  reports which of the sparse, dense and splat clouds exist, the UI lets you switch
  between them and greys out the missing ones. The gaussians are served as
  `/splat.splat` (the 32 byte per gaussian layout the web renderer reads), converted
  on demand from the PLY and cached next to it.
- `storage/datasets/<name>/` — `train/` (input photos), `images/` (resized),
  `database.db`, `sfm/`, `dense/`, `splat/`, `config.json`. Git-ignored.

## Dense reconstruction on CPU

COLMAP's `patch_match_stereo` requires CUDA, so on macOS the dense step is replaced
by `libs/cpu_mvs.py`. Only the depth map estimation is custom, the surrounding
COLMAP steps are reused:

1. `pycolmap.undistort_images` → `dense/` workspace (pinhole images + sparse model).
2. `cpu_mvs.build_depth_maps` → `dense/stereo/{depth_maps,normal_maps}`:
   - source views per image ranked by co-visible points weighted by triangulation angle,
   - plane sweep over ~128 fronto-parallel planes uniform in inverse depth,
     ZNCC over a 7×7 window, cost aggregated over the 3 most consistent sources,
   - sub-plane parabolic refinement, then a cross-view geometric consistency filter
     (reprojection error and relative depth error) producing the `.geometric.bin` maps.
3. `pycolmap.stereo_fusion` → `dense/fused.ply`.

Maps use COLMAP's own binary format (`width&height&channels&` header, slice-major
data), so the workspace stays interchangeable with a real COLMAP install.

Tuning (`python pipeline.py --help`): `--dense-max-size` drives density and cost
quadratically, `--dense-num-samples` the depth resolution, `--dense-num-src` the
number of matched views, `--dense-num-workers` the parallelism. Existing maps are
never recomputed, so an interrupted run resumes.

## Gaussian splatting on CPU

The last pipeline step trains a 3DGS scene from the dense reconstruction without CUDA,
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

`--splat-capacity` trades fidelity for speed: measured against a near-exact render,
64 is ~33 dB and 96 ~44 dB, at 1.4x the cost. `--splat-max-size` (training resolution)
drives cost linearly in pixels, and `--splat-max-gaussians` should follow it: 60k
gaussians already match a 120k render at 400 px. `--splat-device mps` runs the same
code on the Apple GPU, ~2.5x faster than CPU (0.30s vs 0.73s per iteration at 400 px
and 60k gaussians).

Quality with the defaults is in [README.md](README.md). Object-scale scenes converge
well; large outdoor scenes need more gaussians and iterations than the defaults.
