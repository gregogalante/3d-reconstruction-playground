# AGENTS.md

Agent-facing notes for this repo. Project overview and run commands: see [README.md](README.md).

## Purpose

COLMAP playground: SfM + dense reconstruction from image sets, plus single-image
relocalisation against a reconstruction, served to a small web UI.

## Environment

- Python via Conda (`base` env). No system Python.
- `pycolmap` 3.13 (no CUDA on macOS), `opencv-python`, `numpy`, `scipy`, `open3d`.
- Do **not** import `torch` together with `pycolmap` in the same process: both link
  their own `libomp` and the process aborts (`OMP: Error #15`).

## Commands

```bash
python pipeline.py --dataset storage/datasets/home --reset
python relocation.py --dataset storage/datasets/home --image storage/inputs/relocation_home.jpg --output storage/relocations/home
python server.py
```

## Layout

- `pipeline.py` — SfM + dense pipeline, one function per step, all steps skip work
  that already exists on disk (`--reset` to rebuild).
- `libs/cpu_mvs.py` — CPU multi-view stereo (see below).
- `libs/read_write_model.py` — COLMAP model text/binary I/O (upstream script).
- `relocation.py` — localise a query image against an existing reconstruction.
- `server.py` + `ui/` — FastAPI backend and viewer.
- `storage/datasets/<name>/` — `train/` (input photos), `images/` (resized),
  `database.db`, `sfm/`, `dense/`, `config.json`. Git-ignored.

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
