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
  that already exists on disk (`--reset` to rebuild). Two constants at the top:
  `IMAGE_MAX_DIMENSION` and `IMAGE_MAX_ITEMS`, the cap on how many photos of `train/`
  reach `images/`. Over the cap the capture is decimated uniformly rather than cut
  short (400 photos capped at 300 keep three and skip one), so the scene stays
  covered; 0 disables it. Both land in `config.json`, and changing either needs
  `--reset` since `build_images` skips a non empty `images/`.
- `libs/cpu_mvs.py` — CPU multi-view stereo (see below).
- `libs/cpu_splatting.py` — CPU gaussian splatting: rasteriser, training, PLY export.
- `libs/splat_trainer.py` — runnable trainer (`python -m libs.splat_trainer`), launched
  by the pipeline's splat step. See below.
- `libs/console.py` — the coloured `print_*` helpers shared with the subprocess.
- `libs/ply.py` — numpy PLY vertex I/O, torch free so `server.py` can use it too.
- `libs/read_write_model.py` — COLMAP model text/binary I/O (upstream script).
- `libs/colmap2nerf.py` — COLMAP model to `transforms.json` (upstream instant-ngp
  script, kept as is). Run as `python -m libs.colmap2nerf`: it has no main guard, so
  importing it would execute it.
- `relocation.py` — localise a query image against an existing reconstruction (see below).
- `libs/localizer.py` — the retrieval, matching and PnP behind it.
- `server.py` + `ui/` — FastAPI backend and viewer. `/api/datasets/<name>/clouds`
  reports which of the sparse, dense and splat clouds exist, the UI lets you switch
  between them and greys out the missing ones. The gaussians are served as
  `/splat.splat` (the 32 byte per gaussian layout the web renderer reads), converted
  on demand from the PLY and cached next to it. The conversion pre turns every
  gaussian by half a turn around x, because drei's `Splat` displays the rows turned
  that way (it negates the center z on read, the center y on upload, and decodes the
  quaternion to match). Without it the gaussians sit rotated against the point clouds
  and the camera frustums, which are drawn straight from the COLMAP frame. Delete the
  cached `.splat` files after touching the conversion, the cache only tracks the PLY
  timestamp.
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

Workers default to CPU count + 2: each one alternates between single threaded numpy
and multi threaded OpenCV, and the slight oversubscription fills the gaps (1.32s vs
1.43s per image on 10 cores).

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

Quality with the defaults is in [README.md](README.md). Object-scale scenes converge
well; large outdoor scenes need more gaussians and iterations than the defaults.

## Relocalisation

`relocation.py` walks `libs/localizer.py` through four stages. Nothing is cached: the
descriptor index builds from `database.db` in well under a second even on the 416
image dataset, so there is no pipeline artifact to keep in sync.

1. **Rank** the database images. The query is matched against a descriptor index built
   from the tracks (4 descriptors per 3D point, spread over the track) and every match
   votes for the images that observe its point. The runner up of the ratio test has to
   belong to a *different* 3D point, otherwise a point's own sibling descriptors reject
   every correct match. Only a ranking is needed, so 4000 query descriptors are enough.
2. **Match** the query against the top images one by one, mutual nearest neighbours
   plus ratio test. Database keypoints without a 3D point are dropped before matching.
3. **Solve** PnP with LO-RANSAC at 4 px, then refine pose *and* intrinsics.
4. **Report** inliers, mean reprojection error and a two panel `_overlay.jpg`: the
   query on the left, the model reprojected from the estimated pose on the right, and
   a line across the panels per inlier, from the keypoint to where its 3D point lands.
   Both panels share the viewpoint, so the lines are parallel when the pose is right
   and fan out when it is not; they run green to red over the RANSAC threshold. The UI
   opens it from the relocation list.

A pose under `min_inliers` (30, COLMAP's own threshold for registering an image) is
reported as a failure: RANSAC always finds some minimal set that agrees, so a photo of
another scene still comes back with a pose. Localising a `home` photo against `banana`
gives 5 inliers out of 97 and used to be stored as a success.

`relocation.relocate()` is the whole flow as one call, shared by the CLI and by
`POST /api/datasets/<name>/relocate`, which the viewer's "Locate a photo" button posts
an upload to. The endpoint writes into `storage/relocations/<dataset>/` exactly like
the CLI, deletes what it wrote when the pose is rejected, and answers 422 with the
reason. Importing `relocation` from `server.py` is safe, neither pulls in torch.

What actually mattered, measured against the table in [README.md](README.md):

- **Matching per image instead of against the whole model.** The ratio test is
  meaningless against 40k points that all look alike; it only survived at 0.5, leaving
  a few dozen correspondences. Per image it gives 3 to 10 times more, at 0.8.
- **Not trusting `infer_camera_from_image`.** Without a known sensor COLMAP falls back
  to a focal of 1.2 x the largest side. On these phone photos the reconstruction says
  0.76, so the old code localised with a 56% focal error and put the camera metres
  away. The calibrated cameras of the reconstruction now win over any EXIF guess.
- **Refining the focal and the distortion with the pose.** The reconstruction gives one
  camera per image and they disagree by a few percent, which is a few percent of error
  along the view direction. Solving for them takes the median position error on `home`
  from 3.3% of the scene radius to 0.23%, and on `over-office-1` from 0.08% to 0.02%.
  Refining the focal alone is not enough (`home` stops at 1.08%).

### Measured dead ends

- **Dense depth adds correspondences but not accuracy.** Lifting the matched keypoints
  that were never triangulated (`--dense`, `DenseDepth` in `libs/localizer.py`) gives
  50 to 90% more correspondences and changes the pose by less than the noise: `home`
  0.33% of radius against 0.23% without, `over-office-1` 0.02% against 0.04%, at 1.5x
  the matching cost. The MVS depths are simply less accurate than the triangulated
  points. Kept as a flag, off by default.
- **The gaussians have nothing to offer here.** Photometric refinement against a render
  would need the torch subprocess and a differentiable pose, to improve on a PnP that
  already lands within 0.03% of the scene radius and 0.02 degrees.
- **kd-trees on 128 dimensional descriptors.** `scipy.cKDTree` degenerates to a full
  scan and is 5x slower than one matrix product. Matching is brute force, one product
  per pair, with the reverse direction taken as a second reduction over the same block
  (that makes the mutual check nearly free). Two masked `argmin` passes beat
  `argpartition` by 3.5x on matrices this wide.
