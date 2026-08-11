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
python server.py            # the viewer, http://localhost:8000
python capture_server.py    # capture from a phone, https://<lan-ip>:8443
```

## Layout

- `pipeline.py` — SfM + dense pipeline, one function per step, all steps skip work
  that already exists on disk (`--reset` to rebuild). Two constants at the top:
  `IMAGE_MAX_DIMENSION` and `IMAGE_MAX_ITEMS`, the cap on how many photos of `train/`
  reach `images/`. Over the cap the capture is decimated uniformly rather than cut
  short (400 photos capped at 300 keep three and skip one), so the scene stays
  covered; 0 disables it. Both land in `config.json`, and changing either needs
  `--reset` since `build_images` skips a non empty `images/`. A `train/` holding no
  photos but a `.mov`/`.mp4`/`.m4v` is split into frames instead, see below.
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
- `capture_server.py` + `capture/` — the phone capture server and its page (see below).
  Free of pycolmap and torch on purpose: it only writes files, and stays startable while
  a pipeline is busy. `capture/lib/` is plain ES modules, JavaScript Standard Style, and
  the guidance in it is the one part with tests (`node --test 'capture/lib/*.test.js'`).
- `relocation.py` — localise a query image against an existing reconstruction (see below).
- `libs/localizer.py` — the retrieval, matching and PnP behind it.
- `libs/evaluation.py` — leave one out relocalisation over a dataset's own images, run
  as the last pipeline step and written to `relocation.json` (see below).
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
  `database.db`, `sfm/`, `dense/`, `splat/`, `config.json` (the constants and CLI
  options of the run), `time.json` (seconds per step, rewritten every run: a step whose
  output was already on disk reads as ~0). Git-ignored.

## Capturing from a phone

`capture_server.py` plus the static page in `capture/` turn an Android phone into the
camera of the pipeline. The phone opens the page over the local network, walks a guided
capture, and the frames land in `storage/datasets/<name>/train/` with the ARCore poses
beside them in `capture.json`. Nothing downstream changes: `pipeline.py` reads that
`train/` like any other.

```bash
python capture_server.py                 # TLS on 8443, prints a QR of the LAN address
python capture_server.py --http --port 8444   # localhost or adb reverse only
node --test 'capture/lib/*.test.js'      # the guidance rules
```

### Why it insists on HTTPS

The guidance rides on WebXR, which Chrome on Android implements over ARCore, and both
WebXR and the camera are handed out to secure contexts only. `http://192.168.x.y` is not
one. So the server generates a self signed certificate carrying the LAN address as a
subject alternative name — without a matching SAN Chrome refuses the origin instead of
offering the warning you can click through — and serves TLS. Three ways in, in order of
how much they hurt:

1. **USB**: `adb reverse tcp:8443 tcp:8443`, then `https://localhost:8443` on the phone.
   localhost is trusted, so no warning at all.
2. **Wi-Fi**: scan the QR, accept the certificate warning once (Advanced → Proceed).
3. `chrome://flags/#unsafely-treat-insecure-origin-as-secure` with the plain address, if
   the certificate becomes a nuisance.

The key lives in `storage/certs/`, which is git ignored — check that it stays that way.

### What the guidance actually enforces

A capture dies of three things, and each has a rule in `capture/lib/guidance.js`:

- **Rotation without translation.** A panorama has no parallax, so nothing triangulates.
  In orbit mode the shutter is gated on the angle *around the subject*, so standing still
  and turning earns nothing however long you wait; in walk mode turning past 25° having
  moved less than 12 cm raises a warning by name.
- **Gaps.** Shots are taken every `ORBIT.angularStep` (12°) around the subject, which is
  the overlap two neighbouring views need, and the ring targets tell you where to walk
  next. Coverage is the fraction of targets a shot has passed within 18° of.
- **Blur.** Every frame is scored by the variance of its Laplacian — the same measure
  `pipeline.py` uses to pick frames out of a video — and dropped if it falls under 55% of
  the running median of this capture. Relative, because a white wall scores low while
  perfectly in focus.

Two rings of twelve targets at 8° and 28° of elevation, because one ring reconstructs a
band and leaves the top of the object a guess. The radar in the corner is the map of what
is covered; the arrow points at the nearest target you have not.

### Three ways it can capture

| path | pose | frames | when |
|---|---|---|---|
| WebXR `immersive-ar` + `camera-access` | ARCore, 6DoF | the camera texture, read back through a framebuffer | Chrome on Android with ARCore |
| `getUserMedia` | none | `<video>` drawn to a canvas | anything else, guidance falls back to blur and count |
| simulator (`?simulate=1`) | fabricated orbit | a painted canvas, every 8th blurred | testing without a phone |

They share one `shoot()` so the fallbacks cannot rot unnoticed. The simulator takes a
`stepBy(ms)` on its own clock, which is how a whole capture can be replayed in a hidden
tab where `requestAnimationFrame` never fires; `window.captureDebug` is published in that
mode only.

### What lands on disk

`train/capture_00000.jpg` upwards, named so that sorting them replays the capture, and
`capture.json` holding per frame: the WebXR pose (position, orientation, matrix), the
intrinsics derived from the projection matrix *plus the raw matrix*, both sharpness
scores (the phone's and the server's), and the size. Written after every frame through a
temporary file, so a phone that walks out of range still leaves a readable manifest.

The poses are stored, not used. They are in the WebXR frame — right handed, +Y up, camera
down -Z — and COLMAP is the opposite on both counts; the `convention` field in the
manifest is what a conversion has to be written against. Feeding them to COLMAP as pose
priors, which would let spatial matching replace the quadratic exhaustive one, is the
obvious next thing and is not done.

### Verified, and not

Verified here: certificate generation and the SANs, TLS on the LAN address, the whole
API (upload, undo, finish, a name that tries to escape `storage/datasets`, a body that is
not an image), the guidance rules under `node --test`, a full simulated capture to 100%
coverage with blur rejection firing, the `getUserMedia` path against a canvas backed fake
camera, and — the one that matters — 16 real photos replayed through the HTTP API and
reconstructed by the pipeline into the same models the same photos give when copied by
hand.

**Not verified: a real Android phone.** No device was in the loop, so the WebXR session,
`camera-access` readback and the ARCore poses are written against the specification and
Chrome's implementation of it, and have never run. The first phone to open this will find
whatever that missed — the likely candidates are the camera image resolution (it is the
tracking stream, which on some devices is well under the 1024 px the pipeline wants) and
the intrinsics derivation, which assumes the camera image is aligned with the view
frustum.

## Video input

`build_images` falls back to `build_frames` when `train/` has a clip and no photos.
The clip is cut into `IMAGE_MAX_ITEMS` even windows and each contributes its sharpest
frame, scored by the variance of the Laplacian: a handheld pan is mostly redundant and
partly smeared by motion blur, which SIFT cannot match. One decoding pass through
OpenCV, holding one frame at a time. Frames land as `frame_00000.jpg` and the rest of
the pipeline cannot tell the difference, except that they carry no EXIF, so every
camera starts from COLMAP's default focal prior.

Measured on a clip built from `over-office-2`'s 205 photos. Whole clip through the
pipeline: 205 frames extracted, 205 registered in a single SfM model, 2.8M dense
points at 82 to 90% depth coverage per frame, splat trained. Capped at 60 frames it
splits into 3 models of 33/10/7 images, but so does running on 60 of the original
photos (33/13/3): that is the decimation talking, not the video, and it is why
`build_sfm_reconstruction` now warns when the photos do not connect into one model.
Blurring two frames out of three in the clip changed nothing, the selection found a
sharp frame in all 60 windows (scores 481 and up, against 50 to 70 for the blurred).

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
   With `--dense` the image is matched twice: the triangulated keypoints first, then
   the dense lifted ones against the query keypoints still without a correspondence.
   The two passes are what makes dense a fallback instead of a competitor — see the
   dead end below.
3. **Solve** PnP with LO-RANSAC at 4 px, then refine pose *and* intrinsics.
4. **Report** inliers split by source (triangulated against lifted from dense depth, so
   the run says on its own whether the fallback carried it), mean reprojection error and
   a two panel `_overlay.jpg`: the
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

### Leave one out evaluation

The last pipeline step (`libs/evaluation.py`, `--relocation-stride`, `--relocation-dense`)
localises a spread of the dataset's own images against the model with each one hidden.
One image every `RELOCATION_STRIDE`, clamped to `[RELOCATION_MIN_ITEMS,
RELOCATION_MAX_ITEMS]`, spread uniformly over the sorted names so the sample covers
different viewpoints. Report in `storage/datasets/<name>/relocation.json`, deleted by
`--reset`, skipped when the file is already there.

**It reports margins, not accuracy, because accuracy does not discriminate.** Hiding one
image leaves its neighbours 5.7% of the scene radius away on `south-building`, roughly
where the photographer stood: every healthy dataset lands within a hundredth of a
percent, and a score built on that error reads 100% for all of them. What varies in
production is whether a query registers *at all*, which depends on how far it sits from
anything mapped. So each holdout climbs two ladders until the pose leaves the tolerance
(`PASS_POSITION_PCT`, `PASS_ROTATION_DEG`), by binary search — three or four trials
instead of the whole ladder:

- **viewpoint** (`VIEWPOINT_LADDER`): hide the holdout *and* its k nearest views, so the
  query has to be solved from further off the capture path. The margin is the distance
  and viewing angle to the nearest view still in the model at the last rung that held —
  `viewpoint_margin_pct` is the headline number, "it relocalises up to X% of the scene
  radius away from anything mapped".
- **appearance** (`APPEARANCE_LADDERS`): degrade the query itself in resolution, focus
  and JPEG quality, and report the harshest level still solved. This is what decides
  whether a photo taken later with another phone registers. It re-runs SIFT per trial,
  so it covers `RELOCATION_APPEARANCE_ITEMS` of the holdouts, not all of them.

The k = 0 rung is the old accuracy measurement and is still reported under `accuracy`,
as the detail behind the margin rather than the headline.

Both ladders run further than looks sensible on purpose. The first version stopped at 32
hidden views, scale 0.15, blur 5 and JPEG 10, and `south-building` walked through all
four ceilings — a ladder that ends before the model breaks measures the ladder. Rungs
past the breaking point are free, the binary search never visits them. `capped_by_ladder`
counts the holdouts that survived to the top, i.e. the ones whose reported margin is a
floor rather than a measurement.

Two things are hidden from the query, and the test is worthless without either:

- **its rows in the retrieval index** (`localizer.hide_image`). Otherwise the query
  retrieves *itself* first — measured, 553 votes against 90 for the runner up — and PnP
  is handed the keypoints it is supposed to find. The reconstruction is left untouched:
  only the index decides what can be retrieved and matched.
- **its own camera** (`query_camera(exclude=...)`), so the query gets the median
  intrinsics a real photo would get instead of the ones bundle adjustment fit on that
  very image. Barely matters on a fixed lens capture (851.75 against 851.77 px on
  `south-building`), matters on phone photos where the focal moves per shot.

The ground truth is a *pseudo* one: those 3D points were adjusted with the holdout's
observations too, so the model has partly seen the answer. At 128 images that is under
1% of the constraints; on a twenty image dataset read the numbers as optimistic. Query
descriptors come from `database.db` instead of a fresh SIFT pass (same file, same
detector), so the check measures the localiser, not the resize and extract path around
it in `relocation.py`. The degraded queries do go through a real SIFT pass, they have to.

Baseline on `south-building` (128 images, 12 holdouts, 68 localisations, 4m36s, sparse):

| | median | worst holdout | best |
|---|---|---|---|
| viewpoint margin | **56.9% of the scene radius** | 31.6% | 88.9% |
| viewpoint margin, angle | 31.6° | 20.2° | 50.9° |
| views hidden at the margin | 16 | 16 | 32 |
| accuracy at k = 0 | 0.0158% of radius | 0.0317% | 0.0038% |
| appearance: scale | 0.15 | 0.25 | |
| appearance: blur | 5.0 px | 3.0 px | |
| appearance: JPEG quality | 2 | 5 | |

Read as: this model places a photo taken half a scene radius and 30 degrees away from
anything it has seen, at a sixth of the resolution. The JPEG axis still bottoms out at
quality 2 — compression is simply not what breaks relocation on a textured facade, and
that axis only earns its place on a weaker dataset.

The margin does discriminate where the old score did not: 31.6% against 88.9% between
the weakest and the strongest holdout of the same dataset, so a regression in coverage
shows up here long before it shows up in the position error.


### Measured dead ends

- **Dense depth adds correspondences but not accuracy.** Lifting the matched keypoints
  that were never triangulated (`--dense`, `DenseDepth` in `libs/localizer.py`) gives
  50 to 90% more correspondences and changes the pose by less than the noise: `home`
  0.33% of radius against 0.23% without, `over-office-1` 0.02% against 0.04%, at 1.5x
  the matching cost. The MVS depths are simply less accurate than the triangulated
  points. Kept as a flag, off by default.

  Mixing both sources in one matching pass also *costs* triangulated correspondences:
  the dense descriptors of a surface sit next to the triangulated ones, so each kills
  the other's ratio test and mutual check. Measured on three `south-building` queries,
  one pass dropped 5.6 to 7.2% of the sparse correspondences (1647 → 1529, 1658 →
  1547, 3441 → 3247). Hence the two passes, plus a triangulated point winning any dense
  one on the same query keypoint whatever the descriptor distance: `--dense` now leaves
  the sparse correspondences and inliers bit for bit identical to a sparse only run and
  only adds on top.

  What that buys, precisely: on those queries the pose comes out the same either way,
  one pass or two, to well under a millimetre. The two passes are a guarantee that the
  fallback cannot subtract from the sparse solution, not a measured accuracy gain.

  The leave one out above settles the question on queries the model cannot retrieve as
  themselves. Over 12 `south-building` holdouts the dense fallback adds 43% more inliers
  (median 2638 against 1845) and **loses on accuracy**: better position on 3 of 12
  images, better rotation on 2 of 12, worst case 0.0426% of radius against 0.0317%. It
  also loses on the margin that matters, measured over 6 holdouts: median viewpoint
  margin 40.1% of the scene radius against 45.9% sparse. More correspondences, less
  reach. Off by default, and the number to watch when changing it.

  The saved JSON says which source the pose rests on: `dense_fallback`
  (`off` | `unavailable` | `on`), `num_sparse_inliers` and `num_dense_inliers` next to
  the correspondence counts, and `point_source` (`sparse` | `sparse+dense`). Compare
  `inlier_ratio` and `reprojection_error` only between runs with the same
  `dense_fallback`: the dense points change the population they average over.
- **The gaussians have nothing to offer here.** Photometric refinement against a render
  would need the torch subprocess and a differentiable pose, to improve on a PnP that
  already lands within 0.03% of the scene radius and 0.02 degrees.
- **kd-trees on 128 dimensional descriptors.** `scipy.cKDTree` degenerates to a full
  scan and is 5x slower than one matrix product. Matching is brute force, one product
  per pair, with the reverse direction taken as a second reduction over the same block
  (that makes the mutual check nearly free). Two masked `argmin` passes beat
  `argpartition` by 3.5x on matrices this wide.
