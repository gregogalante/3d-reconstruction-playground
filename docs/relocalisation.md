# Relocalisation

Placing a single photo in an existing reconstruction, and measuring how far a
model can be pushed before it stops managing it.

Back to [AGENTS.md](../AGENTS.md).

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
   dead end at the foot of this file.
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
reason. Importing `relocation` from `viewer_server.py` is safe, neither pulls in torch.

What actually mattered, measured against the table in [README.md](../README.md):

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
