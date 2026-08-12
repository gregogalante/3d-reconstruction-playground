# Reconstruction

From photos or a clip to a dense point cloud, and what to do when the model comes
back in more than one piece.

Back to [AGENTS.md](../AGENTS.md).

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

## When a model comes back in pieces

The first thing to check is not the capture, it is whether the matches to join it are
already in the database. Matching is exhaustive — every pair, not a chain — so a frame
of blank wall cannot break a sequence, and two frames that see the same thing are tried
against each other however far apart they were taken.

Build the view graph out of `two_view_geometries` and count its connected components at
a few inlier thresholds. On `test4`, a capture of white walls that came back as three
models, **190 of 205 images were one component at 15 inlier matches** — and at 30, the
graph fell into 23 pieces. COLMAP's `Mapper.abs_pose_min_num_inliers` is 30. The model
was not disconnected, the mapper was refusing links that existed, which is why the
thresholds at the top of `pipeline.py` are lowered to what the view graph offers.

The measurement that justifies them, largest model out of the images available:

| | stock COLMAP | lowered |
|---|---|---|
| `test4` (starved of texture) | 63 of 205 | **107**, reprojection 0.75 px |
| `banana` | 14 + 7, two models | **15, one model**, same points, 0.25 px |
| `south-building` | 128, one model | 128, unchanged, 0.35 px |

The control is the half that matters: a threshold that rescues a bad capture is only
worth having if it invents nothing on a good one, and the reprojection error not moving
is what says the extra images are real rather than forced.

What does *not* help, measured on the same 60 frames: more features. Dropping SIFT's peak
threshold takes the keypoint count from 702 to 1751 and the largest model from 31 to 31;
at 1440 px it reaches 3300 keypoints and *six* models instead of three. On a blank wall
more features are more indistinct blobs, and they match each other wrongly.

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
