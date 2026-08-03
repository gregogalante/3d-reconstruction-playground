"""CPU multi-view stereo for COLMAP dense reconstruction.

COLMAP's dense stereo (`patch_match_stereo`) is CUDA-only, so on Apple Silicon the
dense step of the pipeline is simply unavailable. Only that step needs replacing:
image undistortion and stereo fusion in pycolmap already run on CPU.

This module estimates one depth/normal map per image with a plane-sweep ZNCC search
(OpenCV/numpy) and writes them into the COLMAP dense workspace using COLMAP's own
binary map format, so `pycolmap.stereo_fusion` consumes them unchanged:

  undistort_images -> [cpu_mvs.build_depth_maps] -> stereo_fusion -> fused.ply

Two passes, mirroring COLMAP: a photometric pass (per-image plane sweep) followed by
a geometric pass keeping only the depths confirmed by several neighbouring views.

Torch is deliberately not used here: its libomp clashes with the one pycolmap links,
which aborts the process as soon as both run together.
"""

import os
import time
import numpy as np
import cv2
import pycolmap
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_COST = 2.0        # cost assigned to unusable (occluded / untextured) samples
VARIANCE_EPS = 1e-6   # minimum patch variance to consider a window textured

##############################################################################
# COLMAP DENSE MAP I/O
##############################################################################

def read_colmap_map(path):
  """Read a COLMAP dense map (.bin) as a (height, width[, channels]) array."""
  with open(path, "rb") as f:
    header = b""
    while header.count(b"&") < 3:
      header += f.read(1)
    width, height, channels = (int(value) for value in header.decode().split("&")[:3])
    data = np.fromfile(f, dtype=np.float32)
  # COLMAP stores maps slice-major, with the width index varying fastest
  return data.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()

def write_colmap_map(path, array):
  """Write a (height, width[, channels]) array in COLMAP dense map format."""
  array = array.astype(np.float32)
  if array.ndim == 2:
    array = array[:, :, None]
  height, width, channels = array.shape
  with open(path, "wb") as f:
    f.write(f"{width}&{height}&{channels}&".encode())
    array.transpose(1, 0, 2).ravel(order="F").tofile(f)

##############################################################################
# VIEW SELECTION
##############################################################################

def _triangulation_weight(angles, optimal=5.0, sigma_low=1.0, sigma_high=10.0):
  """Score of a triangulation angle (degrees): peaks at `optimal`, decays around it."""
  sigma = np.where(angles <= optimal, sigma_low, sigma_high)
  return np.exp(-((angles - optimal) ** 2) / (2 * sigma ** 2))

def build_view_graph(reconstruction, num_src_images, candidate_factor=3):
  """Map each image id to the ids of the source views best suited to match it.

  Candidates are ranked by co-visible sparse points, then rescored by summing a
  triangulation-angle weight over those points: a good source view shares many
  points *and* looks at them from a useful baseline.
  """
  point_ids = {}
  centers = {}
  for image_id, image in reconstruction.images.items():
    point_ids[image_id] = [p.point3D_id for p in image.points2D if p.has_point3D()]
    centers[image_id] = image.projection_center()

  observers = {}
  for image_id, ids in point_ids.items():
    for point_id in ids:
      observers.setdefault(point_id, []).append(image_id)

  view_graph = {}
  for image_id, ids in point_ids.items():
    shared = {}
    for point_id in ids:
      for other_id in observers[point_id]:
        if other_id != image_id:
          shared[other_id] = shared.get(other_id, 0) + 1

    candidates = sorted(shared, key=shared.get, reverse=True)[: num_src_images * candidate_factor]
    reference_points = set(ids)
    scores = []
    for other_id in candidates:
      common = [pid for pid in point_ids[other_id] if pid in reference_points]
      if not common:
        continue
      xyz = np.array([reconstruction.points3D[pid].xyz for pid in common])
      ray_reference = xyz - centers[image_id]
      ray_source = xyz - centers[other_id]
      cosine = np.sum(ray_reference * ray_source, axis=1) / (
        np.linalg.norm(ray_reference, axis=1) * np.linalg.norm(ray_source, axis=1) + 1e-12
      )
      angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
      scores.append((_triangulation_weight(angles).sum(), other_id))

    scores.sort(reverse=True)
    view_graph[image_id] = [other_id for _, other_id in scores[:num_src_images]]
  return view_graph

##############################################################################
# VIEW LOADING
##############################################################################

def _scale_calibration(matrix, scale_x, scale_y):
  """Scale a pinhole calibration matrix, accounting for the half-pixel offset."""
  scaled = matrix.copy()
  scaled[0] *= scale_x
  scaled[1] *= scale_y
  scaled[0, 2] += 0.5 * scale_x - 0.5
  scaled[1, 2] += 0.5 * scale_y - 0.5
  return scaled

def load_view(reconstruction, image_id, images_path, max_image_size, with_image=True):
  """Load a view: pose, calibration at full and working scale, optional greyscale."""
  image = reconstruction.images[image_id]
  camera = reconstruction.cameras[image.camera_id]
  pose = image.cam_from_world()
  calibration = camera.calibration_matrix().astype(np.float64)

  view = {
    "image_id": image_id,
    "name": image.name,
    "K_full": calibration,
    "R": pose.rotation.matrix().astype(np.float64),
    "t": pose.translation.astype(np.float64),
    "size": (camera.height, camera.width),
  }
  if not with_image:
    return view

  bitmap = cv2.imread(os.path.join(images_path, image.name), cv2.IMREAD_GRAYSCALE)
  if bitmap is None:
    raise IOError(f"Cannot read image {image.name} in {images_path}")

  height, width = bitmap.shape
  scale = min(1.0, max_image_size / max(width, height))
  if scale < 1.0:
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    bitmap = cv2.resize(bitmap, size, interpolation=cv2.INTER_AREA)

  view["gray"] = (bitmap.astype(np.float32) / 255.0)
  view["K"] = _scale_calibration(calibration, bitmap.shape[1] / width, bitmap.shape[0] / height)
  return view

def depth_range(reconstruction, image_id, margin=0.15, min_points=10):
  """Depth interval spanned by the sparse points visible in an image."""
  image = reconstruction.images[image_id]
  pose = image.cam_from_world()
  points = [p.point3D_id for p in image.points2D if p.has_point3D()]
  if len(points) < min_points:
    return None
  xyz = np.array([reconstruction.points3D[pid].xyz for pid in points])
  depths = xyz @ pose.rotation.matrix()[2] + pose.translation[2]
  depths = depths[depths > 0]
  if len(depths) < min_points:
    return None
  low, high = np.percentile(depths, [1.0, 99.0])
  return max(low * (1.0 - margin), 1e-4), high * (1.0 + margin)

##############################################################################
# PLANE SWEEP
##############################################################################

def _box(image, radius):
  """Mean over a square window (normalised, borders replicated)."""
  size = 2 * radius + 1
  return cv2.boxFilter(image, cv2.CV_32F, (size, size), borderType=cv2.BORDER_REFLECT)

def _homography(ref, src, depth):
  """Reference-to-source homography induced by a fronto-parallel plane at `depth`.

  H = K_src (R_rel + t_rel [0,0,1/depth]) K_ref^-1
  """
  rotation = src["R"] @ ref["R"].T
  translation = src["t"] - rotation @ ref["t"]
  plane = rotation.copy()
  plane[:, 2] += translation / depth
  return src["K"] @ plane @ np.linalg.inv(ref["K"])

def _warp(image, matrix, size, interpolation):
  # our homography maps reference (destination) pixels to source pixels: that is
  # exactly the inverse map OpenCV expects with WARP_INVERSE_MAP
  return cv2.warpPerspective(image, matrix, size, flags=interpolation | cv2.WARP_INVERSE_MAP, borderValue=0)

def _behind_camera(matrix, width, height):
  """True if part of the plane falls behind the source camera (w <= 0 somewhere).

  w = h20 x + h21 y + h22 is affine in the pixel coordinates, so checking the four
  image corners is enough.
  """
  corners = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]], dtype=np.float64)
  return bool((corners @ matrix[2] <= 1e-8).any())

def _pair_cost(ref, src, matrix, window_radius):
  """ZNCC matching cost of the reference against a source warped by `matrix`."""
  gray = ref["gray"]
  size = (gray.shape[1], gray.shape[0])
  warped = _warp(src["gray"], matrix, size, cv2.INTER_LINEAR)

  mean_src = _box(warped, window_radius)
  variance_src = _box(warped * warped, window_radius)
  variance_src -= mean_src * mean_src
  covariance = _box(warped * gray, window_radius)
  covariance -= mean_src * ref["mean"]

  textured_src = variance_src > VARIANCE_EPS
  np.maximum(variance_src, VARIANCE_EPS, out=variance_src)
  cost = 1.0 - covariance / (np.sqrt(variance_src, out=variance_src) * ref["std"])
  np.clip(cost, 0.0, MAX_COST, out=cost)

  # a window is only comparable when fully visible in the source view and textured
  # in both views, otherwise flat regions match everything
  visible = cv2.erode(_warp(src["ones"], matrix, size, cv2.INTER_NEAREST), ref["window"])
  usable = visible.astype(bool) & textured_src & ref["textured"]
  return np.where(usable, cost, MAX_COST)

def _keep_best(best, cost):
  """Insert a cost slice into the ascending list of the best costs seen so far."""
  for slot in best[:-1]:
    higher = np.maximum(slot, cost)
    np.minimum(slot, cost, out=slot)
    cost = higher
  np.minimum(best[-1], cost, out=best[-1])

def plane_sweep(ref, srcs, min_depth, max_depth, num_samples, window_radius, num_best):
  """Winner-takes-all depth map from ZNCC costs over fronto-parallel depth planes.

  Returns the depth map and its matching cost (0 = perfect, MAX_COST = no match).
  The cost volume is never materialised: only the running best cost per pixel and
  the costs of the two neighbouring planes, needed for the sub-plane refinement.
  """
  height, width = ref["gray"].shape

  # planes uniform in inverse depth, i.e. constant relative depth resolution
  inverse_depths = np.linspace(1.0 / max_depth, 1.0 / min_depth, num_samples)
  step = inverse_depths[1] - inverse_depths[0]

  mean_ref = _box(ref["gray"], window_radius)
  variance_ref = np.maximum(_box(ref["gray"] * ref["gray"], window_radius) - mean_ref * mean_ref, 0.0)
  ref = dict(
    ref,
    mean=mean_ref,
    std=np.maximum(np.sqrt(variance_ref), np.sqrt(VARIANCE_EPS)),
    textured=variance_ref > VARIANCE_EPS,
    window=np.ones((2 * window_radius + 1, 2 * window_radius + 1), dtype=np.uint8),
  )
  for src in srcs:
    src["ones"] = np.full(src["gray"].shape, 255, dtype=np.uint8)

  best_cost = np.full((height, width), MAX_COST, dtype=np.float32)
  best_index = np.full((height, width), -1, dtype=np.int32)
  cost_before = np.full((height, width), MAX_COST, dtype=np.float32)
  cost_after = np.full((height, width), MAX_COST, dtype=np.float32)
  previous_cost = None

  for index, depth in enumerate(1.0 / inverse_depths):
    # robust aggregation: mean of the few most consistent source views
    best = [np.full((height, width), MAX_COST, dtype=np.float32) for _ in range(min(num_best, len(srcs)))]
    for src in srcs:
      matrix = _homography(ref, src, depth)
      if _behind_camera(matrix, width, height):
        continue
      _keep_best(best, _pair_cost(ref, src, matrix, window_radius))
    aggregated = np.mean(best, axis=0, dtype=np.float32)

    # keep the cost of the plane following the current winner (parabola right side)
    pending = best_index == index - 1
    cost_after[pending] = aggregated[pending]

    improved = aggregated < best_cost
    if previous_cost is not None:
      cost_before[improved] = previous_cost[improved]
    best_cost[improved] = aggregated[improved]
    best_index[improved] = index
    previous_cost = aggregated

  # sub-plane refinement: parabola through the winning plane and its neighbours
  denominator = cost_before - 2.0 * best_cost + cost_after
  usable = np.abs(denominator) > 1e-8
  offset = np.divide(0.5 * (cost_before - cost_after), denominator, out=np.zeros_like(denominator), where=usable)
  np.clip(offset, -1.0, 1.0, out=offset)
  offset[(best_index <= 0) | (best_index >= num_samples - 1)] = 0.0

  inverse = inverse_depths[np.maximum(best_index, 0)] + offset * step
  depth_map = np.where(best_index >= 0, 1.0 / np.maximum(inverse, 1e-8), 0.0)
  return depth_map.astype(np.float32), best_cost

##############################################################################
# NORMALS
##############################################################################

def _unproject(depth, calibration):
  """Camera-frame points of a depth map, plus the pixel coordinate grids."""
  height, width = depth.shape
  xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
  points = np.stack([
    (xs - calibration[0, 2]) * depth / calibration[0, 0],
    (ys - calibration[1, 2]) * depth / calibration[1, 1],
    depth,
  ], axis=2)
  return points, xs, ys

def depth_to_normals(depth, calibration):
  """Normals in the camera frame from a depth map, oriented towards the camera."""
  points, _, _ = _unproject(depth.astype(np.float64), calibration)
  points = points.astype(np.float32)

  # the fusion rejects points whose normals disagree, so smooth before differentiating
  smoothed = cv2.GaussianBlur(points, (5, 5), 0)
  normals = np.cross(np.gradient(smoothed, axis=1), np.gradient(smoothed, axis=0))
  norm = np.linalg.norm(normals, axis=2, keepdims=True)
  normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-12)

  # a visible surface faces the camera: negative dot product with the viewing ray
  flip = np.sum(normals * points, axis=2) > 0
  normals[flip] *= -1.0
  normals[depth <= 0] = 0.0
  return normals

##############################################################################
# GEOMETRIC CONSISTENCY
##############################################################################

def geometric_filter(ref, srcs, depth_ref, depths_src, max_reproj_error, max_depth_error, min_consistent):
  """Drop depths that are not confirmed by enough neighbouring depth maps.

  A depth is confirmed by a source view when projecting it there, reading the source
  depth and reprojecting it back lands on the original pixel with the same depth.
  """
  points, xs, ys = _unproject(depth_ref, ref["K_full"])
  world = (points - ref["t"]) @ ref["R"]  # R^T (X - t) for row-vector points
  consistent = np.zeros(depth_ref.shape, dtype=np.int32)

  for src, depth_src in zip(srcs, depths_src):
    source_height, source_width = depth_src.shape
    in_source = world @ src["R"].T + src["t"]
    z = in_source[:, :, 2]
    valid = (depth_ref > 0) & (z > 0)
    safe_z = np.where(valid, z, 1.0)

    calibration = src["K_full"]
    u = calibration[0, 0] * in_source[:, :, 0] / safe_z + calibration[0, 2]
    v = calibration[1, 1] * in_source[:, :, 1] / safe_z + calibration[1, 2]
    valid &= (u >= 0) & (u <= source_width - 1) & (v >= 0) & (v <= source_height - 1)
    columns = np.clip(np.round(u), 0, source_width - 1).astype(np.int32)
    rows = np.clip(np.round(v), 0, source_height - 1).astype(np.int32)

    sampled = depth_src[rows, columns]
    valid &= sampled > 0

    back = np.stack([
      (columns - calibration[0, 2]) * sampled / calibration[0, 0],
      (rows - calibration[1, 2]) * sampled / calibration[1, 1],
      sampled,
    ], axis=2)
    back = ((back - src["t"]) @ src["R"]) @ ref["R"].T + ref["t"]
    back_z = back[:, :, 2]
    valid &= back_z > 0
    safe_back = np.where(valid, back_z, 1.0)

    reference = ref["K_full"]
    error = np.hypot(
      reference[0, 0] * back[:, :, 0] / safe_back + reference[0, 2] - xs,
      reference[1, 1] * back[:, :, 1] / safe_back + reference[1, 2] - ys,
    )
    relative_depth_error = np.abs(back_z - depth_ref) / np.maximum(depth_ref, 1e-8)
    consistent += (valid & (error < max_reproj_error) & (relative_depth_error < max_depth_error)).astype(np.int32)

  filtered = depth_ref.astype(np.float32).copy()
  filtered[consistent < min_consistent] = 0.0
  return filtered

##############################################################################
# ENTRY POINT
##############################################################################

def _photometric_map(context, image_id):
  """Estimate and store the raw depth/normal map of one image."""
  reconstruction = context["reconstruction"]
  name = reconstruction.images[image_id].name
  depth_target = os.path.join(context["depth_path"], f"{name}.photometric.bin")
  normal_target = os.path.join(context["normal_path"], f"{name}.photometric.bin")
  if os.path.exists(depth_target):
    return f"{name}: photometric map exists, skipping"

  bounds = depth_range(reconstruction, image_id)
  source_ids = context["view_graph"].get(image_id, [])
  ref = load_view(reconstruction, image_id, context["images_path"], context["max_image_size"], with_image=bool(source_ids and bounds))
  height, width = ref["size"]

  if not source_ids or bounds is None:
    write_colmap_map(depth_target, np.zeros((height, width), dtype=np.float32))
    write_colmap_map(normal_target, np.zeros((height, width, 3), dtype=np.float32))
    return f"{name}: not enough overlap, empty map"

  started = time.time()
  srcs = [load_view(reconstruction, source_id, context["images_path"], context["max_image_size"]) for source_id in source_ids]
  depth, cost = plane_sweep(ref, srcs, bounds[0], bounds[1], context["num_samples"], context["window_radius"], context["num_best_src"])
  depth[cost > 1.0 - context["min_ncc"]] = 0.0
  depth = cv2.medianBlur(depth, 3)

  # maps are stored at full image resolution, like COLMAP's own dense maps
  if depth.shape != (height, width):
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

  write_colmap_map(depth_target, depth)
  write_colmap_map(normal_target, depth_to_normals(depth, ref["K_full"]))
  return f"{name}: {len(srcs)} src, depth {bounds[0]:.2f}-{bounds[1]:.2f}, coverage {(depth > 0).mean():.1%}, {time.time() - started:.1f}s"

def _geometric_map(context, image_id):
  """Filter one raw depth map against its neighbours and store the result."""
  reconstruction = context["reconstruction"]
  name = reconstruction.images[image_id].name
  depth_target = os.path.join(context["depth_path"], f"{name}.geometric.bin")
  if os.path.exists(depth_target):
    return f"{name}: geometric map exists, skipping"

  ref = load_view(reconstruction, image_id, context["images_path"], context["max_image_size"], with_image=False)
  depth_ref = read_colmap_map(os.path.join(context["depth_path"], f"{name}.photometric.bin")).astype(np.float64)

  srcs, depths_src = [], []
  for source_id in context["view_graph"].get(image_id, []):
    source_map = os.path.join(context["depth_path"], f"{reconstruction.images[source_id].name}.photometric.bin")
    if not os.path.exists(source_map):
      continue
    srcs.append(load_view(reconstruction, source_id, context["images_path"], context["max_image_size"], with_image=False))
    depths_src.append(read_colmap_map(source_map).astype(np.float64))

  filtered = geometric_filter(
    ref, srcs, depth_ref, depths_src, context["max_reproj_error"], context["max_depth_error"], context["min_consistent"]
  )
  write_colmap_map(depth_target, filtered)
  write_colmap_map(os.path.join(context["normal_path"], f"{name}.geometric.bin"), depth_to_normals(filtered, ref["K_full"]))
  return f"{name}: geometric coverage {(filtered > 0).mean():.1%}"

def _run_pass(task, context, image_ids, num_workers, log):
  """Run one pass over all images, a few images at a time."""
  with ThreadPoolExecutor(num_workers) as executor:
    futures = {executor.submit(task, context, image_id): image_id for image_id in image_ids}
    for position, future in enumerate(as_completed(futures), start=1):
      log(f"[{position}/{len(image_ids)}] {future.result()}")

def build_depth_maps(
  workspace_path,
  max_image_size=640,
  num_src_images=6,
  num_samples=128,
  window_radius=3,
  min_ncc=0.4,
  num_best_src=3,
  max_reproj_error=2.0,
  max_depth_error=0.01,
  min_consistent=2,
  num_workers=None,
  log=print,
):
  """Fill `stereo/depth_maps` and `stereo/normal_maps` of a COLMAP dense workspace.

  Existing maps are kept, so an interrupted run resumes where it stopped.
  """
  context = {
    "images_path": os.path.join(workspace_path, "images"),
    "depth_path": os.path.join(workspace_path, "stereo", "depth_maps"),
    "normal_path": os.path.join(workspace_path, "stereo", "normal_maps"),
    "max_image_size": max_image_size,
    "num_samples": num_samples,
    "window_radius": window_radius,
    "min_ncc": min_ncc,
    "num_best_src": num_best_src,
    "max_reproj_error": max_reproj_error,
    "max_depth_error": max_depth_error,
    "min_consistent": min_consistent,
  }
  os.makedirs(context["depth_path"], exist_ok=True)
  os.makedirs(context["normal_path"], exist_ok=True)

  reconstruction = pycolmap.Reconstruction(os.path.join(workspace_path, "sparse"))
  context["reconstruction"] = reconstruction
  image_ids = sorted(reconstruction.images, key=lambda image_id: reconstruction.images[image_id].name)
  workers = num_workers or max(1, (os.cpu_count() or 4) - 2)
  log(f"CPU MVS on {len(image_ids)} images (size={max_image_size}, samples={num_samples}, src={num_src_images}, workers={workers})")

  context["view_graph"] = build_view_graph(reconstruction, num_src_images)

  # pass 1: per-image plane sweep
  _run_pass(_photometric_map, context, image_ids, workers, log)

  # pass 2: cross-view geometric consistency, the maps the fusion actually reads
  log("Filtering depth maps by geometric consistency...")
  _run_pass(_geometric_map, context, image_ids, workers, log)
