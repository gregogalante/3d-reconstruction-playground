"""Single image relocalisation against a COLMAP reconstruction.

Matching a query straight against the whole model does not work well: the ratio test
compares each query descriptor with the entire point cloud, where thousands of
unrelated points look alike, so it only survives with a very strict threshold and
leaves a handful of correspondences on which PnP has nothing to reject.

This module takes the retrieval route instead. The query is first matched loosely
against the model to rank the database images, then matched again image by image,
where the ratio test compares a few thousand descriptors from one viewpoint and is
meaningful. Matched database keypoints become 3D through their track, or through the
dense depth map when `use_dense` is on and the sparse model does not cover them.
"""

import os
import sqlite3

import numpy as np
import pycolmap

from libs.cpu_mvs import read_colmap_map

# Descriptors kept per 3D point, spread over its track: the first observations of a
# point are often the least representative ones, and the whole track is redundant.
MAX_DESCRIPTORS_PER_POINT = 4

##############################################################################
# DATABASE
##############################################################################

def read_database(path):
  """Keypoints and descriptors of every image in a COLMAP database, by image id."""
  connection = sqlite3.connect(path)
  names = dict(connection.execute("SELECT image_id, name FROM images"))
  keypoints = {
    image_id: np.frombuffer(data, np.float32).reshape(rows, cols)
    for image_id, rows, cols, data in connection.execute("SELECT image_id, rows, cols, data FROM keypoints")
  }
  descriptors = {
    image_id: np.frombuffer(data, np.uint8).reshape(rows, cols)
    for image_id, rows, cols, data in connection.execute("SELECT image_id, rows, cols, data FROM descriptors")
  }
  connection.close()
  return names, keypoints, descriptors

##############################################################################
# MATCHING
##############################################################################

def _nearest(query, reference, k, chunk=2048):
  """The k closest reference rows for each query row, as squared L2 distances.

  Brute force: SIFT descriptors are 128 dimensional, where a kd-tree degenerates to a
  full scan anyway and does it 5x slower than one matrix product.
  """
  norms = np.einsum("ij,ij->i", reference, reference)
  distances = np.empty((len(query), k), np.float32)
  indices = np.empty((len(query), k), np.int64)
  for start in range(0, len(query), chunk):
    block = query[start:start + chunk]
    costs = norms[None, :] - 2.0 * (block @ reference.T)
    closest = np.argpartition(costs, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(costs, closest, axis=1)
    order = np.argsort(values, axis=1)
    indices[start:start + chunk] = np.take_along_axis(closest, order, axis=1)
    distances[start:start + chunk] = np.take_along_axis(values, order, axis=1)
  # the query norm is constant per row, added back so the ratio test is on real distances
  return distances + np.einsum("ij,ij->i", query, query)[:, None], indices


def match_pair(query, reference, ratio=0.8, chunk=1024):
  """Mutual nearest neighbours passing Lowe's ratio test, as (query, reference, distance).

  One matrix product per pair: the reverse direction is a second reduction over the
  same distance block, so the mutual check costs a fraction of a second full pass.
  """
  reference_norms = np.einsum("ij,ij->i", reference, reference)
  query_norms = np.einsum("ij,ij->i", query, query)
  best = np.empty((len(query), 2), np.float32)
  best_index = np.empty(len(query), np.int64)
  back = np.full(len(reference), np.inf, np.float32)
  back_index = np.zeros(len(reference), np.int64)
  columns = np.arange(len(reference))

  for start in range(0, len(query), chunk):
    stop = min(start + chunk, len(query))
    # the query norm shifts a whole row, but the column reduction below mixes rows, so
    # it goes in before anything is compared
    costs = reference_norms[None, :] - 2.0 * (query[start:stop] @ reference.T)
    costs += query_norms[start:stop, None]

    # two masked argmin passes: 3.5x faster than a partition on a matrix this wide
    lines = np.arange(stop - start)
    first = costs.argmin(axis=1)
    nearest = costs[lines, first]
    costs[lines, first] = np.inf
    second = costs[lines, costs.argmin(axis=1)]
    costs[lines, first] = nearest
    best_index[start:stop] = first
    best[start:stop, 0], best[start:stop, 1] = nearest, second

    rows = costs.argmin(axis=0)
    values = costs[rows, columns]
    better = values < back
    back[better] = values[better]
    back_index[better] = rows[better] + start

  keep = best[:, 0] < (ratio ** 2) * best[:, 1]
  keep &= back_index[best_index] == np.arange(len(query))
  return np.where(keep)[0], best_index[keep], best[keep, 0]

##############################################################################
# POINT INDEX
##############################################################################

def build_point_index(reconstruction, descriptors, max_per_point=MAX_DESCRIPTORS_PER_POINT):
  """One descriptor row per kept observation, tagged with its 3D point and image."""
  rows, point_of_row, image_of_row = [], [], []
  xyz, point_ids = [], []
  for point_id, point in reconstruction.points3D.items():
    track = [element for element in point.track.elements if element.image_id in descriptors]
    if not track:
      continue
    if len(track) > max_per_point:
      track = [track[i] for i in np.linspace(0, len(track) - 1, max_per_point).astype(int)]
    for element in track:
      rows.append(descriptors[element.image_id][element.point2D_idx])
      point_of_row.append(len(xyz))
      image_of_row.append(element.image_id)
    xyz.append(point.xyz)
    point_ids.append(point_id)
  return {
    "descriptors": np.asarray(rows, np.float32),
    "point_of_row": np.asarray(point_of_row),
    "image_of_row": np.asarray(image_of_row),
    "xyz": np.asarray(xyz, np.float64),
    "point_ids": np.asarray(point_ids),
  }


def rank_images(query_descriptors, index, count, ratio=0.9, neighbours=8, sample=4000):
  """Database images ranked by how many query descriptors land on their tracks.

  The runner up of the ratio test has to belong to a *different* 3D point: a point
  contributes several similar descriptors to the index, and comparing a descriptor
  with its own siblings would reject every correct match.

  This only has to produce a ranking, so a few thousand descriptors are enough and
  keep the pass over the whole model cheap on feature rich images.
  """
  if len(query_descriptors) > sample:
    query_descriptors = query_descriptors[::len(query_descriptors) // sample + 1]
  distances, rows = _nearest(query_descriptors, index["descriptors"], neighbours)
  points = index["point_of_row"][rows]
  other = points != points[:, :1]
  runner_up = np.where(other.any(axis=1), other.argmax(axis=1), neighbours - 1)
  second = distances[np.arange(len(distances)), runner_up]
  keep = other.any(axis=1) & (distances[:, 0] < (ratio ** 2) * second)

  images, votes = np.unique(index["image_of_row"][rows[keep, 0]], return_counts=True)
  order = np.argsort(-votes)[:count]
  return images[order].tolist(), votes[order].tolist()

##############################################################################
# DENSE DEPTH
##############################################################################

class DenseDepth:
  """Turns a pixel of a database image into a 3D point using its dense depth map.

  Only a quarter of the keypoints of an image usually carry a 3D point: the rest were
  never triangulated, and without depth they are matches we have to throw away.
  """

  def __init__(self, dense_path):
    self.reconstruction = pycolmap.Reconstruction(os.path.join(dense_path, "sparse"))
    self.by_name = {image.name: image for image in self.reconstruction.images.values()}
    self.maps_path = os.path.join(dense_path, "stereo", "depth_maps")
    self.cache = {}

  @staticmethod
  def available(dense_path):
    return os.path.isdir(os.path.join(dense_path, "stereo", "depth_maps"))

  def _depth_map(self, name):
    if name not in self.cache:
      path = os.path.join(self.maps_path, f"{name}.geometric.bin")
      self.cache[name] = read_colmap_map(path) if os.path.exists(path) else None
    return self.cache[name]

  def points(self, name, pixels, camera):
    """World points for pixels of the original image, with a mask of the valid ones.

    The depth maps live on the undistorted images and at a lower resolution, so the
    pixels are undistorted with the original camera and rescaled before the lookup.
    """
    image = self.by_name.get(name)
    depth = self._depth_map(name)
    if image is None or depth is None:
      return None, None

    rays = np.hstack([np.asarray(camera.cam_from_img(pixels)), np.ones((len(pixels), 1))])
    undistorted = self.reconstruction.cameras[image.camera_id]
    projected = np.asarray(undistorted.img_from_cam(rays)) * (depth.shape[1] / undistorted.width)
    columns = np.round(projected[:, 0]).astype(int)
    lines = np.round(projected[:, 1]).astype(int)

    inside = (columns >= 0) & (columns < depth.shape[1]) & (lines >= 0) & (lines < depth.shape[0])
    values = np.zeros(len(pixels), np.float32)
    values[inside] = depth[lines[inside], columns[inside]]

    cam_from_world = image.cam_from_world()
    local = rays * values[:, None]
    world = (local - cam_from_world.translation) @ cam_from_world.rotation.matrix()
    return world, values > 0

##############################################################################
# CAMERA
##############################################################################

def query_camera(reconstruction, size, name=None, exif_camera=None):
  """The intrinsics to localise with, best source first.

  COLMAP's EXIF inference falls back to a 1.2 x max dimension focal guess when the
  sensor is unknown, which is off by more than 50% on a phone wide angle lens and
  moves the estimated pose by metres. The reconstruction is calibrated on the very
  same photos, so it wins over any guess.
  """
  for image in reconstruction.images.values():
    if image.name == name:
      return reconstruction.cameras[image.camera_id], "reconstruction (same image)"

  if exif_camera is not None and exif_camera.has_prior_focal_length:
    return exif_camera, "exif"

  cameras = list(reconstruction.cameras.values())
  aspect = size[0] / size[1]
  matching = [c for c in cameras if abs(c.width / c.height - aspect) < 0.02] or cameras
  focal = float(np.median([c.params[0] / max(c.width, c.height) for c in matching]))
  distortion = float(np.median([c.params[3] for c in matching if c.model.name == "SIMPLE_RADIAL"] or [0.0]))
  camera = pycolmap.Camera(
    model="SIMPLE_RADIAL",
    width=size[0],
    height=size[1],
    params=[focal * max(size), size[0] / 2.0, size[1] / 2.0, distortion],
  )
  return camera, f"reconstruction ({len(matching)} cameras, median)"

##############################################################################
# LOCALISATION
##############################################################################

def refine_intrinsics(cam_from_world, points2D, points3D, inliers, camera, max_error):
  """Solve for the focal length together with the pose, then re-select the inliers.

  Unless the query comes from the dataset its focal length is a guess, and a few
  percent of error there shifts the camera along its own axis by the same amount.
  With a few hundred inliers it is better solved for than assumed.
  """
  options = pycolmap.AbsolutePoseRefinementOptions()
  options.refine_focal_length = True
  options.refine_extra_params = True
  # the camera is refined in place, the pose comes back in the result
  for _ in range(2):
    result = pycolmap.refine_absolute_pose(cam_from_world, points2D, points3D, inliers, camera, options)
    if result is None:
      break
    cam_from_world = result["cam_from_world"]
    local = cam_from_world * points3D
    projected = np.asarray(camera.img_from_cam(local))
    inliers = (np.linalg.norm(projected - points2D, axis=1) < max_error) & (local[:, 2] > 0)
  return cam_from_world, inliers


def image_points(reconstruction, image, keypoints, dense=None, name=None):
  """A world point for every keypoint of a database image, and which ones got one.

  Triangulated keypoints take their point from the model, the others from the dense
  depth map when it exists. Keypoints left without a 3D point are dropped before
  matching: they cost as much as the useful ones and can only produce dead matches.
  """
  points = np.full((len(keypoints), 3), np.nan)
  sparse = np.array([point.has_point3D() for point in image.points2D])
  if sparse.any():
    points[sparse] = [reconstruction.points3D[image.points2D[i].point3D_id].xyz
                      for i in np.where(sparse)[0]]
  if dense is not None and not sparse.all():
    missing = np.where(~sparse)[0]
    lifted, valid = dense.points(name, keypoints[missing, :2].astype(np.float64),
                                 reconstruction.cameras[image.camera_id])
    if lifted is not None:
      points[missing[valid]] = lifted[valid]
  return points, ~np.isnan(points[:, 0]), sparse


def localize(dataset_path, keypoints, descriptors, camera, num_retrieved=10, ratio=0.8,
             max_error=4.0, use_dense=False, min_inliers=30, min_focal_inliers=50,
             enough=800, min_images=4, log=print):
  """Estimate the pose of a query image against the dataset's reconstruction.

  A photo of another scene still produces a pose, RANSAC will always find some minimal
  set that agrees, so a registration under `min_inliers` counts as a failure. That is
  COLMAP's own threshold for accepting an image into a reconstruction.
  """
  reconstruction = pycolmap.Reconstruction(os.path.join(dataset_path, "sfm", "0"))
  names, db_keypoints, db_descriptors = read_database(os.path.join(dataset_path, "database.db"))
  descriptors = descriptors.astype(np.float32)

  index = build_point_index(reconstruction, db_descriptors)
  retrieved, votes = rank_images(descriptors, index, num_retrieved)
  log(f"Retrieved {len(retrieved)} images, votes {votes}")

  dense_path = os.path.join(dataset_path, "dense")
  dense = DenseDepth(dense_path) if use_dense and DenseDepth.available(dense_path) else None
  if use_dense and dense is None:
    log("No dense depth maps in the dataset, only triangulated keypoints can be matched")

  # best correspondence per query keypoint, closest descriptor wins
  best = {}
  used = 0
  for image_id in retrieved:
    # a few hundred correspondences already over determine six degrees of freedom, and
    # the retrieved images are ranked, so the tail only costs time
    if used >= min_images and len(best) >= enough:
      log(f"Stopped after {used} images with {len(best)} correspondences")
      break
    if image_id not in reconstruction.images:
      continue
    image = reconstruction.images[image_id]
    if not image.has_pose:
      continue

    points, usable, sparse = image_points(reconstruction, image, db_keypoints[image_id],
                                          dense, names[image_id])
    if not usable.any():
      continue
    slots = np.where(usable)[0]
    query_idx, db_idx, distances = match_pair(descriptors, db_descriptors[image_id][usable].astype(np.float32), ratio)
    used += 1

    for query, slot, distance in zip(query_idx, slots[db_idx], distances):
      current = best.get(query)
      if current is None or distance < current[0]:
        best[query] = (distance, points[slot], bool(sparse[slot]))

  def failure(reason, inliers=0):
    return {"success": False, "reason": reason, "num_inliers": inliers,
            "num_correspondences": len(best), "retrieved": [names[i] for i in retrieved]}

  if len(best) < 4:
    return failure(f"only {len(best)} correspondences, PnP needs at least 4")

  query_idx = np.array(sorted(best))
  points2D = keypoints[query_idx, :2].astype(np.float64)
  points3D = np.array([best[i][1] for i in query_idx], np.float64)
  from_sparse = sum(best[i][2] for i in query_idx)
  from_dense = len(best) - from_sparse
  log(f"{len(best)} correspondences" + (f", {from_dense} of them lifted from dense depth" if from_dense else ""))

  options = pycolmap.AbsolutePoseEstimationOptions()
  options.ransac.max_error = max_error
  result = pycolmap.estimate_and_refine_absolute_pose(points2D, points3D, camera, options)
  if result is None:
    return failure(f"PnP found no pose over {len(best)} correspondences")

  cam_from_world = result["cam_from_world"]
  inliers = np.asarray(result["inlier_mask"], bool)
  focal = camera.params[0]
  if min_focal_inliers and inliers.sum() >= min_focal_inliers:
    cam_from_world, inliers = refine_intrinsics(cam_from_world, points2D, points3D, inliers, camera, max_error)
    log(f"Refined the focal length from {focal:.1f} to {camera.params[0]:.1f} px, "
        f"{int(inliers.sum())} inliers")

  if inliers.sum() < min_inliers:
    return failure(f"{int(inliers.sum())} inliers out of {len(best)} correspondences, "
                   f"under the {min_inliers} needed to trust a pose", int(inliers.sum()))

  projected = camera.img_from_cam(cam_from_world * points3D[inliers])
  residuals = np.linalg.norm(np.asarray(projected) - points2D[inliers], axis=1)
  return {
    "success": True,
    "cam_from_world": cam_from_world,
    "num_inliers": int(inliers.sum()),
    "num_correspondences": len(best),
    "num_sparse_correspondences": int(from_sparse),
    "num_dense_correspondences": int(from_dense),
    "inlier_ratio": float(inliers.mean()),
    "reprojection_error": float(residuals.mean()),
    "retrieved": [names[i] for i in retrieved],
    "points2D": points2D,
    "points3D": points3D,
    "inlier_mask": inliers,
  }
