import os
import sys
import json
import math
import shutil
import tempfile
import argparse

import numpy as np
import cv2
import pycolmap
from PIL import Image

from libs.console import print_error, print_success, print_info, print_warning, print_step
from libs.localizer import localize, query_camera, extract_query_features

##############################################################################
# ARGS
##############################################################################

def parse_args():
  parser = argparse.ArgumentParser(description="Visual camera relocalization using COLMAP SfM")
  parser.add_argument("--dataset", required=True, help="Path to dataset directory (e.g. storage/datasets/home)")
  parser.add_argument("--image", required=True, help="Path to query image")
  parser.add_argument("--ratio", type=float, default=0.8, help="Lowe ratio test threshold used against each retrieved image")
  parser.add_argument("--retrieved", type=int, default=10, help="Database images matched against the query")
  parser.add_argument("--max-error", type=float, default=4.0, help="RANSAC reprojection threshold in pixels")
  parser.add_argument("--dense", action="store_true", help="Also match database keypoints without a 3D point, lifting them with the dense depth maps")
  parser.add_argument("--output", default=None, help="Path to output directory (saves image, overlay and JSON with relocation data)")
  return parser.parse_args()

##############################################################################
# QUERY IMAGE
##############################################################################

def prepare_query_image(image_path, tmp_dir, image_max_dimension):
  """Resize the query the way the pipeline resized the dataset, features must match."""
  with Image.open(image_path) as img:
    img = img.convert("RGB")
    scale = image_max_dimension / max(img.size)
    if scale < 1:
      original = img.size
      img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
      print_info(f"Resized query image from {original} to {img.size}")
    out_path = os.path.join(tmp_dir, "query.jpg")
    img.save(out_path, quality=95)
    return out_path, img.size


##############################################################################
# OUTPUT
##############################################################################

def build_relocation_data(result, image_path, dataset_path, camera, camera_source):
  """JSON serialisable summary of the estimate, including how much to trust it."""
  data = {
    "query_image": os.path.abspath(image_path),
    "dataset": os.path.abspath(dataset_path),
    "success": bool(result.get("success")),
    "reason": result.get("reason"),
    "num_correspondences": result.get("num_correspondences", 0),
    "retrieved_images": result.get("retrieved", []),
    "dense_fallback": result.get("dense_fallback", "off"),
    "camera_source": camera_source,
    "camera_params": [float(p) for p in camera.params],
    "fovx": 2 * math.degrees(math.atan(camera.width / (2 * camera.params[0]))),
    "fovy": 2 * math.degrees(math.atan(camera.height / (2 * camera.params[0]))),
  }
  if not result.get("success"):
    return data

  cam_from_world = result["cam_from_world"]
  rotation = cam_from_world.rotation.matrix()
  translation = cam_from_world.translation
  data.update({
    "num_inliers": result["num_inliers"],
    "num_sparse_correspondences": result["num_sparse_correspondences"],
    "num_dense_correspondences": result["num_dense_correspondences"],
    # what the pose actually rests on, as opposed to what was merely available
    "num_sparse_inliers": result["num_sparse_inliers"],
    "num_dense_inliers": result["num_dense_inliers"],
    "point_source": result["point_source"],
    "inlier_ratio": round(result["inlier_ratio"], 3),
    "reprojection_error": round(result["reprojection_error"], 3),
    "camera_center": (-rotation.T @ translation).tolist(),
    "translation": translation.tolist(),
    "rotation_matrix": [row.tolist() for row in rotation],
  })
  return data


def project_cloud(dataset_path, result, camera, frame):
  """Paint the reconstruction, seen from the estimated pose, over a darkened frame."""
  cloud_path = os.path.join(dataset_path, "dense", "fused.ply")
  if not os.path.exists(cloud_path):
    cloud_path = os.path.join(dataset_path, "sfm", "reconstruction.ply")
  if not os.path.exists(cloud_path):
    return None

  from libs.ply import read_ply
  cloud = read_ply(cloud_path)
  xyz = np.stack([np.asarray(cloud[axis]) for axis in ("x", "y", "z")], axis=1).astype(np.float64)
  rgb = np.stack([np.asarray(cloud[channel]) for channel in ("blue", "green", "red")], axis=1)

  local = result["cam_from_world"] * xyz
  front = local[:, 2] > 0
  pixels = np.asarray(camera.img_from_cam(local[front]))
  rgb, depth = rgb[front], local[front, 2]

  height, width = frame.shape[:2]
  columns = np.round(pixels[:, 0]).astype(int)
  lines = np.round(pixels[:, 1]).astype(int)
  inside = (columns >= 0) & (columns < width) & (lines >= 0) & (lines < height)

  panel = (frame * 0.25).astype(np.uint8)
  order = np.argsort(-depth[inside])  # back to front, the closest points stay visible
  panel[lines[inside][order], columns[inside][order]] = rgb[inside][order]
  return panel, int(inside.sum())


def residual_color(residual, max_error):
  """Green for a residual near zero, red at the RANSAC threshold."""
  fraction = min(residual / max_error, 1.0)
  return (0, int(255 * (1 - fraction)), int(255 * fraction))


def build_overlay(dataset_path, query_path, result, camera, max_error, output_path, drawn_matches=40):
  """Save a two panel check of the estimate: the inlier matches and the projected model.

  Without ground truth this is the only honest way to judge a relocalisation. Left the
  query with its inlier keypoints, right the reconstruction seen from the estimated
  pose, and a line across the two panels per inlier, from the keypoint to where its
  3D point reprojects. Both panels share the same viewpoint, so the lines come out
  parallel when the pose is right and fan out when it is not. They are coloured from
  green to red over the RANSAC threshold.
  """
  frame = cv2.imread(query_path)
  inliers = result["inlier_mask"]
  points2D = result["points2D"][inliers]
  projected = np.asarray(camera.img_from_cam(result["cam_from_world"] * result["points3D"][inliers]))
  residuals = np.linalg.norm(projected - points2D, axis=1)

  matches = frame.copy()
  for (x, y), residual in zip(points2D, residuals):
    cv2.circle(matches, (int(x), int(y)), 3, residual_color(residual, max_error), -1, cv2.LINE_AA)

  panels = [matches]
  cloud = project_cloud(dataset_path, result, camera, frame)
  if cloud is None:
    print_warning("No point cloud to draw the projection with")
  else:
    panel, drawn = cloud
    cv2.putText(panel, f"model reprojected, {drawn} points",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    panels.append(panel)

  overlay = np.hstack(panels)
  if len(panels) > 1:
    # a line per inlier would be a wall of ink, an even sample reads the same
    step = max(len(points2D) // drawn_matches, 1)
    for (x, y), (px, py), residual in zip(points2D[::step], projected[::step], residuals[::step]):
      color = residual_color(residual, max_error)
      cv2.line(overlay, (int(x), int(y)), (frame.shape[1] + int(px), int(py)), color, 1, cv2.LINE_AA)
      cv2.circle(overlay, (frame.shape[1] + int(px), int(py)), 3, color, -1, cv2.LINE_AA)

  cv2.putText(overlay, f"{int(inliers.sum())} inliers, {result['reprojection_error']:.2f} px mean",
              (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imwrite(output_path, overlay)
  print_info(f"Saved the verification overlay to {output_path}")


def save_output(output_dir, image_path, query_path, relocation_data, dataset_path, result, camera, max_error):
  os.makedirs(output_dir, exist_ok=True)
  name = os.path.splitext(os.path.basename(image_path))[0]

  shutil.copy2(image_path, os.path.join(output_dir, os.path.basename(image_path)))
  with open(os.path.join(output_dir, f"{name}.json"), "w") as handle:
    json.dump(relocation_data, handle, indent=2)
  print_info(f"Saved relocation data to {os.path.join(output_dir, f'{name}.json')}")

  if result.get("success"):
    build_overlay(dataset_path, query_path, result, camera, max_error,
                  os.path.join(output_dir, f"{name}_overlay.jpg"))

##############################################################################
# RELOCATE
##############################################################################

def relocate(dataset_path, image_path, output_dir=None, retrieved=10, ratio=0.8,
             max_error=4.0, use_dense=False):
  """Locate one photo in a dataset, and write the JSON and the overlay if asked to.

  Returns the relocation data, the same dict that lands in the JSON.
  """
  with open(os.path.join(dataset_path, "config.json")) as handle:
    image_max_dimension = json.load(handle)["image_max_dimension"]

  reconstruction = pycolmap.Reconstruction(os.path.join(dataset_path, "sfm", "0"))
  print_info(f"Loaded reconstruction: {len(reconstruction.images)} images, {len(reconstruction.points3D)} 3D points")

  tmp_dir = tempfile.mkdtemp(prefix="reloc_")
  try:
    print_step("Prepare Query Image")
    query_path, size = prepare_query_image(image_path, tmp_dir, image_max_dimension)

    print_step("Extract Query Features")
    keypoints, descriptors = extract_query_features(query_path, log=print_info)

    print_step("Build Query Camera Model")
    exif_camera = None
    try:
      exif_camera = pycolmap.infer_camera_from_image(query_path)
    except Exception:
      pass
    camera, camera_source = query_camera(reconstruction, size, os.path.basename(image_path), exif_camera)
    print_info(f"Camera from {camera_source}: {camera.model.name} {camera.width}x{camera.height} "
               f"f={camera.params[0]:.1f}")

    print_step("Localize")
    result = localize(dataset_path, keypoints, descriptors, camera, num_retrieved=retrieved,
                      ratio=ratio, max_error=max_error, use_dense=use_dense, log=print_info)

    print_step("Results")
    if not result["success"]:
      print_error(f"Pose estimation failed: {result['reason']}")
    else:
      center = -result["cam_from_world"].rotation.matrix().T @ result["cam_from_world"].translation
      print_success("Pose estimated")
      print_info(f"  Inliers: {result['num_inliers']}/{result['num_correspondences']} "
                 f"({100 * result['inlier_ratio']:.0f}%), mean reprojection error "
                 f"{result['reprojection_error']:.2f} px")
      print_info(f"  Camera center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
      # which of the two sources carried the pose, the whole point of the --dense flag
      fallback = {"off": "not requested", "unavailable": "requested, no depth maps in the dataset",
                  "on": "on"}[result["dense_fallback"]]
      print_info(f"  Points: {result['num_sparse_inliers']} triangulated inliers, "
                 f"{result['num_dense_inliers']} lifted from dense depth "
                 f"(dense fallback {fallback})")
      # the inlier ratio is not a quality signal here, correspondences are collected
      # generously on purpose, but a badly fitting inlier set is
      if result["reprojection_error"] > 2.0:
        print_warning("Weak support: check the overlay before trusting this pose")

    data = build_relocation_data(result, image_path, dataset_path, camera, camera_source)
    if output_dir:
      save_output(output_dir, image_path, query_path, data, dataset_path, result, camera, max_error)
    return data
  finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

##############################################################################
# MAIN
##############################################################################

def main():
  args = parse_args()
  database_path = os.path.join(args.dataset, "database.db")
  config_path = os.path.join(args.dataset, "config.json")

  for path, label in [(args.dataset, "dataset"), (database_path, "database"),
                      (args.image, "query image"),
                      (config_path, "config.json (run pipeline.py first)")]:
    if not os.path.exists(path):
      print_error(f"{label} not found: {path}")
      sys.exit(1)

  data = relocate(args.dataset, args.image, args.output, retrieved=args.retrieved,
                  ratio=args.ratio, max_error=args.max_error, use_dense=args.dense)
  sys.exit(0 if data["success"] else 1)


if __name__ == "__main__":
  main()
