import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import pycolmap
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from libs.read_write_model import read_cameras_binary, write_cameras_text, read_images_binary, write_images_text, read_points3D_binary, write_points3D_text
from libs import cpu_mvs
from libs.console import print_error, print_success, print_info, print_warning, print_step

IMAGE_MAX_DIMENSION = 1024
# Upper bound on the photos fed to the pipeline. Matching is quadratic in their number,
# so a dense capture costs far more than it adds. Set to 0 to use them all.
IMAGE_MAX_ITEMS = 300
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# COLMAP's patch match stereo needs CUDA, so the depth maps come from libs/cpu_mvs.py.
# Those maps are noisier than the GPU ones, hence the relaxed fusion thresholds.
DENSE_MIN_NUM_PIXELS = 3
DENSE_MAX_NORMAL_ERROR = 25.0

##############################################################################
# ARGS
##############################################################################

def parse_args():
  parser = argparse.ArgumentParser(description="COLMAP SfM pipeline")
  parser.add_argument("--dataset", required=True, help="Path to dataset directory (e.g. storage/datasets/home)")
  parser.add_argument("--reset", action="store_true", help="Reset the dataset by deleting existing images, database, SFM, dense and splat reconstruction")
  parser.add_argument("--dense-max-size", type=int, default=640, help="Max image dimension used to match depth maps: higher is denser and slower")
  parser.add_argument("--dense-num-src", type=int, default=6, help="Number of source views matched against each image")
  parser.add_argument("--dense-num-samples", type=int, default=128, help="Number of depth planes swept per image")
  parser.add_argument("--dense-num-workers", type=int, default=None, help="Images matched in parallel (defaults to CPU count + 2)")
  parser.add_argument("--splat-iterations", type=int, default=2000, help="Gaussian splatting optimization steps")
  parser.add_argument("--splat-max-size", type=int, default=400, help="Max image dimension used to train the splat: higher is sharper and slower")
  parser.add_argument("--splat-max-gaussians", type=int, default=60000, help="Upper bound on the gaussians sampled from the dense point cloud")
  parser.add_argument("--splat-capacity", type=int, default=64, help="Gaussians composited per tile, front to back")
  parser.add_argument("--splat-warmup", type=float, default=0.5, help="Fraction of the splat iterations trained on half resolution views (0 disables)")
  parser.add_argument("--splat-holdout", type=int, default=8, help="Keep every Nth view out of the splat training to measure novel view quality (0 disables)")
  parser.add_argument("--splat-device", default="cpu", choices=["cpu", "mps"], help="Torch device used to train the splat (mps is slower here, see AGENTS.md)")
  return parser.parse_args()

##############################################################################
# BUILD IMAGES
##############################################################################

def select_images(filenames, max_items):
  """Spread max_items picks over the sorted photos, dropping the rest.

  Cutting the tail would leave a hole in the scene, so the capture is decimated
  uniformly instead: 400 photos capped at 300 keep three and skip one, all the way
  through. Names are sorted first, a capture is usually named in order.
  """
  filenames = sorted(filenames)
  if not max_items or len(filenames) <= max_items:
    return filenames
  # the stride is above one, so no two picks can round to the same photo
  return [filenames[round(i * len(filenames) / max_items)] for i in range(max_items)]


def build_images(train_path, images_path, max_items=IMAGE_MAX_ITEMS):
  if os.path.exists(images_path) and os.listdir(images_path):
    print_info(f"Images path {images_path} already exists and is not empty. Skipping image build.")
    return

  if not os.path.exists(train_path):
    print_error(f"Train path {train_path} does not exist. Cannot build images.")
    return

  os.makedirs(images_path, exist_ok=True)

  photos = [f for f in os.listdir(train_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
  selected = select_images(photos, max_items)
  if len(selected) < len(photos):
    print_warning(f"Using {len(selected)} of the {len(photos)} photos in {train_path}, "
                  f"capped by IMAGE_MAX_ITEMS")

  def process_image(filename):
    source_path = os.path.join(train_path, filename)
    dest_path = os.path.join(images_path, filename)
    try:
      with Image.open(source_path) as img:
        width, height = img.size
        max_dimension = max(width, height)
        if max_dimension > IMAGE_MAX_DIMENSION:
          scale = IMAGE_MAX_DIMENSION / max_dimension
          new_size = (int(width * scale), int(height * scale))
          img = img.resize(new_size, Image.Resampling.LANCZOS)
          print_info(f"Resized image {filename} from ({width}, {height}) to {new_size}.")
        img.save(dest_path)
        print_success(f"Copied image {filename} to images path.")
    except Exception as e:
      print_error(f"Failed to process image {filename}: {e}")

  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_image, f): f for f in selected}
    for future in as_completed(futures):
      future.result()

##############################################################################
# EXTRACT FEATURES
##############################################################################

def extract_features(database_path, images_path):
  if not os.path.exists(images_path) or not os.listdir(images_path):
    print_error(f"Images path {images_path} does not exist or is empty. Cannot extract features.")
    return

  print_info("Extracting features using pycolmap...")
  pycolmap.extract_features(database_path, images_path)
  print_success("Feature extraction completed.")

##############################################################################
# MATCH FEATURES
##############################################################################

def match_features(database_path):
  if not os.path.exists(database_path):
    print_error(f"Database path {database_path} does not exist. Cannot perform matching.")
    return

  print_info("Performing exhaustive matching using pycolmap...")
  pycolmap.match_exhaustive(database_path)
  print_success("Exhaustive matching completed.")

##############################################################################
# BUILD SFM RECONSTRUCTION
##############################################################################

def build_sfm_reconstruction(database_path, images_path, sfm_path):
  if not os.path.exists(images_path) or not os.listdir(images_path):
    print_error(f"Images path {images_path} does not exist or is empty. Cannot build SFM reconstruction.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  if os.path.exists(sfm_reconstruction_path) and os.listdir(sfm_reconstruction_path):
    print_info(f"SFM reconstruction path {sfm_reconstruction_path} already exists and is not empty. Skipping SFM reconstruction.")
    return

  if not os.path.exists(sfm_path):
    os.makedirs(sfm_path, exist_ok=True)

  print_info("Running incremental mapping to build SFM reconstruction using pycolmap...")
  reconstructions = pycolmap.incremental_mapping(database_path, images_path, sfm_path)
  reconstruction = reconstructions[0]
  print(reconstruction.summary())
  print_success("SFM reconstruction completed.")

##############################################################################
# BUILD SFM RECONSTRUCTION PLY
##############################################################################

def build_sfm_reconstruction_ply(sfm_path):
  if not os.path.exists(sfm_path):
    print_error(f"SFM path {sfm_path} does not exist. Cannot build SFM reconstruction PLY.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  sfm_reconstruction_ply_path = os.path.join(sfm_path, "reconstruction.ply")
  if os.path.exists(sfm_reconstruction_ply_path):
    print_info(f"SFM reconstruction PLY path {sfm_reconstruction_ply_path} already exists. Skipping PLY export.")
    return
  print_info("Exporting SFM reconstruction to PLY using pycolmap...")
  reconstruction = pycolmap.Reconstruction(sfm_reconstruction_path)
  reconstruction.export_PLY(sfm_reconstruction_ply_path)
  print_success(f"SFM reconstruction exported to {sfm_reconstruction_ply_path}.")

##############################################################################
# BUILD SFM RECONSTRUCTION TXT
##############################################################################

def build_sfm_reconstruction_txt(sfm_path):
  if not os.path.exists(sfm_path):
    print_error(f"SFM path {sfm_path} does not exist. Cannot build SFM reconstruction TXT.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  cameras_bin_path = os.path.join(sfm_reconstruction_path, "cameras.bin")
  cameras_txt_path = os.path.join(sfm_reconstruction_path, "cameras.txt")
  images_bin_path = os.path.join(sfm_reconstruction_path, "images.bin")
  images_txt_path = os.path.join(sfm_reconstruction_path, "images.txt")
  points3D_bin_path = os.path.join(sfm_reconstruction_path, "points3D.bin")
  points3D_txt_path = os.path.join(sfm_reconstruction_path, "points3D.txt")

  if os.path.exists(cameras_txt_path) and os.path.exists(images_txt_path):
    print_info(f"SFM reconstruction TXT files already exist. Skipping TXT export.")
    return

  print_info("Converting SFM reconstruction from BIN to TXT...")
  cameras = read_cameras_binary(cameras_bin_path)
  write_cameras_text(cameras, cameras_txt_path)
  print_success(f"SFM reconstruction cameras exported to {cameras_txt_path}.")
  images = read_images_binary(images_bin_path)
  write_images_text(images, images_txt_path)
  print_success(f"SFM reconstruction images exported to {images_txt_path}.")
  points3D = read_points3D_binary(points3D_bin_path)
  write_points3D_text(points3D, points3D_txt_path)
  print_success(f"SFM reconstruction points3D exported to {points3D_txt_path}.")

##############################################################################
# BUILD SFM RECONSTRUCTION transforms.json
##############################################################################

def build_sfm_reconstruction_transforms_json(images_path, sfm_path):
  if not os.path.exists(sfm_path):
    print_error(f"SFM path {sfm_path} does not exist. Cannot build SFM reconstruction transforms.json.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  sfm_transforms_json_path = os.path.join(sfm_path, "transforms.json")

  if os.path.exists(sfm_transforms_json_path):
    print_info(f"SFM reconstruction transforms.json path {sfm_transforms_json_path} already exists. Skipping transforms.json export.")
    return

  print_info("Exporting SFM reconstruction to transforms.json using colmap2nerf...")
  command = [
    sys.executable, "-m", "libs.colmap2nerf",
    "--colmap_matcher", "exhaustive",
    "--aabb_scale", "16",
    "--images", os.path.abspath(images_path),
    "--text", os.path.abspath(sfm_reconstruction_path),
    "--out", os.path.abspath(sfm_transforms_json_path),
  ]
  subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__)))

  if not os.path.exists(sfm_transforms_json_path):
    print_error(f"Failed to export transforms.json to {sfm_transforms_json_path}.")
    return
  print_success(f"SFM reconstruction exported to {sfm_transforms_json_path}.")

##############################################################################
# BUILD DENSE WORKSPACE
##############################################################################

def build_dense_workspace(sfm_path, images_path, dense_path):
  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  if not os.path.exists(sfm_reconstruction_path):
    print_error(f"SFM reconstruction path {sfm_reconstruction_path} does not exist. Cannot build dense workspace.")
    return False

  if os.path.exists(os.path.join(dense_path, "sparse")):
    print_info(f"Dense workspace {dense_path} already exists. Skipping undistortion.")
    return True

  print_info("Undistorting images into the dense workspace using pycolmap...")
  pycolmap.undistort_images(dense_path, sfm_reconstruction_path, images_path)
  print_success(f"Dense workspace prepared in {dense_path}.")
  return True

##############################################################################
# BUILD DENSE DEPTH MAPS
##############################################################################

def build_dense_depth_maps(dense_path, max_image_size, num_src_images, num_samples, num_workers):
  if not os.path.exists(os.path.join(dense_path, "sparse")):
    print_error(f"Dense workspace {dense_path} does not exist. Cannot build depth maps.")
    return

  print_info("Estimating depth and normal maps on CPU (plane sweep, no CUDA required)...")
  cpu_mvs.build_depth_maps(
    dense_path,
    max_image_size=max_image_size,
    num_src_images=num_src_images,
    num_samples=num_samples,
    num_workers=num_workers,
    log=print_info,
  )
  print_success("Depth and normal maps completed.")

##############################################################################
# BUILD DENSE POINT CLOUD
##############################################################################

def build_dense_point_cloud(dense_path):
  depth_maps_path = os.path.join(dense_path, "stereo", "depth_maps")
  if not os.path.exists(depth_maps_path) or not os.listdir(depth_maps_path):
    print_error(f"Depth maps path {depth_maps_path} does not exist or is empty. Cannot fuse the dense point cloud.")
    return

  dense_ply_path = os.path.join(dense_path, "fused.ply")
  if os.path.exists(dense_ply_path):
    print_info(f"Dense point cloud {dense_ply_path} already exists. Skipping fusion.")
    return

  print_info("Fusing depth maps into a dense point cloud using pycolmap...")
  options = pycolmap.StereoFusionOptions()
  options.min_num_pixels = DENSE_MIN_NUM_PIXELS
  options.max_normal_error = DENSE_MAX_NORMAL_ERROR
  reconstruction = pycolmap.stereo_fusion(dense_ply_path, dense_path, input_type="geometric", options=options)
  print_success(f"Dense point cloud with {len(reconstruction.points3D)} points exported to {dense_ply_path}.")

##############################################################################
# BUILD SPLAT
##############################################################################

def build_splat(dataset_path, args):
  dense_ply_path = os.path.join(dataset_path, "dense", "fused.ply")
  if not os.path.exists(dense_ply_path):
    print_error(f"Dense point cloud {dense_ply_path} does not exist. Cannot train the splat.")
    return

  splat_ply_path = os.path.join(dataset_path, "splat", "point_cloud.ply")
  if os.path.exists(splat_ply_path):
    print_info(f"Splat {splat_ply_path} already exists. Skipping training.")
    return

  # the trainer runs in its own process: torch and pycolmap each link their own
  # libomp and abort as soon as they share one
  print_info("Training the gaussian splat with libs/splat_trainer.py...")
  command = [
    sys.executable, "-m", "libs.splat_trainer",
    "--dataset", os.path.abspath(dataset_path),
    "--iterations", str(args.splat_iterations),
    "--max-size", str(args.splat_max_size),
    "--max-gaussians", str(args.splat_max_gaussians),
    "--capacity", str(args.splat_capacity),
    "--warmup", str(args.splat_warmup),
    "--holdout", str(args.splat_holdout),
    "--device", args.splat_device,
  ]
  if subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__))).returncode != 0:
    print_error("Gaussian splatting training failed.")
    return
  print_success(f"Splat exported to {splat_ply_path}.")

##############################################################################
# MAIN
##############################################################################

def main():
  args = parse_args()

  dataset_path = args.dataset
  train_path = os.path.join(dataset_path, 'train')
  images_path = os.path.join(dataset_path, 'images')
  database_path = os.path.join(dataset_path, 'database.db')
  sfm_path = os.path.join(dataset_path, 'sfm')
  dense_path = os.path.join(dataset_path, 'dense')
  splat_path = os.path.join(dataset_path, 'splat')

  if not os.path.exists(dataset_path):
    print_error(f"Dataset path {dataset_path} does not exist.")
    sys.exit(1)

  config_path = os.path.join(dataset_path, 'config.json')
  config = {"image_max_dimension": IMAGE_MAX_DIMENSION, "image_max_items": IMAGE_MAX_ITEMS}
  with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
  print_info(f"Config saved to {config_path}")

  if args.reset:
    if os.path.exists(images_path):
      shutil.rmtree(images_path)
    if os.path.exists(database_path):
      os.remove(database_path)
    if os.path.exists(sfm_path):
      shutil.rmtree(sfm_path)
    if os.path.exists(dense_path):
      shutil.rmtree(dense_path)
    if os.path.exists(splat_path):
      shutil.rmtree(splat_path)

  time_start = time.time()
  print_step("🚀 Build Images")
  build_images(train_path, images_path)
  print_step("🚀 Extract Features")
  extract_features(database_path, images_path)
  print_step("🚀 Match Features")
  match_features(database_path)
  print_step("🚀 Build SFM Reconstruction")
  build_sfm_reconstruction(database_path, images_path, sfm_path)
  print_step("🚀 Build SFM Reconstruction PLY")
  build_sfm_reconstruction_ply(sfm_path)
  print_step("🚀 Build SFM Reconstruction TXT")
  build_sfm_reconstruction_txt(sfm_path)
  print_step("🚀 Build SFM Reconstruction transforms.json")
  build_sfm_reconstruction_transforms_json(images_path, sfm_path)
  print_step("🚀 Build Dense Workspace")
  build_dense_workspace(sfm_path, images_path, dense_path)
  print_step("🚀 Build Dense Depth Maps")
  build_dense_depth_maps(dense_path, args.dense_max_size, args.dense_num_src, args.dense_num_samples, args.dense_num_workers)
  print_step("🚀 Build Dense Point Cloud")
  build_dense_point_cloud(dense_path)
  print_step("🚀 Build Gaussian Splat")
  build_splat(dataset_path, args)

  print_step("✅ Pipeline completed")
  time_end = time.time()
  time_total = time_end - time_start
  print_success(f"Total execution time: {time_total:.2f} seconds")

if __name__ == "__main__":
  main()
