"""Trains and exports the gaussian splat of a dataset.

Launched by pipeline.py as `python -m libs.splat_trainer`, never imported by it:
torch and pycolmap each link their own libomp and abort when they share a process,
so the training gets its own.
"""

import os
import json
import time
import argparse

import numpy as np
import cv2
import torch

from libs import cpu_splatting
from libs.ply import read_ply
from libs.console import print_error, print_success, print_info, print_warning

def parse_args():
  parser = argparse.ArgumentParser(description="Train a 3D Gaussian Splatting scene on CPU from a COLMAP dense reconstruction")
  parser.add_argument("--dataset", required=True, help="Path to dataset directory (e.g. storage/datasets/home)")
  parser.add_argument("--iterations", type=int, default=2000, help="Number of optimization steps")
  parser.add_argument("--max-size", type=int, default=400, help="Max image dimension used for training: higher is sharper and slower")
  parser.add_argument("--max-gaussians", type=int, default=60000, help="Upper bound on the gaussians sampled from the dense point cloud")
  parser.add_argument("--capacity", type=int, default=64, help="Gaussians composited per tile, front to back")
  parser.add_argument("--holdout", type=int, default=8, help="Keep every Nth view out of training to measure novel view quality (0 disables)")
  parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Torch device: mps uses the Apple GPU, ~2.5x faster")
  parser.add_argument("--threads", type=int, default=None, help="Torch CPU threads (defaults to CPU count - 2)")
  return parser.parse_args()

def resolve_device(name):
  if name == "mps" and not torch.backends.mps.is_available():
    print_warning("MPS is not available, falling back to CPU.")
    return "cpu"
  return name

def load_views(dense_path, max_size, holdout, device):
  """Load every view, then split off the holdout slice used to measure novel views."""
  views = cpu_splatting.load_views(dense_path, max_size, device)
  print_info(f"Loaded {len(views)} views at {views[0]['size'][1]}x{views[0]['size'][0]}")

  train = [view for index, view in enumerate(views) if not holdout or index % holdout]
  test = [view for index, view in enumerate(views) if holdout and not index % holdout]
  print_info(f"Training on {len(train)} views, {len(test)} held out")
  return train, test

def load_gaussians(dense_path, max_gaussians, device):
  cloud = read_ply(os.path.join(dense_path, "fused.ply"))
  points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1)
  colors = np.stack([cloud["red"], cloud["green"], cloud["blue"]], axis=1).astype(np.float32) / 255.0
  gaussians = cpu_splatting.init_gaussians(points, colors, device, max_gaussians=max_gaussians)
  print_info(f"Initialized {len(gaussians['means'])} gaussians from {len(points)} dense points")
  return gaussians

def save_comparisons(gaussians, views, capacity, renders_path, count=3):
  """A few photo | render pairs, to eyeball the result without a splat viewer."""
  os.makedirs(renders_path, exist_ok=True)
  for view in views[:: max(1, len(views) // count)][:count]:
    with torch.no_grad():
      prediction = cpu_splatting.render(gaussians, view, capacity=capacity).clamp(0.0, 1.0)
    pair = np.hstack([view["image"].permute(1, 2, 0).cpu().numpy(), prediction.permute(1, 2, 0).cpu().numpy()])
    path = os.path.join(renders_path, f"{os.path.splitext(view['name'])[0]}.jpg")
    cv2.imwrite(path, (pair[:, :, ::-1] * 255).astype(np.uint8))
    print_info(f"Saved comparison (photo | render) to {path}")

def main():
  args = parse_args()

  dense_path = os.path.join(args.dataset, "dense")
  splat_path = os.path.join(args.dataset, "splat")
  if not os.path.exists(os.path.join(dense_path, "fused.ply")):
    print_error(f"Dense point cloud not found in {dense_path}. Run the pipeline dense steps first.")
    return 1
  os.makedirs(splat_path, exist_ok=True)

  torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 2))
  device = resolve_device(args.device)
  print_info(f"Training on {device} with {torch.get_num_threads()} threads")

  time_start = time.time()
  train_views, test_views = load_views(dense_path, args.max_size, args.holdout, device)
  gaussians = load_gaussians(dense_path, args.max_gaussians, device)

  initial_psnr, initial_ssim = cpu_splatting.evaluate(gaussians, train_views, args.capacity)
  print_info(f"Before training: psnr {initial_psnr:.2f} dB, ssim {initial_ssim:.4f}")

  gaussians, history = cpu_splatting.train(
    gaussians, train_views, args.iterations, capacity=args.capacity, log=print_info,
  )

  train_psnr, train_ssim = cpu_splatting.evaluate(gaussians, train_views, args.capacity)
  print_success(f"Training views: psnr {train_psnr:.2f} dB, ssim {train_ssim:.4f}")
  test_psnr, test_ssim = (None, None)
  if test_views:
    test_psnr, test_ssim = cpu_splatting.evaluate(gaussians, test_views, args.capacity)
    print_success(f"Holdout views: psnr {test_psnr:.2f} dB, ssim {test_ssim:.4f}")

  splat_ply_path = os.path.join(splat_path, "point_cloud.ply")
  cpu_splatting.export_ply(splat_ply_path, gaussians)
  print_success(f"{len(gaussians['means'])} gaussians exported to {splat_ply_path}")
  save_comparisons(gaussians, train_views, args.capacity, os.path.join(splat_path, "renders"))

  metrics = {
    "iterations": args.iterations,
    "gaussians": len(gaussians["means"]),
    "max_image_size": args.max_size,
    "capacity": args.capacity,
    "device": device,
    "seconds": round(time.time() - time_start, 1),
    "initial": {"psnr": initial_psnr, "ssim": initial_ssim},
    "train": {"psnr": train_psnr, "ssim": train_ssim, "views": len(train_views)},
    "holdout": {"psnr": test_psnr, "ssim": test_ssim, "views": len(test_views)},
    "history": history,
  }
  metrics_path = os.path.join(splat_path, "metrics.json")
  with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
  print_info(f"Metrics saved to {metrics_path}")
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
