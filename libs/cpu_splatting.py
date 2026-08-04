"""3D Gaussian Splatting trained on CPU.

The reference implementations rasterise gaussians with a CUDA kernel, which rules
them out on Apple Silicon. This module is a self-contained PyTorch rewrite of the
pieces that kernel provides, so training runs on plain CPU tensors (autograd takes
care of the backward pass):

  dense/fused.ply -> gaussians -> tiled differentiable rasteriser -> splat/point_cloud.ply

The exported PLY uses the original 3DGS layout, so any standard viewer opens it.

Design choices that keep CPU training tractable:
- gaussians are initialised on the dense MVS points, so no densification is needed,
  only pruning of the ones that end up transparent or oversized,
- colours are view-independent (SH degree 0): the extra bands trade CPU time for
  view-dependent highlights, which is a poor deal here,
- pixels are grouped in small tiles, each compositing its `capacity` closest
  gaussians: sorted front to back, transmittance saturates well before that limit,
  and small tiles keep the (tile, gaussian) pairs close to the real footprints.

Never import this module together with pycolmap: torch and pycolmap ship different
libomp copies and the process aborts. Camera poses are therefore read from the
COLMAP binary files directly, not through pycolmap.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from libs.ply import read_ply, write_ply
from libs.read_write_model import read_cameras_binary, read_images_binary

SH_DC = 0.28209479177387814  # value of the degree 0 spherical harmonic
TILE = 2                     # tile side in pixels: small tiles fit the splat footprints
MIN_DEPTH = 1e-3             # gaussians closer than this to a camera are culled
MAX_RADIUS = 64.0            # screen radius cap, keeps the tile lists bounded

##############################################################################
# CAMERAS
##############################################################################

def load_views(dense_path, max_image_size, device):
  """Load the undistorted images and their poses from a COLMAP dense workspace."""
  import cv2  # imported lazily: only needed to decode and resize the training images

  sparse_path = os.path.join(dense_path, "sparse")
  cameras = read_cameras_binary(os.path.join(sparse_path, "cameras.bin"))
  images = read_images_binary(os.path.join(sparse_path, "images.bin"))

  views = []
  for image in sorted(images.values(), key=lambda item: item.name):
    camera = cameras[image.camera_id]
    if camera.model != "PINHOLE":
      raise ValueError(f"Expected undistorted PINHOLE cameras, got {camera.model}")

    bitmap = cv2.imread(os.path.join(dense_path, "images", image.name), cv2.IMREAD_COLOR)
    if bitmap is None:
      raise IOError(f"Cannot read image {image.name}")
    scale = min(1.0, max_image_size / max(bitmap.shape[:2]))
    if scale < 1.0:
      bitmap = cv2.resize(bitmap, (round(bitmap.shape[1] * scale), round(bitmap.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    height, width = bitmap.shape[:2]

    fx, fy, cx, cy = camera.params
    views.append({
      "name": image.name,
      # (3, H, W) RGB in [0, 1]
      "image": torch.from_numpy(bitmap[:, :, ::-1].copy().astype(np.float32) / 255.0).permute(2, 0, 1).to(device),
      "focal": (fx * width / camera.width, fy * height / camera.height),
      "principal": (cx * width / camera.width - 0.5, cy * height / camera.height - 0.5),
      "R": torch.from_numpy(image.qvec2rotmat().astype(np.float32)).to(device),
      "t": torch.from_numpy(image.tvec.astype(np.float32)).to(device),
      "size": (height, width),
    })
  return views

##############################################################################
# GAUSSIAN INITIALISATION
##############################################################################

def init_gaussians(points, colors, device, max_gaussians=None, opacity=0.6, isolation=5.0, seed=0):
  """Create the trainable gaussian parameters from a coloured point cloud."""
  if max_gaussians and len(points) > max_gaussians:
    generator = np.random.default_rng(seed)
    keep = generator.choice(len(points), max_gaussians, replace=False)
    points, colors = points[keep], colors[keep]

  # one gaussian per point, sized after the local point spacing
  distances, _ = cKDTree(points).query(points, k=4)
  spacing = np.maximum(distances[:, 1:].mean(axis=1), 1e-6)

  # isolated points are MVS noise, and their spacing would seed huge gaussians
  # that smear over the whole image before training can shrink them
  dense = spacing < isolation * np.median(spacing)
  points, colors, spacing = points[dense], colors[dense], spacing[dense]

  quats = np.zeros((len(points), 4), dtype=np.float32)
  quats[:, 0] = 1.0

  def parameter(array):
    return torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device))

  return {
    "means": parameter(points),
    "log_scales": parameter(np.repeat(np.log(spacing)[:, None], 3, axis=1)),
    "quats": parameter(quats),
    "logit_opacity": parameter(np.full(len(points), math.log(opacity / (1.0 - opacity)))),
    "colors": parameter(colors),  # view-independent RGB in [0, 1]
  }

##############################################################################
# RASTERISER
##############################################################################

def _rotations(quats):
  """Rotation matrices of a batch of (w, x, y, z) quaternions."""
  w, x, y, z = torch.nn.functional.normalize(quats, dim=1).unbind(dim=1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ], dim=1).view(-1, 3, 3)

def _project(params, view, blur=0.3):
  """Project gaussians to screen space: 2D means, conics, radii, depths.

  The 2D covariance is the usual EWA splat, `Sigma_2d = J R Sigma R^t J^t`, with a
  small isotropic blur so that gaussians thinner than a pixel stay rasterisable.
  """
  camera_points = params["means"] @ view["R"].T + view["t"]
  depths = camera_points[:, 2]
  safe_depths = depths.clamp(min=MIN_DEPTH)

  focal_x, focal_y = view["focal"]
  principal_x, principal_y = view["principal"]
  means_2d = torch.stack([
    focal_x * camera_points[:, 0] / safe_depths + principal_x,
    focal_y * camera_points[:, 1] / safe_depths + principal_y,
  ], dim=1)

  # Sigma^(1/2) in camera space, so that Sigma_cam = M M^t
  scaled = _rotations(params["quats"]) * params["log_scales"].exp().unsqueeze(1)
  camera_scaled = view["R"] @ scaled

  zeros = torch.zeros_like(safe_depths)
  jacobian = torch.stack([
    focal_x / safe_depths, zeros, -focal_x * camera_points[:, 0] / safe_depths ** 2,
    zeros, focal_y / safe_depths, -focal_y * camera_points[:, 1] / safe_depths ** 2,
  ], dim=1).view(-1, 2, 3)

  projected = jacobian @ camera_scaled
  covariance = projected @ projected.transpose(1, 2)
  a = covariance[:, 0, 0] + blur
  b = covariance[:, 0, 1]
  c = covariance[:, 1, 1] + blur

  determinant = (a * c - b * b).clamp(min=1e-9)
  conics = torch.stack([c / determinant, -b / determinant, a / determinant], dim=1)

  # 3 sigma footprint from the largest eigenvalue of the 2D covariance
  middle = 0.5 * (a + c)
  eigenvalue = middle + (middle ** 2 - determinant).clamp(min=0.0).sqrt()
  radii = (3.0 * eigenvalue.sqrt()).clamp(max=MAX_RADIUS)
  return means_2d, conics, radii, depths

def _tile_lists(means_2d, conics, opacities, radii, depths, visible, tiles_x, tiles_y, capacity, tile):
  """Assign the visible gaussians to tiles, closest `capacity` ones per tile.

  Returns a (tiles, capacity) index tensor into the visible gaussians and its mask.
  Gaussians past the capacity of a tile are dropped: they sit behind the ones kept,
  where the accumulated transmittance has already gone to zero.
  """
  device = means_2d.device
  indices = visible.nonzero(as_tuple=True)[0]
  if len(indices) == 0:
    return None, None

  centers = means_2d[indices]
  low = ((centers - radii[indices, None]) / tile).floor().long()
  high = ((centers + radii[indices, None]) / tile).floor().long()
  low_x = low[:, 0].clamp(0, tiles_x - 1)
  low_y = low[:, 1].clamp(0, tiles_y - 1)
  span_x = high[:, 0].clamp(0, tiles_x - 1) - low_x + 1
  span_y = high[:, 1].clamp(0, tiles_y - 1) - low_y + 1

  # expand every gaussian into one entry per touched tile
  counts = span_x * span_y
  entries = torch.repeat_interleave(torch.arange(len(indices), device=device), counts)
  starts = torch.cumsum(counts, dim=0) - counts
  offsets = torch.arange(len(entries), device=device) - starts[entries]
  tile_x = low_x[entries] + offsets % span_x[entries]
  tile_y = low_y[entries] + offsets // span_x[entries]
  tiles = tile_y * tiles_x + tile_x

  # drop pairs whose gaussian cannot reach the visibility threshold inside the tile,
  # evaluated at the tile pixel closest to the projected centre
  corner = torch.stack([tile_x * tile, tile_y * tile], dim=1).float()
  offset = centers[entries].clamp(min=corner, max=corner + (tile - 1)) - centers[entries]
  conic = conics[indices][entries]
  peak = opacities[indices][entries] * torch.exp(
    -0.5 * (conic[:, 0] * offset[:, 0] ** 2 + 2 * conic[:, 1] * offset[:, 0] * offset[:, 1] + conic[:, 2] * offset[:, 1] ** 2)
  )
  reaches = peak > 1.0 / 255.0
  tiles, entries = tiles[reaches], entries[reaches]

  # sort by tile, and by depth within each tile, to composite front to back
  order = torch.argsort(depths[indices][entries])
  order = order[torch.argsort(tiles[order], stable=True)]
  tiles, entries = tiles[order], entries[order]

  # keep the first `capacity` entries of every tile
  per_tile = torch.bincount(tiles, minlength=tiles_x * tiles_y)
  ranks = torch.arange(len(tiles), device=device) - (torch.cumsum(per_tile, dim=0) - per_tile)[tiles]
  keep = ranks < capacity

  lists = torch.zeros((tiles_x * tiles_y, capacity), dtype=torch.long, device=device)
  mask = torch.zeros((tiles_x * tiles_y, capacity), dtype=torch.bool, device=device)
  lists[tiles[keep], ranks[keep]] = indices[entries[keep]]
  mask[tiles[keep], ranks[keep]] = True
  return lists, mask

def _chunk_alpha(params, means_2d, conics, indices, valid, pixels):
  """Opacity of the gaussians `indices` (tiles, chunk) at their tile pixels (tiles, P, 2)."""
  delta = pixels[:, None] - means_2d[indices][:, :, None]
  conic = conics[indices]
  power = -0.5 * (
    conic[:, :, None, 0] * delta[..., 0] ** 2
    + 2.0 * conic[:, :, None, 1] * delta[..., 0] * delta[..., 1]
    + conic[:, :, None, 2] * delta[..., 1] ** 2
  )
  alpha = (params["logit_opacity"].sigmoid()[indices][:, :, None] * power.exp()).clamp(max=0.99)
  return torch.where(valid[:, :, None], alpha, torch.zeros_like(alpha))

def render(params, view, background=0.0, capacity=64, tile=TILE, chunk=16, min_transmittance=0.01):
  """Rasterise the gaussians into the view. Differentiable, returns a (3, H, W) image.

  Tiles are composited a chunk of gaussians at a time and drop out of the loop once
  their transmittance is spent, so the cost follows the actual depth complexity
  instead of the worst case `capacity`.
  """
  height, width = view["size"]
  device = params["means"].device
  tiles_x = math.ceil(width / tile)
  tiles_y = math.ceil(height / tile)

  means_2d, conics, radii, depths = _project(params, view)
  visible = (
    (depths > MIN_DEPTH)
    & (radii > 0.5)
    & (means_2d[:, 0] + radii >= 0) & (means_2d[:, 0] - radii < width)
    & (means_2d[:, 1] + radii >= 0) & (means_2d[:, 1] - radii < height)
  )
  with torch.no_grad():
    lists, mask = _tile_lists(
      means_2d, conics, params["logit_opacity"].sigmoid(), radii, depths,
      visible, tiles_x, tiles_y, capacity, tile,
    )
  if lists is None:
    return torch.full((3, height, width), float(background), device=device)

  # pixel coordinates of every tile, tiles padded past the image are cropped at the end
  tile_index = torch.arange(tiles_x * tiles_y, device=device)
  inside = torch.arange(tile, device=device)
  pixels_x = ((tile_index % tiles_x) * tile)[:, None] + inside[None, :]
  pixels_y = ((tile_index // tiles_x) * tile)[:, None] + inside[None, :]
  pixels = torch.stack([
    pixels_x[:, None, :].expand(-1, tile, -1).reshape(len(tile_index), -1),
    pixels_y[:, :, None].expand(-1, -1, tile).reshape(len(tile_index), -1),
  ], dim=-1).float()  # (tiles, tile * tile, 2)

  # front to back alpha compositing, chunk by chunk, tiles leaving once opaque
  colors = torch.zeros((len(tile_index), pixels.shape[1], 3), device=device)
  transmittance = torch.ones((len(tile_index), pixels.shape[1]), device=device)
  active = mask[:, 0].nonzero(as_tuple=True)[0]

  for start in range(0, capacity, chunk):
    if len(active) == 0:
      break
    indices = lists[active, start:start + chunk]
    valid = mask[active, start:start + chunk]
    if not bool(valid.any()):
      break

    alpha = _chunk_alpha(params, means_2d, conics, indices, valid, pixels[active])
    inside_chunk = torch.cumprod(1.0 - alpha, dim=1)
    before = torch.cat([torch.ones_like(inside_chunk[:, :1]), inside_chunk[:, :-1]], dim=1)
    weights = alpha * before * transmittance[active][:, None]

    colors = colors.index_add(0, active, torch.einsum("tkp,tkc->tpc", weights, params["colors"][indices]))
    transmittance = transmittance.index_copy(0, active, transmittance[active] * inside_chunk[:, -1])
    active = active[transmittance[active].max(dim=1).values > min_transmittance]

  colors = colors + transmittance[:, :, None] * background

  # tiles back to a padded image, then crop to the view
  image = torch.zeros((tiles_y * tile * tiles_x * tile, 3), device=device)
  flat = (pixels_y[:, :, None] * (tiles_x * tile) + pixels_x[:, None, :]).reshape(len(tile_index), -1)
  image = image.index_add(0, flat.reshape(-1), colors.reshape(-1, 3))
  return image.view(tiles_y * tile, tiles_x * tile, 3)[:height, :width].permute(2, 0, 1)

##############################################################################
# LOSS
##############################################################################

def _gaussian_window(size=11, sigma=1.5, device=None):
  offsets = torch.arange(size, dtype=torch.float32, device=device) - size // 2
  weights = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
  return (weights / weights.sum()).view(1, 1, 1, size)

def ssim(prediction, target, window=None):
  """Mean SSIM between two (3, H, W) images, separable 11x11 gaussian window."""
  window = _gaussian_window(device=prediction.device) if window is None else window
  channels = prediction.shape[0]
  horizontal = window.expand(channels, 1, 1, -1)
  vertical = horizontal.transpose(2, 3)

  def blur(image):
    image = F.conv2d(image[None], horizontal, padding=(0, window.shape[-1] // 2), groups=channels)
    return F.conv2d(image, vertical, padding=(window.shape[-1] // 2, 0), groups=channels)[0]

  mean_prediction, mean_target = blur(prediction), blur(target)
  variance_prediction = blur(prediction * prediction) - mean_prediction ** 2
  variance_target = blur(target * target) - mean_target ** 2
  covariance = blur(prediction * target) - mean_prediction * mean_target

  c1, c2 = 0.01 ** 2, 0.03 ** 2
  numerator = (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
  denominator = (mean_prediction ** 2 + mean_target ** 2 + c1) * (variance_prediction + variance_target + c2)
  return (numerator / denominator).mean()

def photometric_loss(prediction, target, weight_ssim=0.2, window=None):
  return (1.0 - weight_ssim) * (prediction - target).abs().mean() + weight_ssim * (1.0 - ssim(prediction, target, window))

##############################################################################
# TRAINING
##############################################################################

def make_optimizer(params, scene_extent):
  """Adam with the per-parameter learning rates of the reference implementation."""
  return torch.optim.Adam([
    {"params": [params["means"]], "lr": 0.00016 * scene_extent},
    {"params": [params["log_scales"]], "lr": 0.005},
    {"params": [params["quats"]], "lr": 0.001},
    {"params": [params["logit_opacity"]], "lr": 0.05},
    {"params": [params["colors"]], "lr": 0.0025},
  ], eps=1e-15)

def prune(params, optimizer, scene_extent, min_opacity=0.005, max_scale_ratio=0.1):
  """Drop transparent and oversized gaussians, keeping the optimizer state aligned."""
  with torch.no_grad():
    keep = (params["logit_opacity"].sigmoid() > min_opacity) & (
      params["log_scales"].exp().max(dim=1).values < max_scale_ratio * scene_extent
    )
  if bool(keep.all()):
    return params, optimizer, 0

  removed = int((~keep).sum())
  for group in optimizer.param_groups:
    parameter = group["params"][0]
    state = optimizer.state.pop(parameter, None)
    with torch.no_grad():
      pruned = torch.nn.Parameter(parameter[keep].clone())
    if state is not None:
      state["exp_avg"] = state["exp_avg"][keep].clone()
      state["exp_avg_sq"] = state["exp_avg_sq"][keep].clone()
      optimizer.state[pruned] = state
    group["params"] = [pruned]
    for name, existing in params.items():
      if existing is parameter:
        params[name] = pruned
  return params, optimizer, removed

def downscale_views(views, factor=0.5):
  """The same views at a lower resolution, used to warm the training up cheaply."""
  scaled = []
  for view in views:
    # resized on the cpu: the mps adaptive pool rejects non divisible sizes
    image = F.interpolate(view["image"][None].cpu(), scale_factor=factor, mode="area")[0].to(view["image"].device)
    height, width = image.shape[-2:]
    scale_x, scale_y = width / view["size"][1], height / view["size"][0]
    scaled.append(dict(
      view,
      image=image,
      size=(height, width),
      focal=(view["focal"][0] * scale_x, view["focal"][1] * scale_y),
      principal=((view["principal"][0] + 0.5) * scale_x - 0.5, (view["principal"][1] + 0.5) * scale_y - 0.5),
    ))
  return scaled

def train(params, views, iterations, capacity, background=0.0, prune_every=500, seed=0,
          warmup=0.5, warmup_scale=0.5, log=print, log_every=100):
  """Optimise the gaussians against the training views. Returns the training history.

  The first `warmup` fraction of the iterations runs on half resolution views, where
  a step costs a quarter: measured on two datasets it cuts a third of the training
  time and leaves the holdout quality unchanged (see AGENTS.md).

  The reference implementation also decays the position learning rate, which is a
  loss here: its schedule spans 30k iterations, and compressing it into a couple of
  thousand starves the positions long before they have settled.
  """
  with torch.no_grad():
    center = params["means"].mean(dim=0)
    scene_extent = float((params["means"] - center).norm(dim=1).max())
  optimizer = make_optimizer(params, scene_extent)
  window = _gaussian_window(device=params["means"].device)
  generator = torch.Generator().manual_seed(seed)

  warmup_iterations = round(iterations * warmup)
  warmup_views = downscale_views(views, warmup_scale) if warmup_iterations else []
  if warmup_iterations:
    log(f"warming up for {warmup_iterations} iterations at {warmup_views[0]['size'][1]}x{warmup_views[0]['size'][0]}")

  history = []
  started = time.time()
  for iteration in range(1, iterations + 1):
    active = warmup_views if iteration <= warmup_iterations else views
    view = active[int(torch.randint(len(active), (1,), generator=generator))]
    prediction = render(params, view, background=background, capacity=capacity)
    loss = photometric_loss(prediction, view["image"], window=window)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
      params["colors"].clamp_(0.0, 1.0)

    if prune_every and iteration % prune_every == 0 and iteration < iterations:
      params, optimizer, removed = prune(params, optimizer, scene_extent)
      if removed:
        log(f"iteration {iteration}: pruned {removed} gaussians, {len(params['means'])} left")

    if iteration % log_every == 0 or iteration == 1:
      with torch.no_grad():
        error = float(((prediction - view["image"]) ** 2).mean())
      psnr = -10.0 * math.log10(error + 1e-12)
      history.append({"iteration": iteration, "loss": float(loss.detach()), "psnr": psnr})
      log(f"iteration {iteration}/{iterations}: loss {float(loss.detach()):.4f}, psnr {psnr:.2f} dB, {(time.time() - started) / iteration:.2f}s/it")

  return params, history

def evaluate(params, views, capacity, background=0.0):
  """Mean PSNR and SSIM of the rendered views."""
  psnrs, ssims = [], []
  with torch.no_grad():
    for view in views:
      prediction = render(params, view, background=background, capacity=capacity)
      psnrs.append(-10.0 * math.log10(float(((prediction - view["image"]) ** 2).mean()) + 1e-12))
      ssims.append(float(ssim(prediction, view["image"])))
  return float(np.mean(psnrs)), float(np.mean(ssims))

##############################################################################
# EXPORT
##############################################################################

def export_ply(path, params):
  """Write the gaussians in the 3DGS PLY layout every splat viewer expects."""
  with torch.no_grad():
    columns = {
      "x": params["means"][:, 0], "y": params["means"][:, 1], "z": params["means"][:, 2],
      "nx": torch.zeros(len(params["means"])), "ny": torch.zeros(len(params["means"])), "nz": torch.zeros(len(params["means"])),
      # viewers evaluate colour as 0.5 + SH_DC * f_dc
      "f_dc_0": (params["colors"][:, 0] - 0.5) / SH_DC,
      "f_dc_1": (params["colors"][:, 1] - 0.5) / SH_DC,
      "f_dc_2": (params["colors"][:, 2] - 0.5) / SH_DC,
      "opacity": params["logit_opacity"],
      "scale_0": params["log_scales"][:, 0], "scale_1": params["log_scales"][:, 1], "scale_2": params["log_scales"][:, 2],
      "rot_0": params["quats"][:, 0], "rot_1": params["quats"][:, 1], "rot_2": params["quats"][:, 2], "rot_3": params["quats"][:, 3],
    }
    write_ply(path, {name: value.detach().cpu().numpy() for name, value in columns.items()})
