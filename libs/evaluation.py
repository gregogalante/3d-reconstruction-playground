"""How far a model can be pushed before it stops relocalising, not how well it does it.

Localising a holdout with only itself hidden measures accuracy on the easiest query
there is: the neighbouring views sit a few percent of the scene radius away, the photo
comes from the same camera in the same light, and any healthy dataset lands within a
hundredth of a percent. That number saturates and says nothing about what a real query
will meet, so the report is built on margins instead.

Two axes, each a ladder climbed until the pose falls outside tolerance:

- **viewpoint**: hide the holdout *and* its k nearest views, so the query has to be
  solved from further and further off the capture path. The margin is the distance and
  the viewing angle to the nearest view still in the model at the last setting that
  worked — "it relocalises up to 21% of the scene radius and 12 degrees away from
  anything mapped".
- **appearance**: degrade the query itself, in resolution, focus and JPEG quality, and
  report the harshest level still solved. This is the axis that decides whether a photo
  taken later with another phone will register at all.

The ladders are climbed by binary search, which assumes a level that fails is not going
to pass further up. Three trials instead of five, and the assumption holds in practice:
both axes degrade the same evidence monotonically.

Ground truth is the pose bundle adjustment gave the holdout, a *pseudo* ground truth:
those 3D points were adjusted with the holdout's observations too. At a hundred images
that is 1% of the constraints; on a twenty image dataset read everything as optimistic.
What is hidden from every query, and why, is in `localizer.hide_image` and
`query_camera(exclude=...)`.
"""

import os
import tempfile
import time

import cv2
import numpy as np

from libs.localizer import load_dataset, localize, query_camera, extract_query_features

# A trial passes while the pose stays inside this tolerance. Past it the estimate is not
# usable for relocation even when RANSAC did register it, so it counts as the ladder's
# ceiling. Position is in % of the scene radius, rotation in degrees.
PASS_POSITION_PCT = 1.0
PASS_ROTATION_DEG = 2.0

# Views hidden around the holdout. 0 is the plain leave one out, the rest push the query
# off the capture path: on a 128 image capture 16 hidden views put the nearest remaining
# one at 42% of the scene radius and 21 degrees away.
#
# The ladders run further than looks reasonable on purpose. A well shot capture takes
# absurd punishment — south-building relocalises with 32 of its 128 views hidden, at a
# sixth of its resolution and JPEG quality 10 — and a ladder that stops before the model
# breaks measures the ladder, not the model. Rungs past the breaking point cost nothing,
# the binary search never visits them.
VIEWPOINT_LADDER = (0, 4, 8, 16, 32, 64)

# Query degradations, from untouched to harshest. Scale is a fraction of the dataset's
# own resolution, blur a gaussian sigma in pixels, jpeg an encoder quality.
APPEARANCE_LADDERS = {
  "scale": (1.0, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06),
  "blur": (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
  "jpeg": (95, 50, 30, 20, 10, 5, 2),
}

##############################################################################
# GEOMETRY
##############################################################################

def camera_centre(pose):
  """World position of a camera from its world to camera pose."""
  return -pose.rotation.matrix().T @ pose.translation


def scene_radius(reconstruction):
  """Median distance of the registered cameras from their centroid.

  A COLMAP model has an arbitrary scale, so a distance in model units says nothing on
  its own. Every distance in the report is also given against this radius.
  """
  centres = np.array([camera_centre(image.cam_from_world())
                      for image in reconstruction.images.values() if image.has_pose])
  return float(np.median(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))


def rotation_error(estimated, truth):
  """Angle of the rotation taking one orientation onto the other, in degrees."""
  cosine = (np.trace(estimated.matrix() @ truth.matrix().T) - 1.0) / 2.0
  return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def neighbours(reconstruction, image_id):
  """The other registered images, nearest camera first, with distance and angle.

  The angle is between the two viewing directions: two cameras a metre apart pointing
  the same way share far more of the scene than two in the same spot looking opposite.
  """
  image = reconstruction.images[image_id]
  pose = image.cam_from_world()
  centre, axis = camera_centre(pose), pose.rotation.matrix()[2]

  others = []
  for other_id, other in reconstruction.images.items():
    if other_id == image_id or not other.has_pose:
      continue
    other_pose = other.cam_from_world()
    distance = float(np.linalg.norm(camera_centre(other_pose) - centre))
    angle = float(np.degrees(np.arccos(np.clip(axis @ other_pose.rotation.matrix()[2], -1.0, 1.0))))
    others.append((distance, angle, other_id))
  return sorted(others)

##############################################################################
# HOLDOUTS
##############################################################################

def pick_holdouts(names, stride, minimum, maximum):
  """One image every `stride`, clamped to [minimum, maximum], the same ones every run.

  The count follows the size of the capture instead of being fixed, and the bounds keep
  the check meaningful on a small dataset and short on a large one. The picks are then
  spread uniformly over the sorted names rather than taken as a block: the point is to
  land on different viewpoints, and a capture is usually named in order.
  """
  ordered = sorted(names, key=lambda image_id: names[image_id])
  if stride <= 0:
    return []
  count = min(max(len(ordered) // stride, minimum), maximum, len(ordered))
  return [ordered[round(i * len(ordered) / count)] for i in range(count)]


def escalate(levels, run, control=None):
  """Climb a ladder until it breaks, return every trial and the last passing index.

  Level 0 is the untouched query and always runs, it is the control the rest is read
  against; `control` passes in a result already measured elsewhere. The remaining levels
  are a binary search: a level that fails is taken to mean the harder ones fail too.
  """
  trials = {0: control if control is not None else run(levels[0])}
  if not trials[0]["pass"]:
    return trials, -1

  best, low, high = 0, 1, len(levels) - 1
  while low <= high:
    middle = (low + high) // 2
    trials[middle] = run(levels[middle])
    if trials[middle]["pass"]:
      best, low = middle, middle + 1
    else:
      high = middle - 1
  return trials, best

##############################################################################
# TRIALS
##############################################################################

def measure(result, truth, radius):
  """Turn a localisation into errors against the pose the model gave the holdout."""
  if not result["success"]:
    return {"pass": False, "success": False, "reason": result["reason"],
            "num_inliers": result["num_inliers"]}

  error = float(np.linalg.norm(camera_centre(result["cam_from_world"]) - camera_centre(truth)))
  position = 100.0 * error / radius
  rotation = rotation_error(result["cam_from_world"].rotation, truth.rotation)
  return {
    "pass": position <= PASS_POSITION_PCT and rotation <= PASS_ROTATION_DEG,
    "success": True,
    "position_error": round(error, 5),
    "position_error_pct": round(position, 4),
    "rotation_error_deg": round(rotation, 4),
    "num_inliers": result["num_inliers"],
    "num_correspondences": result["num_correspondences"],
    "num_dense_inliers": result["num_dense_inliers"],
    "inlier_ratio": round(result["inlier_ratio"], 3),
    "reprojection_error": round(result["reprojection_error"], 3),
  }


def run_query(dataset, image_id, keypoints, descriptors, size, hidden, use_dense, max_error):
  """One localisation of a holdout, with itself and `hidden` other images out of reach."""
  reconstruction = dataset["reconstruction"]
  camera, _ = query_camera(reconstruction, size, exclude=reconstruction.images[image_id].name)
  return localize(None, keypoints, descriptors, camera, max_error=max_error, use_dense=use_dense,
                  dataset=dataset, exclude=hidden, log=lambda *args: None)


def viewpoint_trial(dataset, image_id, ranked, radius, hidden_count, use_dense, max_error):
  """Localise the holdout with its `hidden_count` nearest views hidden as well."""
  reconstruction = dataset["reconstruction"]
  image = reconstruction.images[image_id]
  calibrated = reconstruction.cameras[image.camera_id]
  hidden = [image_id] + [other_id for _, _, other_id in ranked[:hidden_count]]

  result = run_query(dataset, image_id, dataset["keypoints"][image_id], dataset["descriptors"][image_id],
                     (calibrated.width, calibrated.height), hidden, use_dense, max_error)
  trial = measure(result, image.cam_from_world(), radius)
  # what the query is actually being asked to bridge at this rung of the ladder
  distance, angle, _ = ranked[hidden_count]
  trial.update({"hidden_views": hidden_count,
                "gap_pct": round(100.0 * distance / radius, 2), "gap_deg": round(angle, 1)})
  return trial


def appearance_trial(dataset, image_id, radius, kind, level, use_dense, max_error, tmp_dir):
  """Localise the holdout from a degraded copy of its own photo."""
  reconstruction = dataset["reconstruction"]
  image = reconstruction.images[image_id]
  frame = cv2.imread(os.path.join(dataset["images_path"], image.name))

  if kind == "scale" and level < 1.0:
    frame = cv2.resize(frame, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
  elif kind == "blur" and level > 0:
    frame = cv2.GaussianBlur(frame, (0, 0), level)
  path = os.path.join(tmp_dir, "query.jpg")
  cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(level) if kind == "jpeg" else 95])

  keypoints, descriptors = extract_query_features(path)
  result = run_query(dataset, image_id, keypoints, descriptors, (frame.shape[1], frame.shape[0]),
                     [image_id], use_dense, max_error)
  trial = measure(result, image.cam_from_world(), radius)
  trial.update({"level": level, "num_keypoints": len(keypoints)})
  return trial

##############################################################################
# REPORT
##############################################################################

def summarise(values, more_is_better=True):
  """Median, worst and best over the holdouts.

  Margins read one way and errors the other, so which end is the bad one is told rather
  than assumed: a report where `worst` is sometimes the minimum and sometimes the
  maximum is a report nobody can read twice.
  """
  values = [value for value in values if value is not None]
  if not values:
    return None
  lowest, highest = round(float(np.min(values)), 4), round(float(np.max(values)), 4)
  return {"median": round(float(np.median(values)), 4),
          "worst": lowest if more_is_better else highest,
          "best": highest if more_is_better else lowest}


def summarise_levels(indices, ladder):
  """Same, over ladder positions: the levels themselves run in opposite directions.

  A lower scale is harsher and a higher blur sigma is, so the median and the worst case
  are taken on the rung reached and read back as the level standing there.
  """
  indices = [index for index in indices if index is not None]
  if not indices:
    return None
  return {"median": ladder[int(round(float(np.median(indices))))], "worst": ladder[int(np.min(indices))]}


def evaluate_viewpoint(dataset, image_id, radius, use_dense, max_error):
  """Climb the viewpoint ladder for one holdout, return its row."""
  reconstruction = dataset["reconstruction"]
  ranked = neighbours(reconstruction, image_id)
  ladder = [k for k in VIEWPOINT_LADDER if k < len(ranked)]

  trials, best = escalate(ladder, lambda k: viewpoint_trial(dataset, image_id, ranked, radius, k,
                                                            use_dense, max_error))
  row = {"name": reconstruction.images[image_id].name, **trials[0]}
  row.pop("pass", None)
  # the k = 0 gap is not a margin, it is how close the capture already passes by
  row["nearest_view_pct"], row["nearest_view_deg"] = row.pop("gap_pct"), row.pop("gap_deg")

  if best < 0:
    # it could not place itself with the whole model available: no margin to speak of,
    # counted as zero rather than dropped, or the failures would flatter the aggregate
    row.update({"margin_pct": 0.0, "margin_deg": 0.0, "hidden_views": 0})
    return row, trials

  passing = trials[best]
  row.update({"margin_pct": passing["gap_pct"], "margin_deg": passing["gap_deg"],
              "hidden_views": ladder[best], "capped": best == len(ladder) - 1})
  return row, trials


def evaluate_appearance(dataset, image_id, radius, control, use_dense, max_error, tmp_dir):
  """Climb every degradation ladder for one holdout, return the harshest level solved."""
  margins, count = {}, 0
  for kind, ladder in APPEARANCE_LADDERS.items():
    # the untouched query is the same image for all three ladders, so it is measured once
    trials, best = escalate(ladder, lambda level: appearance_trial(dataset, image_id, radius, kind,
                                                                  level, use_dense, max_error, tmp_dir),
                            control=control)
    margins[kind] = {"level": ladder[best] if best >= 0 else None, "rung": best if best >= 0 else None}
    count += len(trials) - 1  # the control is shared, not paid for per ladder
  return margins, count


def evaluate(dataset_path, stride=10, minimum=6, maximum=20, appearance_items=3, use_dense=False,
             max_error=4.0, dataset=None, log=print):
  """Both ladders over a spread of holdouts, as a report ready to be written as JSON."""
  dataset = dataset if dataset is not None else load_dataset(dataset_path)
  reconstruction = dataset["reconstruction"]
  radius = scene_radius(reconstruction)

  # the database holds every photo, the model only the ones it could register: a capture
  # that did not connect leaves the rest in sfm/1 and beyond, and an image this model
  # never placed has no pose to be checked against
  registered = {image_id: name for image_id, name in dataset["names"].items()
                if image_id in reconstruction.images and reconstruction.images[image_id].has_pose}
  if len(registered) < len(dataset["names"]):
    log(f"  {len(dataset['names']) - len(registered)} of {len(dataset['names'])} photos are not in "
        f"this model, holdouts are picked among the {len(registered)} it registered")
  holdouts = pick_holdouts(registered, stride, minimum, maximum)
  # the appearance axis costs a SIFT pass per trial, so it runs on a spread of the
  # holdouts rather than all of them
  degraded = holdouts[::max(len(holdouts) // appearance_items, 1)][:appearance_items] if appearance_items else []

  start = time.time()
  rows, localisations = [], 0
  with tempfile.TemporaryDirectory(prefix="reloc_eval_") as tmp_dir:
    for position, image_id in enumerate(holdouts, start=1):
      row, trials = evaluate_viewpoint(dataset, image_id, radius, use_dense, max_error)
      localisations += len(trials)
      if image_id in degraded and trials[0]["pass"]:
        row["appearance"], count = evaluate_appearance(dataset, image_id, radius, trials[0],
                                                       use_dense, max_error, tmp_dir)
        localisations += count
      rows.append(row)

      if not row["success"]:
        log(f"  [{position}/{len(holdouts)}] {row['name']}: failed on its own, {row['reason']}")
      else:
        appearance = row.get("appearance")
        log(f"  [{position}/{len(holdouts)}] {row['name']}: "
            f"{row['position_error_pct']:.3f}% error, margin {row['margin_pct']}% of radius / "
            f"{row['margin_deg']}° ({row['hidden_views']} views hidden)" +
            (f", holds at scale {appearance['scale']['level']}, blur {appearance['blur']['level']}, "
             f"jpeg {appearance['jpeg']['level']}" if appearance else ""))

  solved = [row for row in rows if row["success"]]
  capped = [row for row in solved if row.get("capped")]
  appearances = [row["appearance"] for row in rows if row.get("appearance")]
  return {
    "viewpoint_margin_pct": summarise([row.get("margin_pct") for row in rows]),
    "viewpoint_margin_deg": summarise([row.get("margin_deg") for row in rows]),
    "hidden_views": summarise([row.get("hidden_views") for row in rows]),
    # a holdout that survived the whole ladder has a margin bounded by the ladder, not
    # by the model: the real one is somewhere above what is reported
    "capped_by_ladder": len(capped),
    "appearance": {kind: summarise_levels([margin[kind]["rung"] for margin in appearances], ladder)
                   for kind, ladder in APPEARANCE_LADDERS.items()} if appearances else None,
    "accuracy": {
      "position_error_pct": summarise([row.get("position_error_pct") for row in solved], more_is_better=False),
      "rotation_error_deg": summarise([row.get("rotation_error_deg") for row in solved], more_is_better=False),
      "num_inliers": summarise([row.get("num_inliers") for row in solved]),
    },
    "holdouts": len(rows),
    "failed": [row["name"] for row in rows if not row["success"]],
    "dense_fallback": "on" if use_dense else "off",
    "scene_radius": round(radius, 5),
    "pass_tolerance": {"position_pct": PASS_POSITION_PCT, "rotation_deg": PASS_ROTATION_DEG},
    "localisations": localisations,
    "seconds": round(time.time() - start, 2),
    "images": rows,
  }
