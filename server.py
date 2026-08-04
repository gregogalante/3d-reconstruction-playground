import math
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pycolmap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from libs.ply import read_ply, read_header

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE = Path(__file__).parent / "storage"
DATASETS = STORAGE / "datasets"
RELOCATIONS = STORAGE / "relocations"

# the three point representations a dataset can hold, in pipeline order
CLOUDS = {
    "sparse": Path("sfm") / "reconstruction.ply",
    "dense": Path("dense") / "fused.ply",
    "splat": Path("splat") / "point_cloud.ply",
}
SH_DC = 0.28209479177387814  # value of the degree 0 spherical harmonic


@lru_cache(maxsize=16)
def load_reconstruction(name: str):
    sfm_path = DATASETS / name / "sfm" / "0"
    if not sfm_path.exists():
        raise HTTPException(404, f"SfM data not found for {name}")
    return pycolmap.Reconstruction(str(sfm_path))


def c2w_from_image(img):
    cfw = img.cam_from_world()
    mat34 = cfw.matrix()  # 3x4
    mat44 = np.vstack([mat34, [0, 0, 0, 1]])
    return np.linalg.inv(mat44).tolist()


def c2w_from_relocation(rotation_matrix, camera_center):
    R = np.array(rotation_matrix)  # world-to-cam
    C = np.array(camera_center)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = C
    return c2w.tolist()


@app.get("/api/datasets")
def list_datasets():
    names = sorted(d.name for d in DATASETS.iterdir() if d.is_dir())
    return {"datasets": names}


@app.get("/api/datasets/{name}/cameras")
def get_cameras(name: str):
    recon = load_reconstruction(name)
    cameras = []
    for img in recon.images.values():
        if not img.has_pose:
            continue
        cam = recon.cameras[img.camera_id]
        fl = cam.focal_length
        fovx = 2 * math.atan(cam.width / (2 * fl)) * 180 / math.pi
        fovy = 2 * math.atan(cam.height / (2 * fl)) * 180 / math.pi
        cameras.append({
            "image_name": img.name,
            "camera_to_world": c2w_from_image(img),
            "width": cam.width,
            "height": cam.height,
            "fovx": fovx,
            "fovy": fovy,
        })
    cameras.sort(key=lambda c: c["image_name"])
    return {"cameras": cameras}


@app.get("/api/datasets/{name}/images/{filename}")
def get_image(name: str, filename: str):
    path = DATASETS / name / "images" / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/datasets/{name}/clouds")
def list_clouds(name: str):
    """Which of the sparse, dense and splat clouds this dataset has on disk."""
    clouds = {}
    for kind, relative in CLOUDS.items():
        path = DATASETS / name / relative
        if not path.exists():
            clouds[kind] = {"available": False}
            continue
        _, count, _ = read_header(path)
        clouds[kind] = {
            "available": True,
            "points": count,
            "bytes": path.stat().st_size,
        }
    return {"clouds": clouds}


@app.get("/api/datasets/{name}/clouds/{kind}.ply")
def get_cloud(name: str, kind: str):
    if kind not in CLOUDS:
        raise HTTPException(404, f"Unknown cloud {kind}")
    path = DATASETS / name / CLOUDS[kind]
    if not path.exists():
        raise HTTPException(404, f"{kind} cloud not found for {name}")
    return FileResponse(path, media_type="application/octet-stream")


@app.get("/api/datasets/{name}/splat.splat")
def get_splat(name: str):
    """The trained gaussians in the .splat layout the web renderer reads."""
    ply_path = DATASETS / name / CLOUDS["splat"]
    if not ply_path.exists():
        raise HTTPException(404, f"Splat not found for {name}")

    splat_path = ply_path.with_suffix(".splat")
    if not splat_path.exists() or splat_path.stat().st_mtime < ply_path.stat().st_mtime:
        splat_path.write_bytes(ply_to_splat(ply_path))
    return FileResponse(splat_path, media_type="application/octet-stream")


def ply_to_splat(path: Path) -> bytes:
    """Convert a 3DGS PLY to the .splat rows the viewer expects.

    One 32 byte row per gaussian: position (3 float32), scale (3 float32),
    colour RGBA (4 uint8), rotation quaternion wxyz (4 uint8, mapped to 0..255).

    The renderer displays the rows turned by half a turn around x, a convention
    inherited from the antimatter15 viewer: it negates z when reading a center, then
    negates y again when uploading it, and decodes the quaternion to match. Nobody
    notices while looking at gaussians alone, but here they share the scene with the
    point clouds and the camera frustums, which come straight from the COLMAP frame.

    So the rows are written pre turned by the same half turn, which cancels out:
    position `(x, y, z)` -> `(x, -y, -z)`, rotation `(w, x, y, z)` -> `(w, x, -y, -z)`.
    """
    data = read_ply(path)

    def column(name):
        return np.asarray(data[name], dtype=np.float32)

    rows = np.zeros(len(data), dtype=[("position", "f4", 3), ("scale", "f4", 3), ("color", "u1", 4), ("rotation", "u1", 4)])
    rows["position"] = np.stack([column("x"), -column("y"), -column("z")], axis=1)
    rows["scale"] = np.exp(np.stack([column(f"scale_{i}") for i in range(3)], axis=1))

    colors = 0.5 + SH_DC * np.stack([column(f"f_dc_{i}") for i in range(3)], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-column("opacity")))
    rows["color"] = np.clip(np.concatenate([colors, opacity[:, None]], axis=1) * 255.0, 0, 255)

    quaternions = np.stack([column("rot_0"), column("rot_1"), -column("rot_2"), -column("rot_3")], axis=1)
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    rows["rotation"] = np.clip(quaternions * 128.0 + 128.0, 0, 255)
    return rows.tobytes()


@app.get("/api/relocations")
def list_relocations():
    relocations = []
    if not RELOCATIONS.exists():
        return {"relocations": []}
    for d in sorted(RELOCATIONS.iterdir()):
        if not d.is_dir():
            continue
        for jf in sorted(d.glob("*.json")):
            data = json.loads(jf.read_text())
            dataset_path = data.get("dataset", "")
            dataset_name = Path(dataset_path).name if dataset_path else ""
            entry = {
                "name": jf.stem,
                "folder": d.name,
                "dataset_name": dataset_name,
                "success": data.get("success", False),
                "num_inliers": data.get("num_inliers", 0),
                "num_correspondences": data.get("num_correspondences", 0),
                "inlier_ratio": data.get("inlier_ratio"),
                "reprojection_error": data.get("reprojection_error"),
                "camera_center": data.get("camera_center", [0, 0, 0]),
                "fovx": data.get("fovx", 50),
                "fovy": data.get("fovy", 38),
                "has_overlay": (d / f"{jf.stem}_overlay.jpg").exists(),
            }
            if data.get("rotation_matrix") and data.get("camera_center"):
                entry["camera_to_world"] = c2w_from_relocation(
                    data["rotation_matrix"], data["camera_center"]
                )
            relocations.append(entry)
    return {"relocations": relocations}


@app.get("/api/relocations/{folder}/{name}/image")
def get_relocation_image(folder: str, name: str):
    path = RELOCATIONS / folder / f"{name}.jpg"
    if not path.exists():
        raise HTTPException(404, "Relocation image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/relocations/{folder}/{name}/overlay")
def get_relocation_overlay(folder: str, name: str):
    """The verification image relocation.py draws: matches, and the model reprojected."""
    path = RELOCATIONS / folder / f"{name}_overlay.jpg"
    if not path.exists():
        raise HTTPException(404, "Relocation overlay not found")
    return FileResponse(path, media_type="image/jpeg")


UI_DIST = Path(__file__).parent / "ui" / "dist"
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
