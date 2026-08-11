"""Capture server: an Android phone becomes the camera of the pipeline.

The phone opens this server over the local network, walks through a guided capture and
the frames land straight in `storage/datasets/<name>/train/`, where `pipeline.py` picks
them up with nothing else to do. Poses and intrinsics reported by ARCore travel with
them into `capture.json`, next to the photos.

**Why it has to speak HTTPS.** The guidance is built on WebXR, which Chrome on Android
serves out of ARCore, and both WebXR and the camera are powerful features: the browser
hands them out to secure contexts only. `http://192.168.x.y` is not one, so the server
generates a self signed certificate for the address it is answering on and serves TLS.
The phone will warn that the certificate is unknown, which it is — accept it once and
the origin is secure enough for the browser to hand over the camera. `--http` turns TLS
off for the two cases where it only gets in the way: a desktop browser on localhost, and
`adb reverse tcp:8443 tcp:8443`, which makes the phone see the server as localhost and
therefore as trusted with no certificate at all.

Deliberately free of pycolmap and torch: this process only writes files, and staying out
of their way keeps it starting instantly and running next to a pipeline that is busy.
"""

import io
import re
import json
import uuid
import socket
import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import segno
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STORAGE = Path(__file__).parent / "storage"
DATASETS = STORAGE / "datasets"
CERTS = STORAGE / "certs"
CAPTURE_UI = Path(__file__).parent / "capture"

# A phone camera frame is a few hundred KB; anything past this is not one of ours.
MAX_FRAME_BYTES = 12 * 1024 * 1024
# Frames are named so that sorting them is replaying the capture: pipeline.py decimates
# and picks holdouts on the sorted names, and a capture is meaningful in order.
FRAME_NAME = "capture_{:05d}.jpg"

# WebXR hands out poses in its own frame, and nothing downstream reads them back blindly:
# this line is what a later conversion has to be written against.
CONVENTION = ("WebXR world-from-camera in the session reference space: right handed, "
              "+Y up, the camera looks down -Z. COLMAP is the other way round on both.")

app = FastAPI()
sessions = {}

##############################################################################
# NETWORK AND CERTIFICATE
##############################################################################

def lan_address():
    """The address of this machine on the local network, as the phone will see it.

    Opening a UDP socket towards a public address picks the interface the routing table
    would use, without sending a packet. `gethostname` resolution is not equivalent, on
    macOS it usually answers 127.0.0.1.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def ensure_certificate(host):
    """A self signed certificate covering `host`, reused until the address changes.

    The address goes in as a subject alternative name: browsers stopped reading the
    common name years ago, and without a matching SAN Chrome refuses the origin outright
    instead of offering the warning that can be clicked through.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERTS.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = CERTS / "capture.crt", CERTS / "capture.key"
    if cert_path.exists() and key_path.exists():
        existing = x509.load_pem_x509_certificate(cert_path.read_bytes())
        names = existing.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        covered = {str(entry) for entry in names.get_values_for_type(x509.IPAddress)}
        covered |= set(names.get_values_for_type(x509.DNSName))
        fresh = existing.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc)
        if host in covered and fresh:
            return cert_path, key_path

    import ipaddress
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "play-colmap capture")])
    alternatives = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        alternatives.append(x509.IPAddress(ipaddress.ip_address(host)))
    except ValueError:
        alternatives.append(x509.DNSName(host))

    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    key_path.chmod(0o600)
    return cert_path, key_path

##############################################################################
# DATASETS
##############################################################################

def safe_dataset(name):
    """A dataset folder that cannot escape storage/datasets, or a 400."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", (name or "").strip()).strip(".-_")
    if not cleaned:
        raise HTTPException(400, "The dataset name needs a letter or a digit in it")
    return cleaned[:64]


def train_photos(dataset):
    """The photos already sitting in a dataset's train folder."""
    train = DATASETS / dataset / "train"
    if not train.exists():
        return []
    return sorted(path for path in train.iterdir()
                  if path.suffix.lower() in (".jpg", ".jpeg", ".png"))


def sharpness(image):
    """Variance of the Laplacian, the same blur measure `pipeline.py` uses on frames."""
    small = cv2.resize(image, (320, round(320 * image.shape[0] / image.shape[1])))
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


def next_index(dataset_path):
    """The number the next frame gets, past everything already in the folder.

    A dataset can be captured in more than one pass — a room in two halves, a second
    lap for the side that came out thin — and numbering from zero each time overwrites
    the first pass frame by frame.
    """
    highest = -1
    for path in (dataset_path / "train").glob("capture_*.jpg"):
        try:
            highest = max(highest, int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return highest + 1


def load_manifest(dataset_path):
    """What earlier captures left behind, so this one is added rather than substituted."""
    path = dataset_path / "capture.json"
    if not path.exists():
        return {"dataset": dataset_path.name, "convention": CONVENTION, "sessions": []}
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        # a manifest truncated by a crash is not worth losing a capture over
        return {"dataset": dataset_path.name, "convention": CONVENTION, "sessions": []}
    manifest.setdefault("sessions", [])
    return manifest


def write_manifest(session):
    """Rewrite capture.json, so a session that dies mid capture still leaves its poses.

    Read, replace this session's entry, write: two phones can capture into one dataset
    at once, and each has to find the other's frames still there afterwards. Written
    through a temporary file in the same folder, since a phone that drops off the
    network mid write would otherwise leave a truncated manifest behind.
    """
    manifest = load_manifest(session["path"])
    manifest["dataset"] = session["dataset"]
    manifest["convention"] = CONVENTION
    record = {key: value for key, value in session.items() if key not in ("path", "convention")}
    manifest["sessions"] = [entry for entry in manifest["sessions"]
                            if entry.get("id") != session["id"]] + [record]

    path = session["path"] / "capture.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)

##############################################################################
# API
##############################################################################

@app.get("/api/capture/datasets")
def list_datasets():
    """Existing datasets and how many photos they already hold, to warn about mixing."""
    if not DATASETS.exists():
        return {"datasets": []}
    datasets = []
    for path in sorted(DATASETS.iterdir()):
        if not path.is_dir():
            continue
        datasets.append({
            "name": path.name,
            "photos": len(train_photos(path.name)),
            "reconstructed": (path / "sfm" / "0").exists(),
        })
    return {"datasets": datasets}


@app.post("/api/capture/sessions")
async def open_session(body: dict):
    """Open a capture and return where its frames will land.

    Frames are appended, never replacing what is already in `train/`: this endpoint is
    reachable by anything on the local network and a capture that silently wiped a
    dataset would be a bad way to find that out. The UI warns instead, and clearing a
    dataset stays a deliberate act on the machine that owns it.
    """
    dataset = safe_dataset(body.get("dataset"))
    mode = body.get("mode") if body.get("mode") in ("orbit", "walk") else "orbit"

    path = DATASETS / dataset / "train"
    path.mkdir(parents=True, exist_ok=True)
    existing = train_photos(dataset)

    session = {
        # timestamp for reading, random tail for uniqueness: two phones opening a capture
        # in the same second would otherwise share an id, and the second would inherit
        # the first one's frames and overwrite its entry in the manifest
        "id": f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "dataset": dataset,
        "mode": mode,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "device": body.get("device") or {},
        "tracking": body.get("tracking") or "unknown",
        "frames": [],
        "path": DATASETS / dataset,
    }
    sessions[session["id"]] = session
    write_manifest(session)

    return {
        "session": session["id"],
        "dataset": dataset,
        "mode": mode,
        "existing_photos": len(existing),
        "path": str(path),
    }


@app.post("/api/capture/sessions/{session_id}/frames")
async def upload_frame(session_id: str, frame: UploadFile = File(...), meta: str = Form("{}")):
    """Store one captured frame with the pose the phone had when it took it."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown capture session, it may have died with the server")
    if session["finished"]:
        raise HTTPException(409, "This capture is already finished")

    payload = await frame.read()
    if not payload:
        raise HTTPException(400, "Empty frame")
    if len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(413, f"Frame of {len(payload)} bytes is past the {MAX_FRAME_BYTES} limit")

    try:
        decoded = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))[:, :, ::-1]
    except Exception as error:
        raise HTTPException(400, f"Not a readable image: {error}")

    # numbered against the folder, not against this session: two phones can be filling
    # the same dataset, and a second pass must not land on the first pass's frames
    index = next_index(session["path"])
    while (session["path"] / "train" / FRAME_NAME.format(index)).exists():
        index += 1
    name = FRAME_NAME.format(index)
    (session["path"] / "train" / name).write_bytes(payload)

    try:
        metadata = json.loads(meta)
    except json.JSONDecodeError:
        metadata = {}
    record = {
        "index": index,
        "file": name,
        "width": decoded.shape[1],
        "height": decoded.shape[0],
        "bytes": len(payload),
        # measured here as well as on the phone: the client score decides whether a frame
        # is worth uploading, this one is the number the dataset is judged on later
        "sharpness": round(sharpness(decoded), 1),
        "captured": datetime.datetime.now().isoformat(timespec="milliseconds"),
    }
    record.update({key: value for key, value in metadata.items() if key not in record})
    session["frames"].append(record)
    write_manifest(session)

    return {"index": index, "total": len(session["frames"]), "sharpness": record["sharpness"]}


@app.delete("/api/capture/sessions/{session_id}/frames/last")
def drop_last_frame(session_id: str):
    """Undo the last frame, for the shot taken of the inside of a pocket."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown capture session")
    if not session["frames"]:
        raise HTTPException(409, "Nothing captured yet")

    dropped = session["frames"].pop()
    (session["path"] / "train" / dropped["file"]).unlink(missing_ok=True)
    write_manifest(session)
    return {"dropped": dropped["file"], "total": len(session["frames"])}


@app.post("/api/capture/sessions/{session_id}/finish")
def finish_session(session_id: str):
    """Close the capture and hand back the command that turns it into a reconstruction."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown capture session")

    session["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_manifest(session)

    frames = session["frames"]
    sharp = [frame["sharpness"] for frame in frames]
    return {
        "dataset": session["dataset"],
        "frames": len(frames),
        "sharpness": {"median": round(float(np.median(sharp)), 1) if sharp else None,
                      "worst": round(float(np.min(sharp)), 1) if sharp else None},
        "manifest": str(session["path"] / "capture.json"),
        "command": f"python pipeline.py --dataset storage/datasets/{session['dataset']} --reset",
    }


@app.get("/api/capture/sessions/{session_id}")
def get_session(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown capture session")
    return {key: value for key, value in session.items() if key not in ("path", "frames")} | {
        "frames": len(session["frames"]),
    }


@app.get("/api/capture/sessions/{session_id}/frames/{index}.jpg")
def get_frame(session_id: str, index: int):
    """A captured frame, so the phone can show the last shot back."""
    session = sessions.get(session_id)
    if session is None or index >= len(session["frames"]):
        raise HTTPException(404, "Unknown frame")
    return FileResponse(session["path"] / "train" / session["frames"][index]["file"],
                        media_type="image/jpeg")


@app.exception_handler(404)
async def not_found(request, error):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": getattr(error, "detail", "Not found")}, status_code=404)
    return FileResponse(CAPTURE_UI / "index.html")


if CAPTURE_UI.exists():
    app.mount("/", StaticFiles(directory=str(CAPTURE_UI), html=True), name="capture")

##############################################################################
# MAIN
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Capture datasets from a phone over the local network")
    parser.add_argument("--port", type=int, default=8443, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind")
    parser.add_argument("--http", action="store_true", help="Serve plain HTTP: only useful on localhost or through `adb reverse`, browsers refuse the camera anywhere else")
    return parser.parse_args()


def announce(url, port, secure):
    """Print the address and a QR of it, scanning beats typing an IP on a phone."""
    print()
    segno.make(url, error="m").terminal(compact=True)
    print(f"\n  Capture UI: {url}")
    if secure:
        print("  The certificate is self signed, so Chrome will warn once: Advanced -> Proceed.")
        print(f"  No warning at all over USB: adb reverse tcp:{port} tcp:{port}, "
              f"then open https://localhost:{port}")
    else:
        print("  Plain HTTP: the browser will refuse the camera unless this is localhost")
        print(f"  (over USB that is what `adb reverse tcp:{port} tcp:{port}` gives you).")
    # uvicorn logs to stderr, which is unbuffered, so without this the QR turns up after
    # the server has already printed that it is listening — or not at all until it exits
    print(flush=True)


if __name__ == "__main__":
    args = parse_args()
    host = lan_address()
    scheme = "http" if args.http else "https"
    announce(f"{scheme}://{host}:{args.port}/", args.port, not args.http)

    ssl = {} if args.http else dict(zip(("ssl_certfile", "ssl_keyfile"),
                                        (str(path) for path in ensure_certificate(host))))
    uvicorn.run(app, host=args.host, port=args.port, **ssl)
