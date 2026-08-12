# Capture

How a dataset gets made: the phone capture server, its page, and the rules the
guidance enforces while you walk.

Back to [AGENTS.md](../AGENTS.md).

## Capturing from a phone

`capture_server.py` plus the static page in `capture/` turn an Android phone into the
camera of the pipeline. The phone opens the page over the local network, walks a guided
capture, and the frames land in `storage/datasets/<name>/train/` with the ARCore poses
beside them in `capture.json`. Nothing downstream changes: `pipeline.py` reads that
`train/` like any other.

```bash
python capture_server.py                 # TLS on 8443, prints a QR of the LAN address
python capture_server.py --http --port 8444   # localhost or adb reverse only
node --test 'capture/lib/*.test.js'      # the guidance rules
```

### Why it insists on HTTPS

The guidance rides on WebXR, which Chrome on Android implements over ARCore, and both
WebXR and the camera are handed out to secure contexts only. `http://192.168.x.y` is not
one. So the server generates a self signed certificate carrying the LAN address as a
subject alternative name — without a matching SAN Chrome refuses the origin instead of
offering the warning you can click through — and serves TLS. Three ways in, in order of
how much they hurt:

1. **USB**: `adb reverse tcp:8443 tcp:8443`, then `https://localhost:8443` on the phone.
   localhost is trusted, so no warning at all.
2. **Wi-Fi**: scan the QR, accept the certificate warning once (Advanced → Proceed).
3. `chrome://flags/#unsafely-treat-insecure-origin-as-secure` with the plain address, if
   the certificate becomes a nuisance.

The key lives in `storage/certs/`, which is git ignored — check that it stays that way.

### What the guidance actually enforces

A capture dies of four things, and each has a rule in `capture/lib/guidance.js`:

- **Gaps.** Two neighbouring shots have to see some of the same thing, so the turn
  between them is capped at a quarter of the field of view — `limitsFor()` reads the lens
  out of the intrinsics rather than assuming one. **The thresholds are not constants,
  and the first real capture is why**: a phone in portrait sees 34° across, where the
  original fixed rule of 30 cm or 25° left a quarter of the frame shared. 35 of 67
  neighbouring pairs turned more than half the lens, 7 shared nothing at all, and COLMAP
  registered 16 frames out of 68. Replaying that same walk through the current rule
  gives 157 frames, a median turn of 8.5°, a maximum of 9°, and no pair under half the
  lens.
- **Steps too long for the subject.** The sideways step is capped at an eighth of the
  distance to what is in front, which the ARCore hit test measures live — a step that
  is fine across a courtyard is a different view entirely across a table.
- **Rotation without translation.** A panorama has no parallax, so nothing triangulates.
  In orbit mode the shutter is gated on the angle *around the subject*, so standing still
  and turning earns nothing however long you wait. In walk mode a turn past the overlap
  limit *does* take a shot, and says what you are doing wrong at the same time: refusing
  it is what tore the first capture apart, since the frames either side of an unrecorded
  swing have nothing in common to join them by.
- **Blur, and its quieter twin.** Every frame is scored by the variance of its Laplacian
  — the same measure `pipeline.py` uses on video — and dropped under 55% of this
  capture's running median. **But the bar falls as the chain waits**: past the overlap
  limit the frame is the only thing holding two halves of the capture together, and a
  blurred frame that joins them beats the hole that replaces it. The second real capture
  is why. Its six broken pairs were not fast swings outrunning the shutter — 2.84 seconds
  passed between frames there against 0.55 elsewhere, at ordinary turning speed, so
  something refused every frame through the turn and the view had moved 34 to 80 degrees
  by the time one was accepted. But a white wall is in perfect focus and equally useless, so
  the server also counts FAST corners on arrival and the phone says so out loud when the
  recent frames are starved. Measured: a stone facade that reconstructs gives 2846
  corners a frame, a fruit on a table 1694, and the white freezer in a white corner that
  failed, 254. The corner count tracks COLMAP's own keypoint count at 0.83 where the
  Laplacian variance, which confuses flat with blurred, manages 0.70.

Two rings of twelve targets at 8° and 28° of elevation, because one ring reconstructs a
band and leaves the top of the object a guess.

### Told without looking

Walking around a subject you are watching the subject, not the phone, so the guidance
leaves by two doors:

- **the coverage dial around the shutter**, two rings of segments filling in as you go.
  It answers *which side have I not done*, which a percentage cannot and a map only can
  if you stop to read it. World fixed, since a dial spinning with the phone is unusable
  while walking. In walk mode, where there is no orbit to cover, the same dial fills
  towards the next frame instead, and the breadcrumb radar sits beside it.
- **the vibration motor**: one buzz per frame, a triple at each quarter of the orbit.
  Silent on purpose — a capture happens in rooms with other people in them. Spoken
  guidance was built and then removed for the same reason.

Two more things the UI refuses to let happen quietly. Finishing a capture under twelve
frames or under 60% coverage is argued with once — *keep capturing* or *finish anyway* —
because the expensive way to discover a thin capture is an hour into a pipeline run.
And a JPEG encode that never comes back cannot wedge the session: `toBlob` races a
timeout, so a stalled encoder costs one frame instead of every frame after it.

### Three ways it can capture

| path | pose | frames | when |
|---|---|---|---|
| WebXR `immersive-ar` + `camera-access` | ARCore, 6DoF | the camera texture, read back through a framebuffer | Chrome on Android with ARCore |
| `getUserMedia` | none | `<video>` drawn to a canvas | anything else, guidance falls back to blur and count |
| simulator (`?simulate=1`) | fabricated orbit | a painted canvas, every 8th blurred | testing without a phone |

They share one `shoot()` so the fallbacks cannot rot unnoticed. The simulator takes a
`stepBy(ms)` on its own clock, which is how a whole capture can be replayed in a hidden
tab where `requestAnimationFrame` never fires; `window.captureDebug` is published in that
mode only.

### What lands on disk

`train/capture_00000.jpg` upwards, named so that sorting them replays the capture, and
`capture.json` as `{dataset, convention, sessions: [...]}`, each session holding per
frame: the WebXR pose (position, orientation, matrix), the intrinsics derived from the
projection matrix *plus the raw matrix*, both sharpness scores (the phone's and the
server's), and the size. Written after every frame through a temporary file, so a phone
that walks out of range still leaves a readable manifest.

**A dataset can be captured in several passes** — a room in two halves, a second lap for
the side that came out thin, or two phones at once — so frames are numbered against the
folder rather than the session, and the manifest is read-modify-written to keep the
earlier sessions. The first version of this did neither: a second capture into the same
dataset restarted at `capture_00000.jpg`, overwriting the first pass frame by frame, and
replaced the manifest with its own frames. Session ids carry a random tail for the same
reason — they were timestamps to the second, and two sessions opened in one second
shared an id, which made the second inherit the first one's frames.

The poses are stored, not used. They are in the WebXR frame — right handed, +Y up, camera
down -Z — and COLMAP is the opposite on both counts; the `convention` field in the
manifest is what a conversion has to be written against. Feeding them to COLMAP as pose
priors, which would let spatial matching replace the quadratic exhaustive one, is the
obvious next thing and is not done.

### Verified, and not

Verified here: certificate generation and the SANs, TLS on the LAN address, the QR at
startup, the whole API (upload, undo, finish, two passes into one dataset keeping both,
a name that tries to escape `storage/datasets`, a body that is not an image), the
guidance rules under `node --test`, a full simulated capture to 100% coverage with blur
rejection firing, the vibration cues, the argument against finishing a thin
capture, the `getUserMedia` path against a canvas backed fake camera, and — the one that
matters — 16 real photos replayed through the HTTP API and reconstructed by the pipeline
into the same models the same photos give when copied by hand.

**Then a real phone ran it**, which settled the parts that could only be guessed at. The
WebXR session, the `camera-access` readback and the ARCore poses all work: an Android
Chrome captured 68 frames at 868×1920 with poses, intrinsics and timestamps, and the
manifest came back whole. What it also produced was a reconstruction in pieces, which is
where the field of view rule above comes from — the guess that hurt was not the API, it
was assuming a wide lens.

Still unverified: a capture that reconstructs. `test1` is the only real one so far and it
failed twice over, on overlap and on texture, so the fixed thresholds have been replayed
against its trajectory but never walked. The diagnosis in `capture.json` is the thing to
read after the next attempt.

### After a capture

`capture.json` carries a `diagnosis` per session: the field of view, the median corner
count, how many neighbouring pairs turned past half the lens, how many shared nothing,
and a verdict in words. On `test1` it reads *"7 pairs of neighbouring frames share no
view at all, the model will break there; the scene is short of texture (254 corners a
frame, against 1700 to 2800 on captures that reconstruct)"* — which is the whole
post mortem, available before the pipeline runs rather than 45 minutes into it.
