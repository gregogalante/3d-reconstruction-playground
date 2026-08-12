# AGENTS.md

Agent-facing notes for this repo. Project overview and run commands: see [README.md](README.md).

## Purpose

COLMAP playground that runs end to end without CUDA: a phone captures a dataset, the
pipeline turns it into a sparse then dense reconstruction (and optionally a gaussian
splat), single photos are localised against the result, and a web viewer shows all of
it. Everything a GPU would normally do — dense stereo, splat rasterising — has a CPU
implementation in `libs/`.

## Environment

- Python via Conda (`base` env). No system Python.
- `pycolmap` 3.13 (no CUDA on macOS), `opencv-python`, `numpy`, `scipy`, `open3d`,
  `fastapi`/`uvicorn`, plus `segno` and `cryptography` for the capture server's QR and
  its self signed certificate.
- Node via nvm, for two things only: building the viewer (`cd viewer && yarn build`) and
  running the capture guidance tests (`node --test 'capture/lib/*.test.js'`). Nothing in
  `capture/` is bundled — it is served as it is on disk.
- Do **not** import `torch` together with `pycolmap` in the same process: both link
  their own `libomp` and the process aborts (`OMP: Error #15`). `pipeline.py`
  (pycolmap) therefore launches `libs/splat_trainer.py` (torch) as a subprocess
  instead of importing it, and the trainer reads the COLMAP model through
  `libs/read_write_model.py`.

## Commands

```bash
python pipeline.py --dataset storage/datasets/<name> --reset   # images, sfm, dense, relocation check
python relocation.py --dataset storage/datasets/<name> --image query.jpg --output storage/relocations/<name>
python viewer_server.py     # the viewer, http://localhost:8000
python capture_server.py    # capture from a phone, https://<lan-ip>:8443
node --test 'capture/lib/*.test.js'   # the capture guidance rules
cd viewer && yarn build     # the viewer is served from viewer/dist
```

**There are no datasets in the repo.** Everything under `storage/` is git-ignored bar the
`.keep` files, so a fresh clone has nothing to run on and the first move is always to
make a dataset — from a phone with `capture_server.py`, or by dropping photos or a clip
into `storage/datasets/<name>/train/`. Every dataset named in these notes (`banana`,
`south-building`, `test1`, `test2`, `test4`, `home`, `over-office-1`, `over-office-2`)
is a local capture that measurements were taken on, not something you can obtain by
cloning.

## Layout

Two servers, each with the page it serves, named the same way both times:
`<name>_server.py` serves `<name>/`. `capture_server.py` + `capture/` make datasets,
`viewer_server.py` + `viewer/` show them. Everything they share sits in `libs/`, and
`pipeline.py` is what runs between the two.

- `pipeline.py` — SfM + dense pipeline, one function per step, all steps skip work
  that already exists on disk (`--reset` to rebuild). Two constants at the top:
  `IMAGE_MAX_DIMENSION` and `IMAGE_MAX_ITEMS`, the cap on how many photos of `train/`
  reach `images/`. Over the cap the capture is decimated uniformly rather than cut
  short (400 photos capped at 300 keep three and skip one), so the scene stays
  covered; 0 disables it. Both land in `config.json`, and changing either needs
  `--reset` since `build_images` skips a non empty `images/`. A `train/` holding no
  photos but a `.mov`/`.mp4`/`.m4v` is split into frames instead, see
  [docs/reconstruction.md](docs/reconstruction.md).
- `libs/cpu_mvs.py` — CPU multi-view stereo ([docs/reconstruction.md](docs/reconstruction.md)).
- `libs/cpu_splatting.py` — CPU gaussian splatting: rasteriser, training, PLY export.
- `libs/splat_trainer.py` — runnable trainer (`python -m libs.splat_trainer`), launched by
  the pipeline's splat step, which is currently commented out of the step list
  ([docs/splatting.md](docs/splatting.md)).
- `libs/console.py` — the coloured `print_*` helpers shared with the subprocess.
- `libs/ply.py` — numpy PLY vertex I/O, torch free so `viewer_server.py` can use it too.
- `libs/read_write_model.py` — COLMAP model text/binary I/O (upstream script).
- `libs/colmap2nerf.py` — COLMAP model to `transforms.json` (upstream instant-ngp
  script, kept as is). Run as `python -m libs.colmap2nerf`: it has no main guard, so
  importing it would execute it.
- `capture_server.py` + `capture/` — where a dataset comes from: the phone capture server
  and its page ([docs/capture.md](docs/capture.md)). Free of pycolmap and torch on
  purpose, it only writes files
  and stays startable while a pipeline is busy. `capture/lib/` is plain ES modules,
  JavaScript Standard Style, and the guidance in it is the one part of the repo with
  tests (`node --test 'capture/lib/*.test.js'`).
- `relocation.py` — localise a query image against an existing reconstruction
  ([docs/relocalisation.md](docs/relocalisation.md)).
- `libs/localizer.py` — the retrieval, matching and PnP behind it.
- `libs/evaluation.py` — leave one out relocalisation over a dataset's own images, run
  as the last pipeline step and written to `relocation.json`
  ([docs/relocalisation.md](docs/relocalisation.md)).
- `viewer_server.py` + `viewer/` — where a dataset ends up: FastAPI backend and the React
  viewer it serves out of `viewer/dist`. `/api/datasets/<name>/clouds`
  reports which of the sparse, dense and splat clouds exist, the UI lets you switch
  between them and greys out the missing ones. The gaussians are served as
  `/splat.splat` (the 32 byte per gaussian layout the web renderer reads), converted
  on demand from the PLY and cached next to it. The conversion pre turns every
  gaussian by half a turn around x, because drei's `Splat` displays the rows turned
  that way (it negates the center z on read, the center y on upload, and decodes the
  quaternion to match). Without it the gaussians sit rotated against the point clouds
  and the camera frustums, which are drawn straight from the COLMAP frame. Delete the
  cached `.splat` files after touching the conversion, the cache only tracks the PLY
  timestamp.
- `storage/datasets/<name>/` — `train/` (input photos or a clip), `images/` (resized),
  `database.db`, `sfm/`, `dense/`, `splat/`, `config.json` (the constants and CLI options
  of the run), `time.json` (seconds per step, rewritten every run: a step whose output was
  already on disk reads as ~0), `relocation.json` (the leave one out check), and
  `capture.json` when the dataset came from a phone. All git-ignored, including the
  datasets themselves.

## Conventions

- **Indentation is not uniform, and that is on purpose per file, not per repo.**
  `pipeline.py`, `relocation.py` and everything in `libs/` use two spaces; the two
  servers use four. Match the file you are in rather than the repo.
- **JavaScript follows Standard Style** — no semicolons, single quotes, two spaces,
  `window.` on browser globals. That is `capture/`; `viewer/` is an older React app that
  predates the rule and still has semicolons, so again: match the file.
- **Comments say why, not what.** A comment that restates the line above it is noise; a
  comment carrying the measurement or the failure that produced a threshold is the reason
  the threshold survives a rewrite. There are a lot of numbers in these notes for the
  same reason.
- **Defaults are measured, and the measurement is recorded** — including the controls
  that show a change did no harm elsewhere, and the attempts that failed. `docs/` is
  mostly that.
- Steps skip work already on disk. Anything expensive checks for its own output first,
  which is what makes `pipeline.py` cheap to re-run and `--reset` the way to force it.

## Deeper notes

The parts with enough behind them to argue about live in `docs/`, one domain each:

- [docs/capture.md](docs/capture.md) — the phone capture server and its page: why it
  insists on HTTPS, what the guidance enforces, what a real capture measured.
- [docs/reconstruction.md](docs/reconstruction.md) — video input, the CPU dense stereo,
  and the mapper thresholds that decide whether a model stays whole.
- [docs/relocalisation.md](docs/relocalisation.md) — retrieval, matching and PnP, plus
  the leave one out check that runs as the last pipeline step.
- [docs/splatting.md](docs/splatting.md) — the CPU gaussian splatting trainer.

Each one records what was measured as well as what was decided, including the things
that did not work, so a default can be checked rather than trusted.
