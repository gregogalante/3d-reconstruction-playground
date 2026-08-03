# Play Colmap

## Running

```bash
python pipeline.py --dataset storage/datasets/home --reset
python relocation.py --dataset storage/datasets/home --image storage/inputs/relocation_home.jpg --output storage/relocations/home
```

## Dense reconstruction

The pipeline ends with a dense reconstruction that runs entirely on CPU (COLMAP's
`patch_match_stereo` is CUDA-only), producing `storage/datasets/<name>/dense/fused.ply`.
Depth maps come from the plane-sweep MVS in `libs/cpu_mvs.py`, the fusion from pycolmap.

```bash
python pipeline.py --dataset storage/datasets/home --dense-max-size 1024  # denser, ~2.5x slower
```

Roughly 2s per image at the default `--dense-max-size 640` on an M4 (10 cores).
See [AGENTS.md](AGENTS.md) for the details and the remaining tuning flags.

## Util links

- https://github.com/ruili3/awesome-dust3r
- https://github.com/naver/dust3r [https://www.perplexity.ai/search/running-colmap-dense-on-a-macb-l3mZtelGQxmjZizgbo49XA#1]
- https://github.com/cvg/Hierarchical-Localization
- https://github.com/freddewitt/CorbeauSplat?tab=readme-ov-file
- https://github.com/OpsiClear/DepthDensifier