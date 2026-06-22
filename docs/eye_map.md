# Retinotopic eye map (Tier A)

Replaces the crude "split the photoreceptor list in half + inject optic-flow
scalars" front-end with a **real, retinotopic** mapping from the camera to the
fly's photoreceptors.

## What it does

1. **Real eye geometry.** `eye_map/receptor_directions_buchner71.csv` gives the
   measured viewing direction of every ommatidium (1,398 total, 699/eye) from
   Buchner (1971) via `strawlab/drosophila_eye_map` (BSD). Frame: +X frontal,
   +Y left, +Z dorsal. See `eye_map/NOTICE.md`.
2. **Camera → ommatidia.** `flybrain_eye_map.EyeMap` pinhole-projects each
   ommatidial direction into the Tello image (≈66°×50° FOV), samples local
   luminance with a ~5° acceptance window, and returns per-ommatidium intensity.
   Ommatidia outside the frontal cone read 0 (that part of the eye sees nothing).
3. **Luminance → R1-6.** Each R1-6 cell is driven by the luminance of its
   ommatidium. **Motion (T4/T5) is then computed downstream by the connectome
   itself** — the biologically correct direction of information flow, vs. the
   old code which injected pre-computed optic flow straight into photoreceptors.

## How to run

```bash
# self-test (no drone, no connectome):
python flybrain_eye_map.py

# enable in the real-brain deployment (EXPERIMENTAL — bench-test props off):
FLYBRAIN_RETINOTOPIC=1 python flybrain_tello_real_brain.py
```

Default is off; the optic-flow path is unchanged when the flag is unset.

## Honest scope

- The eye **geometry** is real measured data.
- A single forward Tello camera only covers a **frontal ~66°×50° cone** (≈186 of
  1,398 ommatidia, azimuth ±31°). The lateral/rear eye is unstimulated — a
  hardware limit, not a modelling choice. Full-eye coverage needs a fisheye or
  multi-camera rig the Tello can't carry.
- Which **physical FlyWire R1-6 cell** maps to which ommatidium is assigned in
  retinotopic order, **not** from the connectome's column identity. A faithful
  cell↔column join needs FlyWire Codex visual-column data (Mi1 ↔ ommatidium hex
  coords), which is not in the local files. So the geometry is real; the
  cell↔ommatidium identity is modelled.

## Regenerating the data

`eye_map/build_buchner71_eyemap.py` reproduces the CSV from the upstream source
(NumPy only). You don't need to run it — the CSV is committed.
