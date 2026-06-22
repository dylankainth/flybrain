#!/usr/bin/env python3
"""
FLYBRAIN RETINOTOPIC EYE MAP
============================
Maps a forward-facing camera frame onto the *Drosophila* compound eye using the
real measured ommatidial viewing directions of Buchner (1971), digitized by the
Straw lab (`strawlab/drosophila_eye_map`, BSD). See `eye_map/NOTICE.md`.

This is the "Tier A" retinotopic front-end: instead of collapsing the camera to
a few global optic-flow scalars, each ommatidium samples the visual direction it
actually looks at, and feeds its photoreceptors (R1-6) the local luminance.
Motion (T4/T5) is then computed *downstream* by the connectome itself, which is
the biologically correct direction of information flow.

Honest scope / caveats
-----------------------
* The eye *geometry* (699 ommatidia/eye, their viewing directions) is real,
  measured data.
* A single forward Tello camera only covers a frontal cone (~66 x 50 deg), so
  only the frontal ommatidia are stimulated; the lateral/rear eye sees nothing.
  That is a hardware limit, not a modelling choice.
* Which physical FlyWire R1-6 cell maps to which ommatidium is assigned in
  retinotopic order, NOT from the connectome's column identity (that join needs
  FlyWire Codex visual-column data, which is not in the local files). So the
  geometry is real; the cell<->ommatidium identity is modelled.

Coordinate frame (from the Buchner data): +X frontal, +Y left, +Z dorsal.
"""

import os
import numpy as np

EYE_MAP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'eye_map', 'receptor_directions_buchner71.csv')

# Tello camera: ~82.6 deg diagonal FOV, 4:3 sensor -> ~66 deg H, ~50 deg V.
TELLO_HFOV_DEG = 66.0
TELLO_VFOV_DEG = 50.0
# Acceptance (half-) angle of an ommatidium, Delta-rho ~ 5 deg in Drosophila.
ACCEPTANCE_DEG = 5.0


class EyeMap:
    """Real Buchner-1971 ommatidial directions + a pinhole camera sampler."""

    def __init__(self, csv_path=EYE_MAP_CSV):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Eye map not found at {csv_path}. Generate it with "
                "eye_map/build_buchner71_eyemap.py.")
        dirs, eye = [], []
        with open(csv_path) as f:
            header = f.readline()
            for line in f:
                dx, dy, dz, e = line.strip().split(',')
                dirs.append((float(dx), float(dy), float(dz)))
                eye.append(e)
        self.dirs = np.asarray(dirs, dtype=np.float64)        # (N, 3) unit vectors
        self.dirs /= np.linalg.norm(self.dirs, axis=1, keepdims=True)
        self.eye = np.asarray(eye)
        self.n = len(self.dirs)
        self.left_mask = self.eye == 'left'
        self.right_mask = self.eye == 'right'
        # azimuth (+left), elevation (+up), degrees
        self.azimuth = np.degrees(np.arctan2(self.dirs[:, 1], self.dirs[:, 0]))
        self.elevation = np.degrees(np.arcsin(np.clip(self.dirs[:, 2], -1, 1)))

    # ------------------------------------------------------------------ #
    def visible_mask(self, hfov_deg=TELLO_HFOV_DEG, vfov_deg=TELLO_VFOV_DEG):
        """Which ommatidia fall inside a forward (+X) camera frustum."""
        dx, dy, dz = self.dirs[:, 0], self.dirs[:, 1], self.dirs[:, 2]
        infront = dx > 1e-6
        u_ang = np.degrees(np.arctan2(-dy, dx))   # horizontal (image right = -Y)
        v_ang = np.degrees(np.arctan2(dz, dx))    # vertical (image up = +Z)
        return infront & (np.abs(u_ang) <= hfov_deg / 2) & (np.abs(v_ang) <= vfov_deg / 2)

    def project_to_pixels(self, frame_shape, hfov_deg=TELLO_HFOV_DEG, vfov_deg=TELLO_VFOV_DEG):
        """Pinhole-project each ommatidial direction to (col, row) pixels.

        Returns (cols, rows, visible_mask). Pixels for non-visible ommatidia are
        clamped and should be ignored via the mask.
        """
        H, W = frame_shape[:2]
        dx = np.clip(self.dirs[:, 0], 1e-6, None)
        fx = (W / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        fy = (H / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
        cols = W / 2.0 + fx * (-self.dirs[:, 1] / dx)   # +X image right = -Y
        rows = H / 2.0 - fy * (self.dirs[:, 2] / dx)    # +up = -row
        vis = self.visible_mask(hfov_deg, vfov_deg)
        cols = np.clip(cols, 0, W - 1)
        rows = np.clip(rows, 0, H - 1)
        return cols, rows, vis

    def sample(self, gray, hfov_deg=TELLO_HFOV_DEG, vfov_deg=TELLO_VFOV_DEG,
               accept_deg=ACCEPTANCE_DEG):
        """Sample a single-channel frame at each ommatidium.

        `gray` is HxW, any numeric range (it is normalised to 0-1 internally).
        Returns intensities in [0, 1], length N; ommatidia outside the camera
        FOV are 0 (that part of the eye sees nothing).
        """
        gray = np.asarray(gray, dtype=np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0
        H, W = gray.shape
        cols, rows, vis = self.project_to_pixels((H, W), hfov_deg, vfov_deg)

        # Acceptance radius in pixels (angular acceptance -> pixels via focal length)
        fx = (W / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        rad = max(1, int(round(np.tan(np.radians(accept_deg)) * fx)))

        out = np.zeros(self.n, dtype=np.float64)
        ci = np.clip(np.round(cols).astype(int), 0, W - 1)
        ri = np.clip(np.round(rows).astype(int), 0, H - 1)
        for k in np.nonzero(vis)[0]:
            r0, r1 = max(0, ri[k] - rad), min(H, ri[k] + rad + 1)
            c0, c1 = max(0, ci[k] - rad), min(W, ci[k] + rad + 1)
            out[k] = gray[r0:r1, c0:c1].mean()
        return out

    # ------------------------------------------------------------------ #
    def assign_photoreceptors(self, r16_left_idx, r16_right_idx, per_ommatidium=6):
        """Map R1-6 neuron indices to ommatidia (retinotopic-order proxy).

        Returns an int array `omma_of_cell` aligned with the concatenation
        [r16_left_idx, r16_right_idx], giving the ommatidium index (into self.n)
        each cell belongs to. Cells beyond the available ommatidia wrap.
        """
        left_omma = np.nonzero(self.left_mask)[0]
        right_omma = np.nonzero(self.right_mask)[0]
        cell_idx = np.concatenate([np.asarray(r16_left_idx), np.asarray(r16_right_idx)])
        omma_of_cell = np.empty(len(cell_idx), dtype=np.int64)
        nl = len(r16_left_idx)
        for j in range(nl):
            omma_of_cell[j] = left_omma[(j // per_ommatidium) % len(left_omma)]
        for j in range(len(r16_right_idx)):
            omma_of_cell[nl + j] = right_omma[(j // per_ommatidium) % len(right_omma)]
        return cell_idx, omma_of_cell


if __name__ == '__main__':
    # Self-test: load the eye map and sample a synthetic frame.
    em = EyeMap()
    print(f"Loaded {em.n} ommatidia "
          f"({em.left_mask.sum()} left, {em.right_mask.sum()} right)")
    vis = em.visible_mask()
    print(f"Visible in Tello FOV ({TELLO_HFOV_DEG}x{TELLO_VFOV_DEG} deg): "
          f"{vis.sum()} ommatidia")
    print(f"  frontal azimuth span of visible: "
          f"{em.azimuth[vis].min():.0f}..{em.azimuth[vis].max():.0f} deg")

    # Synthetic frame: bright on the left half, dark on the right half.
    H, W = 240, 320
    frame = np.zeros((H, W), dtype=np.uint8)
    frame[:, :W // 2] = 255
    inten = em.sample(frame)
    left_vis = vis & em.left_mask
    right_vis = vis & em.right_mask
    # Image left half (cols < W/2) corresponds to +Y = fly's LEFT eye/azimuth.
    print(f"mean intensity, visible LEFT-eye ommatidia:  {inten[left_vis].mean():.2f}")
    print(f"mean intensity, visible RIGHT-eye ommatidia: {inten[right_vis].mean():.2f}")
    assert inten[left_vis].mean() > inten[right_vis].mean(), "retinotopy flipped!"
    print("OK: left-bright frame drives left-eye ommatidia more than right. ")
