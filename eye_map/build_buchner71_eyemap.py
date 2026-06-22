#!/usr/bin/env python3
"""
Regenerate receptor_directions_buchner71.csv (the vendored eye map).

This reproduces the direction computation from the Straw lab's
`strawlab/drosophila_eye_map` (BSD): the hand-digitized stereographic points of
Buchner's (1971) eye map are converted stereographic -> long/lat -> rotated head
frame -> 3D unit vectors, using only NumPy. See NOTICE.md.

Usage (only needed to regenerate the CSV from upstream):
    curl -sLO https://raw.githubusercontent.com/strawlab/drosophila_eye_map/master/drosophila_eye_map/precompute_buchner71_optics.py
    python build_buchner71_eyemap.py

The committed receptor_directions_buchner71.csv is the artifact; you do NOT need
to run this to use the eye map.

Head frame: +X frontal, +Y left, +Z dorsal.
"""
import csv
import numpy as np

SRC = 'precompute_buchner71_optics.py'   # upstream file (not vendored)


def load_digitized_points(path=SRC):
    src = open(path).read().splitlines()

    def idx(prefix):
        for i, l in enumerate(src):
            if l.startswith(prefix):
                return i
        raise ValueError(prefix)
    xi, yi, gi = idx('x= array('), idx('y= array('), idx('def get_rot_mat')
    ns = {'array': np.array, 'numpy': np}
    exec("\n".join(src[xi:yi]), ns)
    exec("\n".join(src[yi:gi]), ns)
    return np.asarray(ns['x']), np.asarray(ns['y'])


def get_rot_mat(theta, ax, ay, az):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c + (1 - c) * ax * ax, (1 - c) * ax * ay + s * az, (1 - c) * ax * az - s * ay],
                     [(1 - c) * ay * ax - s * az, c + (1 - c) * ay * ay, (1 - c) * ay * az + s * ax],
                     [(1 - c) * az * ax + s * ay, (1 - c) * az * ay - s * ax, c + (1 - c) * az * az]])


def long_lat2xyz(lon, lat, R=1.0):
    colat = np.pi / 2 - lat
    return R * np.sin(colat) * np.cos(lon), R * np.sin(colat) * np.sin(lon), R * np.cos(colat)


def xyz2long_lat(xn, yn, zn):
    lon = np.arctan2(yn, xn)
    colat = np.arctan2(np.sqrt(xn ** 2 + yn ** 2), zn)
    return lon, np.pi / 2 - colat, np.sqrt(xn ** 2 + yn ** 2 + zn ** 2)


def rotate(M, lon, lat, R):
    x3, y3, z3 = long_lat2xyz(lon, lat, R)
    xn = M[0, 0] * x3 + M[0, 1] * y3 + M[0, 2] * z3
    yn = M[1, 0] * x3 + M[1, 1] * y3 + M[1, 2] * z3
    zn = M[2, 0] * x3 + M[2, 1] * y3 + M[2, 2] * z3
    return xyz2long_lat(xn, yn, zn)


def stereo2ll(x, y, R=1.0):
    rho = np.hypot(x, y)
    return np.arctan2(y, x), np.pi / 2 - 2 * np.arctan(rho / (2 * R)), R


def main():
    x, y = load_digitized_points()
    Mf = get_rot_mat(-np.pi / 2, 1, 0, 0)
    sc = np.eye(3); sc[2, 2] = -1
    Mf = Mf.dot(sc)
    Mr = np.linalg.inv(Mf)
    hlon, hlat, hR = stereo2ll(x, y)
    lon, lat, R = rotate(Mr, hlon, hlat, hR)
    lx, ly, lz = long_lat2xyz(lon, lat, 1.0)
    left = np.stack([lx, ly, lz], axis=1)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right = left * np.array([1, -1, 1])      # mirror Y -> right eye
    dirs = np.vstack([left, right])
    eye = ['left'] * len(left) + ['right'] * len(right)

    with open('receptor_directions_buchner71.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dx', 'dy', 'dz', 'eye'])
        for v, e in zip(dirs, eye):
            w.writerow([f'{v[0]:.6f}', f'{v[1]:.6f}', f'{v[2]:.6f}', e])
    print(f"wrote receptor_directions_buchner71.csv ({len(dirs)} ommatidia)")


if __name__ == '__main__':
    main()
