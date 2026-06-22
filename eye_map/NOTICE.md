# Eye-map data — attribution & license

`receptor_directions_buchner71.csv` contains the viewing direction of each
*Drosophila melanogaster* ommatidium as a 3D unit vector.

- **Columns:** `dx,dy,dz,eye` — unit vector in the fly head frame
  (**+X frontal, +Y left, +Z dorsal**) and the eye it belongs to (`left`/`right`).
- **Count:** 1,398 ommatidia (699 per eye).

## Provenance

The directions are derived from the digitized eye map in:

- **Buchner, E. (1971).** *Dunkelanregung des stationären Flugs der
  Fruchtfliege Drosophila.* Diplom thesis, Tübingen. (Also Buchner 1984, in
  *Photoreception and Vision in Invertebrates*, Ali ed.)

Digitization and the stereographic→spherical reconstruction come from the
**Straw lab** repository
[`strawlab/drosophila_eye_map`](https://github.com/strawlab/drosophila_eye_map)
(BSD license; © California Institute of Technology, author Andrew Straw).

`build_buchner71_eyemap.py` reproduces that repo's
`precompute_buchner71_optics.py` direction computation (stereographic →
long/lat → rotated frame → xyz) using only NumPy, and writes the CSV. It is the
provenance record; the CSV is the vendored artifact.

This vendored data is used here under the upstream BSD license.
