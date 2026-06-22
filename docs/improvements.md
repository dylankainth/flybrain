# FlyBrain Tello — Fidelity & Real-Time Improvements

Goal: strengthen the claim *"running a fly brain on my PC and streaming the
controls/camera to/from a drone"* by closing the gaps between what the code
claims and what it actually does.

Applies to `flybrain_tello_real_brain.py` and `flybrain_tello_camera.py`
(both load the real FlyWire *Drosophila* connectome and command a Tello).

## Tier 1 — makes "fly brain" honest (fidelity / correctness)

1. **Use the full connectome.** Both scripts subsampled synapses with
   `chunk.iloc[np.arange(0, len(chunk), 100)]`, keeping only ~1% of the wiring.
   The real file is ~80M synapses (3.7 GB), so it is streamed in chunks keeping
   only the numeric edge arrays (bounded memory). Defaults to the full
   connectome; `FLYBRAIN_SYNAPSE_STRIDE=N` env var is a hardware fallback.

2. **Fix the E/I sign bug — inhibitory neurons currently do nothing.** A GABA
   neuron contributed `+w` in `n_exc` and `-w` in `n_inh`, netting exactly zero.
   Bake the neurotransmitter sign into the synapse weights (Dale's law) so
   inhibition actually inhibits, and use a single signed synaptic matmul.

3. **Ground I/O wiring in real cell types (exact matching).** The old substring
   regex misclassified cells: `R1|R2|...` matched `ER3d`/`FR1`/`ExR3`
   (central-complex ring neurons) and `DN` matched `s-CPDN3A`. Use exact type
   matching — photoreceptors = `R1-6`/`R7`/`R8`, descending = type starts with
   `DN`. NOTE: this dataset's `side`/`x,y,z` fields are empty, so a *true*
   left/right hemisphere split is impossible; the L/R split is kept as an
   explicit, documented arbitrary proxy rather than implied biology.

## Tier 2 — makes "running" honest (real-time)

4. **Vectorize photoreceptor input injection.** Replace per-timestep Python
   loops over tens of thousands of indices with tensor index assignment.

5. **Precompute the transposed connectivity once** instead of calling
   `.t().coalesce()` twice every timestep.

6. **Close the wall-clock vs simulated-time gap.** `dt = 1e-3` advanced only one
   step per control loop, so the brain ran ~20-50x slower than a real fly. Run
   N sub-steps per control cycle so simulated time tracks real time.

## Safety note

These scripts command a real drone. Motor-decoding / mapping changes (#3) must
be bench-tested with props off before any flight.
