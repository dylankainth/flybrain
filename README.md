# Fruit Fly Brain on Drone: Complete Implementation

Load the complete adult *Drosophila melanogaster* connectome (139k neurons, 50M synapses) into a neural simulator, test sensorimotor control in simulation, then deploy to a real DJI Tello drone.

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Phase 1: Simulation

**Step 1: Download Connectome Data**
```bash
python download_connectome.py
```
Creates `fly_neurons.csv` and `fly_synapses.csv` from FlyWire (~10 min on good internet)

**Step 2-5: Run Full Simulation**
```bash
python simulate.py
```
This orchestrates:
- `fly_brain.py`: 127k neuron LIF model with full FlyWire connectivity
- `drone_sim.py`: PyBullet physics simulator for quadcopter
- `io_mapping.py`: Sensory encoding (vision → neurons) and motor decoding (neurons → thrust commands)
- Generates `simulation_results.png` showing trajectory and neural control signals

### 3. Phase 2: Hardware (Optional)

After validating simulation behavior:
```bash
python tello_brain.py
```
Deploys the same fly brain to a real DJI Tello drone using live camera feed.

## Architecture

```
FlyWire Connectome (139k neurons, 50M synapses)
        ↓
  [FlyBrain LIF Model]
   ↑         ↓
Sensory    Motor
Input    Decoding
   ↑         ↓
[Vision] → [Thrust Commands]
   ↑         ↓
[DroneSimulator (PyBullet) / DJI Tello]
```

## Key Files

- `download_connectome.py` — Fetch real fly connectome from FlyWire API
- `fly_brain.py` — Brian2 leaky integrate-and-fire neural model
- `drone_sim.py` — PyBullet quadcopter physics
- `io_mapping.py` — Sensory encoding & motor decoding
- `simulate.py` — Main simulation loop
- `tello_brain.py` — Hardware deployment to DJI Tello

## Validation Checklist

- [ ] `download_connectome.py` runs, creates CSV files
- [ ] `simulate.py` runs for 60 seconds without errors
- [ ] `simulation_results.png` shows reasonable drone trajectory
- [ ] Adjust `MotorDecoder.gain_*` parameters for stable behavior
- [ ] Test on Tello in open space

## Troubleshooting

**Brain initialization fails**
- Verify `fly_neurons.csv` and `fly_synapses.csv` exist in current directory
- Check they're readable (not corrupted)

**Drone crashes immediately in sim**
- Reduce `MotorDecoder.gain_forward` by half
- Adjust `SensoryEncoder.brightness_threshold` (try 100-150)

**Drone doesn't respond to vision**
- Add debug prints in `io_mapping.py` to check `sensory_input`
- Visual neuron indices should be continuous range
- Try injecting constant current: `brain.inject_sensory({range(1000, 1500): 50*pA})`

## References

- Dorkenwald et al. 2024. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138
- Shiu et al. 2024. A Drosophila computational brain model. *Nature* 634, 210–219
- FlyWire: https://codex.flywire.ai
- CAVEclient: https://github.com/seung-lab/CAVEclient
- Brian2: https://brian2.readthedocs.io
- PyBullet: https://pybullet.org
