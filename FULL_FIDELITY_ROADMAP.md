# Full-Fidelity Fruit Fly Brain Model

## The Goal
Build a neurobiologically accurate computational model of the Drosophila melanogaster brain that:
- Respects circuit structure (visual, olfactory, motor systems)
- Uses realistic neuron models (not generic LIF)
- Includes plasticity and neuromodulation
- Validates against real fly behavior
- Can drive a robot controller

**Timeline:** 3-6 months for a real model (not months for a toy)

---

## Phase 1: Connectome Analysis & Literature Integration

### 1.1 Identify Major Subsystems from Connectome
```
Step 1: Parse connectome for neuropil regions
  - Visual system (medulla, lobula, lamina, T4/T5)
  - Olfactory system (antennal lobe, mushroom body)
  - Central complex (ellipsoid body, fan-shaped body, protocerebral bridge)
  - Motor circuits (descending neurons, premotor circuits)
  - Modulatory systems (dopamine, octopamine neurons)

Step 2: Extract subgraphs for each system
  - Get connectivity matrix for each region
  - Identify key populations (e.g., T4/T5 cells → directional tuning)
  - Map inputs/outputs across regions

Step 3: Identify bottleneck circuits
  - What's the minimal sufficient model?
  - Which circuits are well-characterized?
```

### 1.2 Gather Circuit-Specific Literature
We need papers that describe actual dynamics:

**Visual System:**
- Takemura et al. 2015 (medulla circuitry)
- Shinomiya et al. 2019 (lamina circuits)
- Klapoetke et al. 2017 (T4/T5 direction selectivity) ← KEY
- Joesch et al. 2010 (optic flow processing)

**Motor Circuits:**
- Namiki et al. 2023 (descending neurons) ← MOST RECENT
- Fox & Daniel 2008 (motor control)
- Mendes et al. 2013 (escape behavior)

**Learning/Plasticity:**
- Owald & Sigrist 2021 (synaptic plasticity in Drosophila)
- Eschbach et al. 2020 (neuromodulation in mushroom body)
- Cohn et al. 2015 (dopamine-driven learning)

---

## Phase 2: Build Circuit-Specific Neuron Models

### 2.1 Replace Generic LIF with Circuit-Appropriate Models

**Visual System Neurons (T4/T5):**
```python
# Hodgkin-Huxley style, with direction selectivity
dv/dt = -g_L(v - E_L) - g_Na*m³*h*(v - E_Na) - g_K*n⁴*(v - E_K) + I
dm/dt = α_m(v)(1-m) - β_m(v)*m
... (gating variables)

# Specific property: Motion integration
# Requires: dendritic computation (not soma-only)
```

**Descending Neurons (motor output):**
```python
# Smaller, faster neurons with strong outputs
# Model: compartmental (axon, soma, dendrite)
# Properties: all-or-nothing, strong synaptic gain
```

**Dopamine Neurons (learning signals):**
```python
# Tonic firing + burst response to reward/punishment
# Modulates synaptic plasticity of readout neurons
```

### 2.2 Implement Realistic Synaptic Models
Current: `I += w * spike`
Better: `g_syn = g_max * (V - E_syn) * (rise * decay)` with actual kinetics

Need per-synapse:
- Rise/decay time constants (from physiology)
- Transmission delays (axonal propagation)
- Plasticity rules (STDP, neuromodulation-dependent)

---

## Phase 3: Sensory Encoding (Grounded in Biology)

### 3.1 Visual Input Processing
```
Real fly vision chain:
  Photoreceptor (R1-R8)
    ↓ (transient, logarithmic)
  Lamina L-cells
    ↓ (bandpass filtering)
  Medulla M-cells
    ↓ (spatial filtering, temporal processing)
  T4/T5 neurons
    ↓ (DIRECTIONAL TUNING via motion opponency)
  
Current model: brightness → current
Should be: optic flow → T4/T5 direction tuples
```

### 3.2 Implement Optic Flow Encoding
```python
class OpticalFlow:
    def compute(self, video_frame_t0, video_frame_t1):
        # Compute motion vectors (Lucas-Kanade or Fluss-Engels)
        flow = compute_flow(frame_t0, frame_t1)
        
        # Decompose into 4 directions (preferred by T4/T5)
        up, down, left, right = decompose_to_cardinal(flow)
        
        # Activate direction-selective neurons
        # T4: prefers front-to-back motion (upward on retina)
        # T5: prefers back-to-front motion (downward on retina)
        return {
            'T4_up': up,
            'T4_down': down,
            'T5_up': down,  # (opposite polarity)
            'T5_down': up,
        }
```

---

## Phase 4: Motor Circuit Mapping & Control

### 4.1 Identify Real Descending Neurons
From Namiki et al. 2023 and connectome:
```
Descending neuron types:
- DNg02 → controls forward walking
- DNa02 → controls left turn
- DNa01 → controls right turn
- DDC04 → controls climbing
... (and 20+ others)

NOT: arbitrary neuron ranges
But: validated IDs from literature
```

### 4.2 Decode Motor Output from Connectome
```python
# Instead of: "read motor neurons, average firing rate"
# Do: "identify synapse strengths to motor neurons"

# Get actual synaptic weights from connectome
motor_output = sum(
    w_ij * spike_rate_i 
    for each upstream neuron i
)
# This is data-driven, not assumed
```

---

## Phase 5: Learning & Plasticity

### 5.1 Synaptic Plasticity Rules
Implement STDP + neuromodulation:

```python
class Synapse:
    def update(self, pre_spike, post_spike, dopamine_level):
        # Spike-timing dependent plasticity
        dt = post_spike_time - pre_spike_time
        delta_w = STDP_rule(dt)
        
        # Dopamine-gated: only potentiate/depress if DA present
        if dopamine_level > threshold:
            self.w += dopamine_level * delta_w
        
        # Different signs for punishment (negative DA)
        if dopamine_level < -threshold:
            self.w -= abs(dopamine_level) * delta_w
```

### 5.2 Implement Reward/Punishment Signals
```python
# Real fly: learns to approach odor+ (reward) or avoid odor- (punishment)
# Dopamine encodes reward prediction error

class RewardSignal:
    def compute(self, stimulus, outcome):
        # outcome: 1 (reward), -1 (punishment), 0 (neutral)
        error = outcome - expected_outcome
        dopamine = dopamine_neurons.fire_at_rate(error)
        return dopamine  # Modulates learning
```

---

## Phase 6: Behavioral Validation

### 6.1 Benchmark Against Real Fly Experiments
```
Test cases (published data):
1. Optomotor response
   - Rotate visual scene → fly turns toward motion
   - Check: Does T4/T5 activation → correct turn command?
   
2. Odor avoidance
   - Present aversive odor → fly turns away
   - Check: Does mushroom body learning drive avoidance?
   
3. Escape responses
   - Looming stimulus → escape jump
   - Check: Do descending neurons trigger right motor sequence?
   
4. Habituation
   - Repeated stimulus → response decreases
   - Check: Does fatigue/adaptation match real fly timescale?
```

### 6.2 Compare Neural Activity to Recordings
```
Real fly data (two-photon imaging):
- Visual neurons respond to motion (direction-selective)
- Motor neurons fire phase-locked to behavior
- Dopamine neurons burst with reward

Our model should match:
- Receptive fields and temporal tuning
- Population response structure
- Learning rates and timescales
```

---

## Phase 7: Robot Integration

### 7.1 Once Model is Validated
```python
# Only after behavioral validation:
robot.vision = get_camera_frame()
optic_flow = compute_flow(robot.vision)  # Biological encoding
brain.sensory_input = optic_flow         # Feed into model
brain.simulate(dt=10ms)                  # Run one timestep
motor_command = brain.get_descending_neuron_activity()
robot.execute(motor_command)
```

---

## What This Requires

### Knowledge
- [ ] Connectomics (how to read FlyWire data)
- [ ] Neurophysiology (fly neural dynamics)
- [ ] Computational neuroscience (modeling circuits)
- [ ] Behavior (what flies actually do)
- [ ] Learning theory (plasticity rules)

### Tools
- [ ] Circuit analysis: custom Python scripts
- [ ] Modeling: Brian2 (small circuits) + NEST/GeNN (larger)
- [ ] Validation: real fly behavior data
- [ ] Simulation: GPU acceleration needed

### Data
- [ ] FlyWire connectome (✓ we have it)
- [ ] Cell type annotations (✓ we have it)
- [ ] Published electrophysiology recordings
- [ ] Behavioral video data (e.g., from VRtrack, FlyTracker)
- [ ] Descending neuron identifications (Namiki et al. 2023)

---

## Realistic Milestones

| Phase | Goal | Timeline | Validation |
|-------|------|----------|-----------|
| **1** | Identify visual subsystem | 2 weeks | Confirm T4/T5 architecture |
| **2** | Build T4/T5 direction-selective model | 3 weeks | Directional tuning curves |
| **3** | Implement optic flow input | 2 weeks | Motion response correct |
| **4** | Add motor circuits | 2 weeks | Descending neurons fire right timing |
| **5** | Add learning (optional first pass) | 2 weeks | Simple conditioning |
| **6** | Validate against 3-4 behavior tests | 4 weeks | Pass published benchmarks |
| **7** | Robot integration | 1 week | Closed-loop flight |
| **TOTAL** | | **3-4 months** | Real neuroscience |

---

## Why This Actually Works

1. **Data-driven:** Every choice grounded in connectome or papers
2. **Modular:** Build one circuit at a time, validate each
3. **Testable:** Compare to real fly behavior quantitatively
4. **Scalable:** Start small (visual system), add circuits as needed
5. **Publishable:** This becomes a real paper, not a toy

---

## The Real Output

At the end:
- A computational model that **actually** respects fly neuroscience
- Validated against real behavior
- Could be deployed on a robot (or fly)
- Contributes to understanding of neural computation
- **Honest science**

---

## Starting Point: Visual System Deep-Dive

**Best place to start:** T4/T5 direction-selective circuits

Why:
- Well-characterized (Joesch, Klapoetke papers)
- Connectome is clear (Takemura, Shinomiya)
- Function is testable (rotate visual scene)
- Not too large (∼1000 neurons in core circuit)
- **Can be done properly in 2-3 weeks**

Next step: Extract T4/T5 subgraph from FlyWire, implement direction selectivity model.

---

**Ready to do this right?**
