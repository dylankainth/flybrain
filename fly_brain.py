"""
Leaky integrate-and-fire model of Drosophila connectome.
Based on Shiu et al. 2024 (Nature).
"""

from brian2 import *
import pandas as pd
import numpy as np

class FlyBrain:
    """127k neuron LIF model with FlyWire connectome"""

    # LIF parameters (from Shiu et al. 2024)
    El = -65 * mV          # Resting potential
    EI = -90 * mV          # Inhibitory reversal
    tau = 10 * ms          # Membrane time constant
    Vt = -50 * mV          # Spike threshold

    def __init__(self, synapse_file='fly_synapses.csv',
                 neuron_file='fly_neurons.csv',
                 verbose=True):
        """
        Initialize brain from FlyWire connectome.

        Args:
            synapse_file: Path to fly_synapses.csv
            neuron_file: Path to fly_neurons.csv
            verbose: Print progress
        """

        self.verbose = verbose
        self._log("[FlyBrain] Initializing...")

        # Load connectome
        self._log(f"  Loading neurons from {neuron_file}...")
        self.neurons_df = pd.read_csv(neuron_file)
        self.n_neurons = len(self.neurons_df)

        self._log(f"  Loading synapses from {synapse_file}...")
        self.synapses_df = pd.read_csv(synapse_file)
        self.n_synapses = len(self.synapses_df)

        # Create neuron group
        self._log(f"  Creating {self.n_neurons} neurons in Brian2...")
        start_scope()

        eqs = '''
        dv/dt = (El - v + I) / tau : volt
        I : amp
        '''

        self.neurons = NeuronGroup(
            self.n_neurons, eqs,
            threshold='v > Vt',
            reset='v = El',
            method='exponential_euler',
            namespace={
                'El': self.El,
                'tau': self.tau,
                'Vt': self.Vt
            }
        )

        # Initialize to resting
        self.neurons.v = self.El
        self.neurons.I = 0 * amp

        # Lookup: root_id → neuron index (needed before building synapses)
        self.root_id_to_idx = {
            root_id: idx
            for idx, root_id in enumerate(self.neurons_df['root_id'].values)
        }

        # Build synapses
        self._log(f"  Building {self.n_synapses} synapses...")
        self._build_synapses()

        # Monitors for debugging
        self.spikes = SpikeMonitor(self.neurons)

        self._log("[OK] FlyBrain initialized")

    def _build_synapses(self):
        """Construct synaptic connectivity from FlyWire data"""

        # Extract connectivity
        pre_ids = self.synapses_df['pre_root_id'].values
        post_ids = self.synapses_df['post_root_id'].values

        # Neurotransmitter type (excitatory/inhibitory)
        transmitter = self.synapses_df['neurotransmitter'].fillna('unknown').values

        # Synapse size (proxy for strength)
        size = self.synapses_df['size'].values

        # Map root IDs to neuron indices
        pre_indices = np.array([
            self.root_id_to_idx.get(pid, -1) for pid in pre_ids
        ])
        post_indices = np.array([
            self.root_id_to_idx.get(pid, -1) for pid in post_ids
        ])

        # Filter out unmapped neurons
        valid = (pre_indices >= 0) & (post_indices >= 0)
        pre_indices = pre_indices[valid]
        post_indices = post_indices[valid]
        transmitter = transmitter[valid]
        size = size[valid]

        n_valid = np.sum(valid)
        self._log(f"    Valid synapses: {n_valid}/{len(valid)} "
                  f"({100*n_valid/len(valid):.1f}%)")

        # Compute weights from synapse size + transmitter type
        weights = np.zeros_like(size, dtype=float)

        for i, nt in enumerate(transmitter):
            # Normalize synapse size
            w = size[i] / (size.max() + 1e-6)

            # Sign based on neurotransmitter (Eckstein et al. 2024 predictions)
            if nt in ['acetylcholine', 'ACh']:
                weights[i] = w * 100      # Excitatory (pA)
            elif nt in ['GABA', 'glutamate', 'Glu']:
                weights[i] = -w * 100     # Inhibitory
            else:
                weights[i] = w * 50       # Unknown: weak excitatory

        # Create synapse group
        self._log(f"    Creating Synapses object...")
        self.synapses = Synapses(
            self.neurons, self.neurons,
            '''w : amp''',
            on_pre='I += w',
            method='exponential_euler'
        )

        self.synapses.connect(i=pre_indices, j=post_indices)
        self.synapses.w[:] = weights * pA

        self._log(f"    [OK] {len(self.synapses)} synapses connected")

    def inject_sensory(self, sensory_input):
        """
        Inject sensory input into designated sensory neurons.

        Args:
            sensory_input: dict of {neuron_root_ids: activation}
                           Activation in pA (dimensionless values are converted)
        """
        for root_ids, activation in sensory_input.items():
            # Convert root IDs to neuron indices
            if isinstance(root_ids, (list, tuple)):
                root_ids = list(root_ids)
            elif isinstance(root_ids, (np.ndarray, range)):
                root_ids = list(root_ids)

            # Map root IDs to indices
            neuron_indices = []
            for rid in root_ids:
                if rid in self.root_id_to_idx:
                    neuron_indices.append(self.root_id_to_idx[rid])

            if neuron_indices:
                neuron_indices = np.array(neuron_indices)
                # Ensure activation has proper units
                if not hasattr(activation, 'units'):
                    activation = activation * pA
                self.neurons.I[neuron_indices] += activation

    def step(self, duration_ms=50):
        """
        Simulate one timestep.

        Args:
            duration_ms: Duration in milliseconds
        """
        run(duration_ms * ms)

    def get_spikes(self, neuron_indices, reset=False):
        """
        Get spike count for neurons in last step.

        Args:
            neuron_indices: Indices to read (range, list, or array)
            reset: Clear spike monitor after reading
        """
        if isinstance(neuron_indices, range):
            neuron_indices = set(neuron_indices)
        elif not isinstance(neuron_indices, set):
            neuron_indices = set(neuron_indices)

        # Count spikes from this set of neurons
        spikes = np.sum(np.isin(self.spikes.i, list(neuron_indices)))
        if reset:
            self.spikes.record = False
            self.spikes.record = True
        return spikes

    def get_firing_rate(self, neuron_indices, window_ms=50):
        """
        Get firing rate (Hz) for a set of neurons.

        Args:
            neuron_indices: Indices to read
            window_ms: Time window in ms
        """
        spikes = self.get_spikes(neuron_indices)
        firing_rate = spikes / (len(neuron_indices) * window_ms / 1000)
        return firing_rate

    def get_voltage(self, neuron_indices):
        """Read membrane voltage of neurons"""
        return self.neurons.v[neuron_indices]

    def reset(self):
        """Reset brain to resting state"""
        self.neurons.v = self.El
        self.neurons.I = 0 * amp
        self.spikes.record = False
        self.spikes.record = True

    def _log(self, msg):
        if self.verbose:
            print(msg)
