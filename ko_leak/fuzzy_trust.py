import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecedents & Consequent universes
F1_UNI = np.arange(0, 1.01, 0.01)         # macro-F1 in [0,1]
DEV_UNI = np.arange(0, 2.01, 0.01)        # normalized deviation in [0,2]
ANOM_UNI = np.arange(0, 1.01, 0.01)       # attack ratio in [0,1]
TRUST_UNI = np.arange(0, 1.01, 0.01)      # trust in [0,1]

# Define fuzzy variables
f1 = ctrl.Antecedent(F1_UNI, 'f1')
deviation = ctrl.Antecedent(DEV_UNI, 'deviation')
anomaly = ctrl.Antecedent(ANOM_UNI, 'anomaly')
trust = ctrl.Consequent(TRUST_UNI, 'trust')

# Membership functions
f1['low'] = fuzz.trimf(f1.universe, [0, 0, 0.5])
f1['medium'] = fuzz.trimf(f1.universe, [0.3, 0.6, 0.9])
f1['high'] = fuzz.trimf(f1.universe, [0.7, 1.0, 1.0])

deviation['small'] = fuzz.trimf(deviation.universe, [0, 0, 0.8])
deviation['moderate'] = fuzz.trimf(deviation.universe, [0.5, 1.0, 1.5])
deviation['large'] = fuzz.trimf(deviation.universe, [1.2, 2.0, 2.0])

anomaly['benign'] = fuzz.trimf(anomaly.universe, [0, 0, 0.3])
anomaly['suspicious'] = fuzz.trimf(anomaly.universe, [0.2, 0.5, 0.8])
anomaly['malicious'] = fuzz.trimf(anomaly.universe, [0.6, 1.0, 1.0])

trust['low'] = fuzz.trimf(trust.universe, [0, 0, 0.4])
trust['medium'] = fuzz.trimf(trust.universe, [0.3, 0.6, 0.9])
trust['high'] = fuzz.trimf(trust.universe, [0.7, 1.0, 1.0])

# Rule base (with broad coverage)
rules = [
    ctrl.Rule(f1['high'] & deviation['small'] & anomaly['benign'], trust['high']),
    ctrl.Rule(f1['medium'] & deviation['moderate'] & anomaly['suspicious'], trust['medium']),
    ctrl.Rule(f1['low'] | deviation['large'] | anomaly['malicious'], trust['low']),
    ctrl.Rule(anomaly['suspicious'] & f1['high'], trust['medium']),
    ctrl.Rule(anomaly['benign'] & deviation['small'], trust['high']),
    ctrl.Rule(f1['medium'] & deviation['small'] & anomaly['benign'], trust['high']),
    ctrl.Rule(f1['low'] & deviation['small'] & anomaly['benign'], trust['medium']),
]

trust_ctrl = ctrl.ControlSystem(rules)

def compute_fuzzy_trust(f1_score, deviation_value, anomaly_score):
    """Compute a robust fuzzy trust score.
    - Creates a fresh simulation per call (avoids stale state).
    - Clips inputs to their universes.
    - Falls back to a weighted heuristic if the system cannot infer an output.
    """
    import numpy as _np
    # Clip to universes
    f1_in = float(_np.clip(f1_score, 0.0, 1.0))
    dev_in = float(_np.clip(deviation_value, 0.0, 2.0))
    anom_in = float(_np.clip(anomaly_score, 0.0, 1.0))

    sim = ctrl.ControlSystemSimulation(trust_ctrl)
    try:
        sim.input['f1'] = f1_in
        sim.input['deviation'] = dev_in
        sim.input['anomaly'] = anom_in
        sim.compute()
        out = sim.output.get('trust', None)
        if out is None or _np.isnan(out):
            raise ValueError('Fuzzy engine produced no output')
        return float(_np.clip(out, 0.0, 1.0))
    except Exception:
        # Fallback: interpretable weighted heuristic
        fallback = 0.5 * f1_in + 0.3 * (1.0 - min(dev_in / 2.0, 1.0)) + 0.2 * (1.0 - anom_in)
        return float(_np.clip(fallback, 0.0, 1.0))
