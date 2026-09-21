"""Recovery quality measured over time; falling cannot improve a trial's outcome."""
import numpy as np

from recovery_reward import CONFIG, contact_transition, settled_state, support_state


LEGACY_METRIC_VERSION = 'height_proxy_v2'
CONTACT_METRIC_VERSION = 'loaded_contact_v3'


class RecoveryMetrics:
    def __init__(self, count, dt, flight_seconds):
        self.dt, self.flight_seconds = dt, flight_seconds
        self.hold = np.zeros(count)
        self.shots = np.zeros(count, dtype=int)
        self.since_shot = np.zeros(count)
        self.sliding = np.zeros(count)
        self.rapid_replants = np.zeros(count)
        self.action_change = np.zeros(count)
        self.alive_samples = np.zeros(count)

    def record(self, i, features, angular_velocity, alive, shots, action_delta):
        if shots != self.shots[i]:
            self.since_shot[i] = 0.0
            self.hold[i] = 0.0
            self.shots[i] = shots
        self.since_shot[i] += self.dt
        f = features
        error, _ = support_state(np, f['com'], f['velocity'], f['feet'], f['grounded'])
        settled = alive and settled_state(
            np, f['up'], f['pelvis_z'] / f['rest_pelvis_z'], f['velocity'],
            angular_velocity, f['foot_velocity'], f['grounded'], error)
        self.hold[i] = self.hold[i] + self.dt if settled else 0.0
        if alive:
            self.alive_samples[i] += 1
            speed = (np.sqrt(f['slip_speed_sq']) if 'slip_speed_sq' in f
                     else np.linalg.norm(f['foot_velocity'][:, :2], axis=-1))
            self.sliding[i] += (speed * f['grounded']).sum() * self.dt
            self.rapid_replants[i] += f['rapid_replants'] > 0
            self.action_change[i] += np.mean(np.abs(action_delta))

    def result(self, survived, require_shot):
        exposed = ((self.shots > 0)
                   & (self.since_shot >= self.flight_seconds + CONFIG.settle_seconds))
        recovered = (survived & (self.hold >= CONFIG.settle_seconds - 1e-9)
                     & (exposed if require_shot else True))
        return dict(recovered=100.0 * float(recovered.mean()), recovered_mask=recovered,
                    foot_sliding_m=float(self.sliding.mean()),
                    rapid_replants=float(self.rapid_replants.mean()),
                    action_change=float(self.action_change.sum()
                                        / max(self.alive_samples.sum(), 1)),
                    settled_hold_s=float(self.hold.mean()))


class HeightProxyRecoveryMetrics(RecoveryMetrics):
    """Freeze the existing promotion gate while contact measurements are tested.

    Own the legacy contact history independently of the environment's new load
    sensor. This preserves per-world outcomes, replants and sliding, not only the
    headline definition of a settled hold. Observations and physics are unchanged.
    """
    def __init__(self, count, dt, flight_seconds, rest_foot_z):
        super().__init__(count, dt, flight_seconds)
        self.rest_foot_z = np.asarray(rest_foot_z).copy()
        self.grounded = np.ones((count, 2), dtype=bool)
        self.since_landing = np.full((count, 2), CONFIG.replant_interval)

    def record(self, i, features, angular_velocity, alive, shots, action_delta):
        ground, elapsed, rapid = contact_transition(
            np, features['feet'][:, 2] - self.rest_foot_z, self.grounded[i],
            self.since_landing[i], self.dt)
        self.grounded[i], self.since_landing[i] = ground, elapsed
        historical = dict(features, grounded=ground, rapid_replants=rapid)
        historical.pop('slip_speed_sq', None)
        super().record(i, historical, angular_velocity, alive, shots, action_delta)
