"""Deterministic small-push protocol and recovery measurements; no reward changes."""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PushTrial:
    name: str
    phase: float
    force: tuple[float, float, float]
    start_step: int = 60
    duration_steps: int = 6
    trial_steps: int = 300
    body: str = "Chest"

    def validate(self, dt, reference_duration):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(reference_duration):
            raise ValueError("Invalid simulation timing")
        if len(self.force) != 3 or not all(math.isfinite(x) for x in (*self.force, self.phase)):
            raise ValueError("Nonfinite push parameters")
        if self.phase < 0 or self.phase + self.trial_steps * dt > reference_duration:
            raise ValueError("Trial must fit inside the reference")
        if any(type(x) is not int for x in (self.start_step, self.duration_steps, self.trial_steps)) \
                or self.start_step < 0 or self.duration_steps < 1 \
                or self.start_step + self.duration_steps + math.ceil(0.5 / dt) > self.trial_steps:
            raise ValueError("Push must leave at least half a second for recovery")

    def force_at(self, step):
        return self.force if self.start_step <= step < self.start_step + self.duration_steps else (0., 0., 0.)

    def to_dict(self):
        return asdict(self)


def baseline_trials():
    trials = []
    for phase in (0., 0.8):
        for start in (60, 120):
            trials.append(PushTrial(f"none-p{phase}-t{start}", phase, (0., 0., 0.), start))
            for strength in (10., 20., 40.):
                for label, direction in (("+x", (1, 0, 0)), ("-x", (-1, 0, 0)),
                                         ("+y", (0, 1, 0)), ("-y", (0, -1, 0))):
                    trials.append(PushTrial(f"{strength:g}N-{label}-p{phase}-t{start}", phase,
                                            tuple(strength * x for x in direction), start))
    return trials


RECOVERY = {"height_m": 0.75, "tilt_degrees": 15., "horizontal_speed_m_s": 0.2,
            "angular_speed_rad_s": 1., "each_foot_load_n": 5., "settle_seconds": 0.5}


class RecoveryMetrics:
    """Recovery requires the final uninterrupted settled window, plus trial survival."""
    def __init__(self, trial, dt):
        self.trial, self.dt = trial, dt
        self.origin = self.previous_feet = self.previous_contacts = None
        self.stable_steps = 0
        self.recovery_seconds = None
        self.displacement = self.foot_travel = 0.
        self.contact_switches = 0
        self.last_step = 0

    def sample(self, completed_step, q, v, feet, loads):
        self.last_step = completed_step
        # Capture the state just before the pulse; include the pulse in travel metrics.
        if completed_step <= self.trial.start_step:
            self.origin = np.array(q[:2])
            self.previous_feet = np.array(feet)
            self.previous_contacts = np.asarray(loads) > RECOVERY["each_foot_load_n"]
            return
        if self.origin is None:
            raise ValueError("Metrics require an initial sample before the push")
        contacts = np.asarray(loads) > RECOVERY["each_foot_load_n"]
        self.displacement = max(self.displacement, float(np.linalg.norm(q[:2] - self.origin)))
        self.foot_travel += float(np.linalg.norm(np.asarray(feet) - self.previous_feet, axis=-1).sum())
        self.contact_switches += int(np.count_nonzero(contacts != self.previous_contacts))
        self.previous_feet, self.previous_contacts = np.array(feet), contacts
        if completed_step <= self.trial.start_step + self.trial.duration_steps:
            return
        up_z = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
        settled = q[2] >= RECOVERY["height_m"] and up_z >= math.cos(math.radians(RECOVERY["tilt_degrees"])) \
            and np.linalg.norm(v[:2]) <= RECOVERY["horizontal_speed_m_s"] \
            and np.linalg.norm(v[3:6]) <= RECOVERY["angular_speed_rad_s"] and contacts.all()
        self.stable_steps = self.stable_steps + 1 if settled else 0
        if not settled:
            self.recovery_seconds = None
        elif self.stable_steps == math.ceil(RECOVERY["settle_seconds"] / self.dt):
            self.recovery_seconds = (completed_step - self.trial.start_step - self.trial.duration_steps) * self.dt

    def result(self, fallen):
        survived = not fallen and self.last_step >= self.trial.trial_steps
        return {"survived": survived, "recovered": survived and self.recovery_seconds is not None,
                "recovery_seconds": self.recovery_seconds if survived else None,
                "survival_seconds": self.last_step * self.dt,
                "max_horizontal_displacement_m": self.displacement,
                "foot_travel_m": self.foot_travel, "contact_switches": self.contact_switches}
