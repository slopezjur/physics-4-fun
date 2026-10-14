"""Advance posture pressure using measured standing, never an episode clock."""
from dataclasses import dataclass, field


STAGES = [
    dict(name='balance', posture_weight=.10,
         tilt_degrees=None, clearance_deficit_m=None),
    dict(name='extension', posture_weight=.25,
         tilt_degrees=[45., 45., 60.], clearance_deficit_m=[.10, .15, .20]),
    dict(name='upright', posture_weight=.50,
         tilt_degrees=[30., 30., 40.], clearance_deficit_m=[.07, .09, .13]),
    dict(name='quiet_upright', posture_weight=.70,
         tilt_degrees=[20., 20., 30.], clearance_deficit_m=[.05, .06, .10]),
]
CURRICULUM = dict(schema='torque_stand_posture_curriculum_v1', stages=STAGES,
    consecutive_assessments_to_advance=2,
    consecutive_retention_failures_to_stop=2,
    advancement='All eight native cases survive 30 seconds and finish with two stable seconds at the current stage',
    retention='Preserve every initially surviving case and every initially balance-settled case',
    regression='Reduce posture pressure by one stage; stop after repeated source-retention loss',
    final_admission='Fixed whole-body gate, independent of curriculum stage')
UPRIGHT_HORIZONS = (2., 5., 10., 30.)
UPRIGHT_CURRICULUM = dict(schema='torque_stand_upright_horizon_curriculum_v1',
    horizons_seconds=list(UPRIGHT_HORIZONS), consecutive_assessments_to_advance=2,
    consecutive_retention_failures_to_stop=2,
    readiness='All eight cases survive the current horizon; finish upright and balanced for min(2, horizon/4) seconds',
    retention=CURRICULUM['retention'],
    regression='Shorten the horizon on quiet-retention loss; stop after two consecutive losses',
    final_admission='Unchanged 30-second whole-body gate and conditional 600-second hold')


@dataclass
class PostureCurriculum:
    stage: int = 0
    ready_streak: int = 0
    retention_failures: int = 0
    baseline: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def set_baseline(self, metrics):
        self.baseline = {case['case']: dict(survived=case['survived'],
            balance_settled=case['survived'] and case['final_balance_stable_seconds'] >= 2.)
            for case in metrics['cases']}

    def retention(self, metrics):
        if {case['case'] for case in metrics['cases']} != set(self.baseline):
            raise ValueError('Curriculum assessment cases differ from the initial standing cases')
        lost = []
        for case in metrics['cases']:
            previous = self.baseline[case['case']]
            if ((previous['survived'] and not case['survived']) or
                    (previous['balance_settled'] and
                     not (case['survived'] and case['final_balance_stable_seconds'] >= 2.))):
                lost.append(case['case'])
        return dict(passed=not lost, lost_cases=lost)

    def observe(self, metrics, iteration):
        retention = self.retention(metrics)
        self.retention_failures = 0 if retention['passed'] else self.retention_failures + 1
        before = self.stage
        decision = 'hold'
        if not retention['passed'] or (self.stage > 0 and not metrics['balance_passed']):
            self.stage = max(0, self.stage - 1)
            self.ready_streak = 0
            decision = 'reduce_posture_pressure' if self.stage != before else 'retention_regression'
        elif metrics['stage_passed']:
            self.ready_streak += 1
            if self.ready_streak >= CURRICULUM['consecutive_assessments_to_advance'] and self.stage < len(STAGES)-1:
                self.stage += 1
                self.ready_streak = 0
                decision = 'advance'
        else:
            self.ready_streak = 0
        stop = self.retention_failures >= CURRICULUM['consecutive_retention_failures_to_stop']
        event = dict(iteration=iteration, assessed_stage=before, next_stage=self.stage,
            decision=decision, ready_streak=self.ready_streak, retention=retention,
            consecutive_retention_failures=self.retention_failures, stop_requested=stop)
        self.events.append(event)
        return event

    def report(self):
        return dict(protocol=CURRICULUM, final_stage=self.stage,
                    final_stage_name=STAGES[self.stage]['name'], baseline=self.baseline, events=self.events)


@dataclass
class UprightHorizonCurriculum(PostureCurriculum):
    """Short upright holds must improve without discarding long quiet-standing evidence."""

    @property
    def horizon(self):
        return UPRIGHT_HORIZONS[self.stage]

    def observe(self, short_metrics, quiet_metrics, iteration):
        if short_metrics['seconds'] != self.horizon:
            raise ValueError('Short assessment duration differs from the training horizon')
        if {case['case'] for case in short_metrics['cases']} != set(self.baseline):
            raise ValueError('Short assessment cases differ from the retained standing cases')
        retention = self.retention(quiet_metrics)
        self.retention_failures = 0 if retention['passed'] else self.retention_failures+1
        before = self.stage
        stable_seconds = min(2., self.horizon/4.)
        ready = all(case['survived'] and case['final_stable_seconds'] >= stable_seconds
                    for case in short_metrics['cases'])
        decision = 'hold'
        if not retention['passed']:
            self.stage = max(0, self.stage-1)
            self.ready_streak = 0
            decision = 'shorten_horizon' if self.stage != before else 'retention_regression'
        elif ready:
            self.ready_streak += 1
            if self.ready_streak >= UPRIGHT_CURRICULUM['consecutive_assessments_to_advance'] and self.stage < len(UPRIGHT_HORIZONS)-1:
                self.stage += 1
                self.ready_streak = 0
                decision = 'extend_horizon'
        else:
            self.ready_streak = 0
        stop = self.retention_failures >= UPRIGHT_CURRICULUM['consecutive_retention_failures_to_stop']
        event = dict(iteration=iteration, assessed_stage=before, next_stage=self.stage,
            assessed_horizon_seconds=UPRIGHT_HORIZONS[before], next_horizon_seconds=self.horizon,
            required_stable_seconds=stable_seconds, short_upright_ready=ready,
            decision=decision, ready_streak=self.ready_streak, retention=retention,
            consecutive_retention_failures=self.retention_failures, stop_requested=stop)
        self.events.append(event)
        return event

    def report(self):
        return dict(protocol=UPRIGHT_CURRICULUM, final_stage=self.stage,
            final_stage_name=f'upright_{self.horizon:g}s', training_horizon_seconds=self.horizon,
            baseline=self.baseline, events=self.events)
