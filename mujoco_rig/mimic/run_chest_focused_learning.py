"""Concentrate impact practice on +X while retaining matched quiet rehearsal."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import traceback
from unittest.mock import patch

from .audit_stand_failures import read, save
from . import run_chest_rehearsal_learning as rehearsal


def run(args):
    settings = read(args.protocol)
    assert settings['training_direction'] == [1, 0, 0]
    original_curriculum = rehearsal.study.onset_curriculum
    original_protect = rehearsal.study.support.protect_used

    @contextmanager
    def focused(settings, schedule):
        with original_curriculum(settings, schedule):
            from . import stand_ball_task
            base = stand_ball_task.StandBallAMPTask
            evidence_base = rehearsal.study.pilot.TrainingEvidence

            class FocusedTask(base):
                def __init__(self, *values, **keywords):
                    super().__init__(*values, **keywords)
                    self.direction.zero_()
                    self.direction[:, 0] = 1.

                def observation_contract(self):
                    return dict(super().observation_contract(),
                                training_impact_direction=[1, 0, 0])

            class FocusedEvidence(evidence_base):
                def step(self, action):
                    direction = self.task.direction.detach().clone()
                    result = super().step(action)
                    self.rows[-1]['training_direction'] = direction
                    return result

            with patch.object(stand_ball_task, 'StandBallAMPTask', FocusedTask), \
                    patch.object(rehearsal.study.pilot, 'TrainingEvidence', FocusedEvidence):
                yield

    def protect(out, paths, protocol, settings):
        return original_protect(out, [*paths, Path(__file__)], protocol, settings)

    with patch.object(rehearsal.study, 'onset_curriculum', focused), \
            patch.object(rehearsal.study.support, 'protect_used', protect):
        rehearsal.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
