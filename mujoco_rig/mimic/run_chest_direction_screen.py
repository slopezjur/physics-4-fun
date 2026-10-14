"""Frozen directional throws with read-only native insertion-clearance admission."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import traceback
from unittest.mock import patch

from .audit_stand_failures import save
from . import run_chest_frozen_followup as followup
from . import run_frozen_ball_screen as screen


def run(args):
    original_evaluate, original_compare, original_protect = screen.native_evaluate, screen.compare, screen.protect_used

    def evaluate(*values, **kwargs):
        from . import stand_ball_native
        engine_type = stand_ball_native.StandBallNative
        original_launch = engine_type.launch_ball
        events = []

        def launch(engine, ids, position, velocity):
            original_launch(engine, ids, position, velocity)
            # launch_ball has already called mj_forward; read contacts without advancing physics.
            for case in ids.tolist():
                contacts = []
                data = engine.datas[case]
                for contact in data.contact[:data.ncon]:
                    a, b = engine.rig.model.geom_bodyid[contact.geom]
                    body = b if a == engine.ball_body else a if b == engine.ball_body else 0
                    if 0 < body < engine.ball_body:
                        contacts.append(dict(body=engine.rig.model.body(body).name, distance_m=float(contact.dist)))
                events.append(dict(case=case, contacts=contacts,
                    clear=all(c['distance_m'] >= -1e-9 for c in contacts)))

        with patch.object(engine_type, 'launch_ball', launch):
            result = original_evaluate(*values, **kwargs)
        result['launch_geometry'] = dict(cases=events, all_cases_launched=len(events) == 8,
            all_clear=len(events) == 8 and all(e['clear'] for e in events), penetration_tolerance_m=1e-9)
        save(values[4] / 'report.json', result)
        return result

    def compare(native, gpu, dt):
        result = original_compare(native, gpu, dt)
        result['native_launch_clear'] = native['launch_geometry']['all_clear']
        result['passed'] &= result['native_launch_clear']
        return result

    def protect(out, paths, protocol, settings):
        return original_protect(out, [*paths, Path(__file__)], protocol, settings)

    with ExitStack() as stack:
        stack.enter_context(patch.object(screen, 'native_evaluate', evaluate))
        stack.enter_context(patch.object(screen, 'compare', compare))
        stack.enter_context(patch.object(screen, 'protect_used', protect))
        followup.run(args)


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
