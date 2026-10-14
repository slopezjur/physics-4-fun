"""Run the sealed recovery comparison against the corrected provenance preflight."""
import json
from pathlib import Path
from unittest.mock import patch

from . import analyze_retracted_recovery as prior
from .run_support_skill_study import digest


OUT = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')


def main():
    inputs = json.loads((OUT/'inputs.json').read_text(encoding='utf-8'))
    source = Path(__file__)
    assert inputs[str(source)] == digest(source), 'Analysis wrapper changed after source sealing'
    with patch.object(prior, 'OUT', OUT):
        prior.main()


if __name__ == '__main__':
    main()
