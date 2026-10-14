"""Adapt audit batches to the sealed head's two-dimensional input contract."""
from pathlib import Path
from unittest.mock import patch

from . import verify_recovery_exploration as original
from .run_support_skill_study import digest, save


class FlatAuditHead(original.AnkleHead):
    def forward(self, x):
        leading = x.shape[:-1]
        value = super().forward(x.reshape(-1, x.shape[-1]))
        return value.reshape(leading) if self.critic else value.reshape(*leading, 1)


def main():
    with patch.object(original, 'AnkleHead', FlatAuditHead):
        original.main()
    save(original.OUT/'audit-harness.json', dict(
        reason='AnkleHead actor expects [world, feature]; flatten audit [time, world, feature] before evaluation.',
        physical_rollouts_repeated=False, sealed_verifier_unchanged=True,
        sources={str(p): digest(p) for p in (Path(__file__), Path(original.__file__))}))


if __name__ == '__main__':
    main()
