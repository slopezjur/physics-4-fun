"""On-demand source prefixes for continuation-only scalar ankle PPO."""
import torch

from .recovery_start_bank import RecoveryStart
from .train_coordinated_ankle import AnkleRecovery, ONSET_FRAMES


class RecoveryStartTraining(AnkleRecovery):
    def __init__(self, rig, anchor):
        self.source = None
        self.training_started = False
        self.collected = 0
        self.prefix_intervals = 0
        self.exact_transfers = 0
        super().__init__(rig, anchor)
        self.source = AnkleRecovery(rig, anchor)
        self.engine.policy_control.set_reference(*self.task.sample_reference(self.task.offset))

    @torch.no_grad()
    def reset(self, ids=None, rng=None):
        if rng is None:
            return super().reset(ids)
        ids = self.task.ids[self.task.ids % 8 < 6] if ids is None else ids
        assert len(ids) and (ids % 8 < 6).all()
        n = len(ids)
        sign = 2*torch.randint(2, (n,), generator=rng)-1
        quiet = torch.rand(n, generator=rng) < .25
        onset = ONSET_FRAMES[torch.randint(len(ONSET_FRAMES), (n,), generator=rng)]
        source = self.source
        source.reset()
        source.force.zero_()
        source.force[ids, 0] = 150.*sign*(~quiet)
        source.onset[ids] = onset
        for frame in range(int(onset.max())+1):
            active = torch.zeros(16, dtype=torch.bool)
            active[ids] = frame <= onset
            _, terminal, info = source.step(torch.zeros(16, 1), active)
            assert not terminal[active].any() and not info['warning'].any()
            self.prefix_intervals += int(active.sum())
            finished = ids[onset == frame]
            if len(finished):
                snapshot = RecoveryStart.capture(source)
                snapshot.restore(self, finished)
                assert torch.equal(self.features()[finished], source.features()[finished])
                assert (self.task.steps[finished] == self.onset[finished]+1).all()
                assert not self.previous[finished].any() and not self.last_latent[finished].any()
                self.exact_transfers += len(finished)
                self.reset_records.extend(dict(world=int(i), pose=int(i % 8),
                    force_n=self.force[i].tolist(), onset_frame=int(self.onset[i]),
                    start_frame=int(self.task.steps[i])) for i in finished)
                del snapshot
        self.training_started = True

    @torch.no_grad()
    def step(self, latent, active):
        if self.training_started:
            assert (self.task.ids[active] % 8 < 6).all()
            assert (self.task.steps[active] >= self.onset[active]+1).all()
            self.collected += int(active.sum())
        return super().step(latent, active)
