"""Count every rollout transition without per-step CPU synchronization."""
import torch


class RecoverySignals:
    KEYS = ('world_steps','post_hit_world_steps','credited_events','measured_placements',
            'rejected_support_or_initial_risk','rejected_capture_progress','rejected_articulation',
            'rejected_landing_load','endpoint_missed_peak','post_hit_slip_proxy_sum_m_s')

    def __init__(self, device):
        self.totals = torch.zeros(len(self.KEYS),device=device,dtype=torch.float64)

    def record(self, hit, diagnostics, slip_speed):
        mask = hit[:,None]
        values = [torch.ones_like(hit).sum(),hit.sum(),(diagnostics['useful_events']*hit).sum()]
        values += [(diagnostics[key]*mask).sum() for key in self.KEYS[3:9]]
        values.append((slip_speed*hit).sum())
        self.totals += torch.stack(values)

    def drain(self):
        values = dict(zip(self.KEYS,self.totals.cpu().tolist()))
        self.totals.zero_()
        hit = values['post_hit_world_steps']
        values['mean_post_hit_slip_proxy_m_s'] = values['post_hit_slip_proxy_sum_m_s']/hit if hit else 0.
        values['credited_events_per_1000_post_hit_steps'] = 1000*values['credited_events']/hit if hit else 0.
        values['scope'] = 'Every control transition since last drain; rejection reasons overlap; reset retains window totals'
        return values
