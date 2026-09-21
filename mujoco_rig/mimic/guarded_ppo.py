"""Local PPO extension; the external MimicKit checkout stays unmodified."""
import torch

from learning.ppo_agent import PPOAgent
from learning.distribution_gaussian_diag import DistributionGaussianDiag
from util import mp_util, torch_util

from .update_guard import UpdateGuard


class GuardedPPO(PPOAgent):
    """Warm-start fine-tuning with a fixed input mapping and an exact rollout KL bound."""
    def __init__(self, config, env, device):
        if mp_util.get_num_procs() != 1:
            raise ValueError("Guarded PPO currently supports one training process")
        super().__init__(config, env, device)
        self._guard = UpdateGuard(self._actor_optimizer._param_list,
                                  self._actor_optimizer._optimizer, config["actor_max_kl"])

    def _need_normalizer_update(self):
        # Enabled only after loading a validated, trained checkpoint. Do not change
        # the effective policy outside the measured actor update.
        return False

    def _update_actor(self, batch_size, num_steps):
        if self._obs_norm.get_count().item() <= 0:
            raise ValueError("Guarded fine-tuning requires a trained observation normalizer")
        obs = self._exp_buffer.get_data_flat("obs")
        with torch.no_grad():
            norm_obs = self._obs_norm.normalize(obs)
            distribution = self._model.eval_actor(norm_obs)
            behavior = DistributionGaussianDiag(distribution.mean.clone(), distribution.logstd.clone())

        @torch.no_grad()
        def measure_kl():
            return behavior.kl(self._model.eval_actor(norm_obs)).mean().clamp_min(0).item()

        info, accepted, attempted, largest_kl = {}, 0, 0, 0.0
        for _ in range(num_steps):
            batch = self._exp_buffer.sample(batch_size)
            loss_info = self._compute_actor_loss(batch)
            kept, kl = self._guard.step(loss_info["actor_loss"], measure_kl)
            attempted += 1
            largest_kl = max(largest_kl, kl)
            torch_util.add_torch_dict({key: value.detach() for key, value in loss_info.items()}, info)
            if not kept:
                break
            accepted += 1
            self._actor_optimizer._steps += 1
        torch_util.scale_torch_dict(1.0 / attempted, info)
        info.update(actor_kl=measure_kl(), actor_attempted_kl=largest_kl,
                    actor_steps_accepted=accepted, actor_steps_rejected=attempted - accepted)
        return info
