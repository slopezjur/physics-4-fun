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
                                  self._actor_optimizer._optimizer, config["actor_max_kl"],
                                  config.get("actor_backtracks", 0))
        self.quiet_anchor = None
        self._project_anchor = config.get("actor_project_anchor", False)

    def _need_normalizer_update(self):
        # Enabled only after loading a validated, trained checkpoint. Do not change
        # the effective policy outside the measured actor update.
        return False

    def _update_actor(self, batch_size, num_steps):
        if self._obs_norm.get_count().item() <= 0:
            raise ValueError("Guarded fine-tuning requires a trained observation normalizer")
        if self._project_anchor and self.quiet_anchor is None:
            raise ValueError("Anchor projection requires an installed fixed teacher")
        obs = self._exp_buffer.get_data_flat("obs")
        with torch.no_grad():
            norm_obs = self._obs_norm.normalize(obs)
            distribution = self._model.eval_actor(norm_obs)
            behavior = DistributionGaussianDiag(distribution.mean.clone(), distribution.logstd.clone())

        @torch.no_grad()
        def measure_kl():
            return behavior.kl(self._model.eval_actor(norm_obs)).mean().clamp_min(0).item()

        info, accepted, attempted, largest_kl = {}, 0, 0, 0.0
        search = dict(actor_search_trials=0., actor_backtracked_steps=0., actor_step_scale_sum=0.,
                      actor_search_loss_decrease=0., actor_constraint_rejections=0., actor_loss_rejections=0.,
                      actor_projected_steps=0., actor_projected_trials=0.)
        for _ in range(num_steps):
            batch = self._exp_buffer.sample(batch_size)
            loss_info = self._compute_actor_loss(batch)
            kept, kl = self._guard.step(loss_info["actor_loss"], measure_kl,
                accept=(lambda: self.quiet_anchor.accepts(self)) if self.quiet_anchor is not None else None,
                measure_loss=(lambda: self._compute_actor_loss(batch)["actor_loss"]) if self._guard.max_backtracks else None,
                project_direction=(lambda d: self.quiet_anchor.project_direction(self, self._guard.parameters, d))
                    if self._project_anchor and self.quiet_anchor is not None else None)
            if self._guard.max_backtracks:
                result = self._guard.last_search
                search["actor_search_trials"] += result["trials"]
                search["actor_backtracked_steps"] += kept and result["scale"] < 1
                search["actor_step_scale_sum"] += result["scale"]
                search["actor_search_loss_decrease"] += result["loss_decrease"]
                search["actor_constraint_rejections"] += result["constraint_rejections"]
                search["actor_loss_rejections"] += result["loss_rejections"]
                search["actor_projected_steps"] += result["projected"]
                search["actor_projected_trials"] += result["projected_trials"]
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
        if self.quiet_anchor is not None:
            info["quiet_anchor_max_kl"] = self.quiet_anchor.measure(self)
        if self._guard.max_backtracks:
            info.update(search)
        return info
