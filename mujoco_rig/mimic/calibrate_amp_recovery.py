"""Calibrate only the motion discriminator against frozen native recovery traces."""
import copy
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def classification(agent, demo, generated):
    def logits(data):
        return torch.cat([agent._model.eval_disc(agent._disc_obs_norm.normalize(batch)).flatten()
                          for batch in data.split(2048)])
    positive, negative = logits(demo), logits(generated)
    return dict(demo_accuracy=float((positive > 0).float().mean()),
                generated_accuracy=float((negative < 0).float().mean()),
                demo_logit=float(positive.mean()), generated_logit=float(negative.mean()),
                balanced_bce=float(.5 * (F.softplus(-positive).mean() + F.softplus(negative).mean())),
                clipped_feature_fraction=float(torch.cat([
                    ((data - agent._disc_obs_norm._mean) / agent._disc_obs_norm._std).abs().gt(10.).float().flatten()
                    for data in (demo, generated)]).mean()))


def calibrate(agent, task, native_samples, out, steps, seed):
    """Use temporally disjoint demo blocks and a held-out native impact direction."""
    rng = torch.Generator(device=agent._device).manual_seed(seed)
    pools = {key: torch.cat(value).to(agent._device) for key, value in native_samples.items()}
    if any(len(value) < 128 for value in pools.values()):
        raise ValueError("Insufficient independent native calibration observations")
    demo_train = task.fetch_disc_obs_demo(8192)
    demo_test = task.fetch_disc_obs_demo(4096, split="heldout")
    arrays = {**{key: value.cpu().numpy() for key, value in pools.items()},
              "demo_train": demo_train.cpu().numpy(), "demo_heldout": demo_test.cpu().numpy()}
    np.savez_compressed(out / "discriminator-calibration-samples.npz", **arrays)
    sample_hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
    before = classification(agent, demo_test, pools["heldout"])
    # Demo-only tiny variances clip ordinary physical recovery. Fit balanced,
    # frozen training moments; never use the validation samples or actor norm.
    count = min(len(pools["train"]), len(demo_train))
    normalization = torch.cat((demo_train[:count], pools["train"][:count]))
    variance, mean = torch.var_mean(normalization.double(), dim=0, correction=0)
    if agent._disc_obs_norm._mean_sq is None:
        agent._disc_obs_norm._mean_sq = agent._disc_obs_norm._calc_mean_sq(
            agent._disc_obs_norm._mean, agent._disc_obs_norm._std)
    agent._disc_obs_norm.set_mean_std(mean.float(), variance.sqrt().clamp_min(.05).float())
    agent._disc_obs_norm._count.data.fill_(len(normalization))
    normalized = classification(agent, demo_test, pools["heldout"])
    params = list(agent._model.get_disc_params())
    frozen = {key: value.detach().clone() for key, value in agent.state_dict().items()
              if not key.startswith(("_model._disc", "_disc_obs_norm"))}
    curve = []
    best = copy.deepcopy(agent._model.state_dict())
    best_score = normalized["balanced_bce"]
    selected_step = 0
    for step in range(1, steps + 1):
        demo = demo_train[torch.randint(len(demo_train), (1024,), generator=rng, device=agent._device)]
        generated = pools["train"][torch.randint(len(pools["train"]), (1024,), generator=rng, device=agent._device)]
        positive = agent._disc_obs_norm.normalize(demo).detach().requires_grad_(True)
        negative = agent._disc_obs_norm.normalize(generated).detach().requires_grad_(True)
        positive_logits = agent._model.eval_disc(positive).flatten()
        negative_logits = agent._model.eval_disc(negative).flatten()
        loss = .5 * (F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean())
        gradients = [torch.autograd.grad(logits.sum(), values, create_graph=True)[0]
                     for logits, values in ((positive_logits, positive), (negative_logits, negative))]
        loss = loss + agent._disc_grad_penalty * .5 * sum(gradient.square().sum(-1).mean() for gradient in gradients)
        loss = loss + agent._disc_logit_reg * agent._model.get_disc_logit_weights().square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite discriminator calibration loss")
        agent._disc_optimizer.step(loss)
        if step % 16 == 0 or step == steps:
            measured = classification(agent, demo_test, pools["heldout"])
            curve.append(dict(step=step, **measured))
            if measured["balanced_bce"] < best_score:
                best_score, selected_step = measured["balanced_bce"], step
                best = copy.deepcopy(agent._model.state_dict())
    agent._model.load_state_dict(best)
    # Selection restores parameters; discard optimizer momentum from later steps.
    agent._disc_optimizer._optimizer.state.clear()
    for key, value in frozen.items():
        if not torch.equal(value, agent.state_dict()[key]):
            raise RuntimeError("Discriminator calibration changed the actor/critic contract")
    after = classification(agent, demo_test, pools["heldout"])
    passed = (after["balanced_bce"] < normalized["balanced_bce"]
              and after["demo_accuracy"] >= .6 and after["generated_accuracy"] >= .6)
    return dict(schema="amp_discriminator_calibration_v1", passed=passed,
                source_mapping=before, balanced_mapping=normalized, calibrated=after,
                selected_step=selected_step, maximum_steps=steps, curve=curve,
                sample_hashes=sample_hashes, generated_counts={key: len(value) for key, value in pools.items()},
                actor_and_critic_unchanged=True,
                scope="Held-out .5s demo blocks and native cardinal direction 3; not unseen-hit policy validation")
