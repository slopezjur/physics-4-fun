"""CPU standing gate shared by fine-tuning, chaining and promotion."""
import math


def quiet_rejection(survived, recovered):
    if not all(math.isfinite(v) for v in (survived, recovered)):
        return "missing finite quiet-room results"
    if survived < 100.0 or recovered < 100.0:
        return f"quiet-room gate failed: {survived:.1f}% survived, {recovered:.1f}% settled"
    return None


def validate_checkpoint(path):
    """A seed must stand and settle for 40 seconds before spending GPU time on it."""
    import torch
    from eval import rollout
    from ppo import ActorCritic
    from observation_contract import checkpoint_version

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(checkpoint['num_obs'], checkpoint['num_actions'])
    net.load_state_dict(checkpoint['model'])
    net.eval()
    result = rollout(net, 16, 40.0, 17, ball=False, label="SEED VALIDATION, quiet room",
                     observation_version=checkpoint_version(checkpoint))
    reason = quiet_rejection(result['survived'], result['recovered'])
    if reason:
        raise ValueError(f"Refusing fine-tuning seed {path}: {reason}")
    return result
