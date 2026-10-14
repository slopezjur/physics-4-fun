"""Single actor-rate contrast over the sealed projected-and-retracted PPO learner."""
import json
from unittest.mock import patch

import torch

from . import quiet_constrained_ppo as base
from . import retracted_ppo as retracted
from .run_support_skill_study import save

ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 3e-4
CONSTRUCTOR_ACTOR_RATE = 1e-4
NOMINAL_ACTOR_SLOTS = 288
HALF_SIGMA = base.HALF_SIGMA


def learn(env, initial, folder, seed, sigma, quiet_features):
    """Set only fresh actor Adam's effective rate before its first recorded attempt."""
    original_retracted_step = retracted.retracted_step
    original_constrained_step = base.constrained_step
    actor_optimizer = None
    attempts = 0

    def rate_step(actor, optimizer, *args, **kwargs):
        nonlocal actor_optimizer, attempts
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 1
        actor_parameters = tuple(actor.parameters())
        optimizer_parameters = tuple(optimizer.param_groups[0]['params'])
        assert len(actor_parameters) == len(optimizer_parameters)
        assert all(left is right for left, right in zip(actor_parameters, optimizer_parameters))
        if actor_optimizer is None:
            assert not optimizer.state
            assert optimizer.param_groups[0]['lr'] == optimizer.defaults['lr'] == CONSTRUCTOR_ACTOR_RATE
            optimizer.param_groups[0]['lr'] = ACTOR_LEARNING_RATE
            optimizer.defaults['lr'] = ACTOR_LEARNING_RATE
            actor_optimizer = optimizer
        else:
            assert optimizer is actor_optimizer
        assert optimizer.param_groups[0]['lr'] == optimizer.defaults['lr'] == ACTOR_LEARNING_RATE
        # The effective rate is fixed before the sealed step records any actor/Adam hash.
        record = original_retracted_step(actor, optimizer, *args, **kwargs)
        assert optimizer.param_groups[0]['lr'] == optimizer.defaults['lr'] == ACTOR_LEARNING_RATE
        record['actor_learning_rate'] = ACTOR_LEARNING_RATE
        attempts += 1
        return record

    # Both this patch and the sealed learner's inner patch restore on exceptions.
    with patch.object(retracted, 'retracted_step', rate_step):
        actor = retracted.learn(env, initial, folder, seed, sigma, quiet_features)
    assert retracted.retracted_step is original_retracted_step
    assert base.constrained_step is original_constrained_step
    assert actor_optimizer is not None and 32 <= attempts <= NOMINAL_ACTOR_SLOTS
    history = json.loads((folder/'history.json').read_text(encoding='utf-8'))
    assert attempts == sum(row['actor_updates']+row['rolled_back'] for row in history)
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    report.update(actor_learning_rate=ACTOR_LEARNING_RATE, critic_learning_rate=CRITIC_LEARNING_RATE,
        nominal_actor_slots=NOMINAL_ACTOR_SLOTS, actor_rate_attempts=attempts,
        initial_constructor_actor_rate=CONSTRUCTOR_ACTOR_RATE,
        effective_rate_set_before_first_update=True, actor_optimizer_initial_state_empty=True,
        actor_rate_constant_across_attempts=True, global_adam_constructor_patched=False,
        unchanged_learning_loop='retracted_ppo.learn -> quiet_constrained_ppo.learn')
    save(folder/'report.json', report)
    return actor
