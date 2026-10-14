"""Match shared actor/critic initialization when changing residual action width."""
from unittest.mock import patch
import torch

from . import support_skill_learning as learning


original=learning.build


def matched(config,env,initialize=None):
    if not initialize:raise ValueError('Action-width comparison needs a saved common initialization')
    agent=original(config,env,None)
    old=torch.load(initialize,map_location=env.device,weights_only=True)
    state=agent.state_dict()
    for key,value in old.items():
        if key in state and state[key].shape==value.shape:state[key]=value
    agent.load_state_dict(state)
    with torch.no_grad():
        for p in agent._model._action_dist._mean_net.parameters():p.zero_()
    from .policy import StandPolicy
    assert not StandPolicy(agent)(env.observations()).any()
    assert all(torch.equal(agent.state_dict()[k],v) for k,v in old.items()
        if k in state and state[k].shape==v.shape and '_mean_net' not in k)
    agent._sync_optimizer();agent._init_train()
    agent._curr_obs,agent._curr_info=agent._reset_envs()
    return agent


if __name__=='__main__':
    with patch.object(learning,'build',matched):learning.main()
