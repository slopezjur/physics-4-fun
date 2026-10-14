"""Map a legacy 30-motor torque actor into the versioned neck-enabled policy."""
import torch


def migrate_state(source, destination, old_contract, new_contract):
    """Preserve existing outputs; new neck means and input columns start at zero."""
    old_names, new_names = old_contract['actuator_names'], new_contract['actuator_names']
    if len(old_names) != 30 or len(new_names) != 33 or set(new_names)-set(old_names) != {'Head_rx', 'Head_ry', 'Head_rz'}:
        raise ValueError('Only the explicit 30-to-33 neck extension can be migrated')
    actions = torch.tensor([new_names.index(name) for name in old_names], dtype=torch.long)
    new_layout = {row['name']: row for row in new_contract['observation_layout']}
    observation = []
    for old in old_contract['observation_layout']:
        new = new_layout[old['name']]
        if old['name'] == 'pending_actions_oldest_first':
            for slot in range(2):
                observation.extend((new['offset']+slot*33+actions).tolist())
        elif old['width'] == new['width']:
            observation.extend(range(new['offset'], new['offset']+new['width']))
        else:
            raise ValueError(f"Unsupported observation migration: {old['name']}")
    inputs = torch.tensor(observation, dtype=torch.long)
    if source.keys() != destination.keys() or len(inputs) != 300 or new_contract['num_obs'] != 306:
        raise ValueError('Unreviewed model or observation extension')
    result = {}
    for name, template in destination.items():
        value = source[name].to(template.device)
        if value.shape == template.shape:
            result[name] = value
            continue
        mapped = template.clone()
        if name.startswith('_obs_norm.') and value.shape == (300,) and template.shape == (306,):
            mapped[inputs.to(mapped.device)] = value
        elif name.startswith('_a_norm.') and value.shape == (30,) and template.shape == (33,):
            mapped[actions.to(mapped.device)] = value
        elif name.startswith(('_model._actor_layers.', '_model._critic_layers.')) and value.ndim == 2 \
                and value.shape[1] == 300 and template.shape == (value.shape[0], 306):
            mapped.zero_()
            mapped[:, inputs.to(mapped.device)] = value
        elif name.startswith('_model._action_dist.') and value.shape[0] == 30 \
                and template.shape == (33, *value.shape[1:]):
            if '_mean_net.' in name:
                mapped.zero_()
            mapped[actions.to(mapped.device)] = value
        else:
            raise ValueError(f'Unreviewed checkpoint tensor migration: {name}')
        result[name] = mapped
    return result
