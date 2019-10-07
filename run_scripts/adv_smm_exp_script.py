import yaml
import argparse
import joblib
import numpy as np

import torch

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.irl.disc_models.simple_disc_models import MLPDisc, ResNetAIRLDisc
from rlkit.torch.state_marginal_matching.adv_smm import AdvSMM
from rlkit.envs.wrappers import ScaledEnv


def experiment(variant):
    with open('expert_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    buffer_save_dict = joblib.load(demos_path)
    target_state_buffer = buffer_save_dict['data']
    state_indices = torch.LongTensor(variant['state_indices'])

    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])
    training_env = get_env(env_specs)
    training_env.seed(env_specs['training_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))
    
    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1
    
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # build the discriminator model
    if variant['disc_model_type'] == 'resnet_disc':
        disc_model = ResNetAIRLDisc(
            len(state_indices),
            num_layer_blocks=variant['disc_num_blocks'],
            hid_dim=variant['disc_hid_dim'],
            hid_act=variant['disc_hid_act'],
            use_bn=variant['disc_use_bn'],
            clamp_magnitude=variant['disc_clamp_magnitude']
        )
    else:
        disc_model = MLPDisc(
            len(state_indices),
            num_layer_blocks=variant['disc_num_blocks'],
            hid_dim=variant['disc_hid_dim'],
            hid_act=variant['disc_hid_act'],
            use_bn=variant['disc_use_bn'],
            clamp_magnitude=variant['disc_clamp_magnitude']
        )
    
    # set up the algorithm
    trainer = SoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['sac_params']
    )
    algorithm = AdvSMM(
        env=env,
        training_env=training_env,
        exploration_policy=policy,

        discriminator=disc_model,
        policy_trainer=trainer,

        target_state_buffer=target_state_buffer,
        state_indices=state_indices,
        **variant['adv_smm_params']
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
