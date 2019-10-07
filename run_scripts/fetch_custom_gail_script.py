import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ObsPreprocessedReparamTanhMultivariateGaussianPolicy
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.irl.gail import GAIL
from rlkit.torch.irl.gail_with_traj_batches import GAILWithTrajBatches
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.irl.disc_models.gail_disc import Model as GAILDiscModel
from rlkit.torch.irl.disc_models.gail_disc import MlpGAILDisc
from rlkit.torch.irl.disc_models.gail_disc import ResnetDisc
from rlkit.torch.irl.disc_models.gail_disc import SingleColorFetchCustomDisc, \
    SecondVersionSingleColorFetchCustomDisc, ThirdVersionSingleColorFetchCustomDisc
from rlkit.envs.wrapped_absorbing_env import WrappedAbsorbingEnv

from rlkit.envs import get_env

import torch

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'



class ObsPreprocessedQFunc(FlattenMlp):
    '''
        This is a weird thing and I didn't know what to call.
        Basically I wanted this so that if you need to preprocess
        your inputs somehow (attention, gating, etc.) with an external module
        before passing to the policy you could do so.
        Assumption is that you do not want to update the parameters of the preprocessing
        module so its output is always detached.
    '''
    def __init__(self, preprocess_model, *args, wrap_absorbing=False, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        self.preprocess_model_list = [preprocess_model]
        self.wrap_absorbing = wrap_absorbing
    

    @property
    def preprocess_model(self):
        # this is a hack so that it is not added as a submodule
        return self.preprocess_model_list[0]


    def preprocess_fn(self, obs_batch):
        mode = self.preprocess_model.training
        self.preprocess_model.eval()
        processed_obs_batch = self.preprocess_model(obs_batch, self.wrap_absorbing).detach()
        self.preprocess_model.train(mode)
        return processed_obs_batch
    

    def forward(self, obs, actions):
        obs = self.preprocess_fn(obs).detach()
        return super().forward(obs, actions)


class ObsPreprocessedVFunc(FlattenMlp):
    '''
        This is a weird thing and I didn't know what to call.
        Basically I wanted this so that if you need to preprocess
        your inputs somehow (attention, gating, etc.) with an external module
        before passing to the policy you could do so.
        Assumption is that you do not want to update the parameters of the preprocessing
        module so its output is always detached.
    '''
    def __init__(self, preprocess_model, *args, wrap_absorbing=False, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        self.preprocess_model_list = [preprocess_model]
        self.wrap_absorbing = wrap_absorbing
    

    @property
    def preprocess_model(self):
        # this is a hack so that it is not added as a submodule
        return self.preprocess_model_list[0]


    def preprocess_fn(self, obs_batch):
        mode = self.preprocess_model.training
        self.preprocess_model.eval()
        processed_obs_batch = self.preprocess_model(obs_batch, self.wrap_absorbing).detach()
        self.preprocess_model.train(mode)
        return processed_obs_batch
    

    def forward(self, obs):
        obs = self.preprocess_fn(obs).detach()
        return super().forward(obs)


def experiment(variant):
    # NEW WAY OF DOING EXPERT REPLAY BUFFERS USING ExpertReplayBuffer class
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    print(listings.keys())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    expert_replay_buffer = joblib.load(file_to_load)['replay_buffer']
    # this script is for the non-meta-learning GAIL
    expert_replay_buffer.policy_uses_task_params = variant['gail_params']['policy_uses_task_params']
    expert_replay_buffer.concat_task_params_to_policy_obs = variant['gail_params']['concat_task_params_to_policy_obs']

    # Now determine how many trajectories you want to use
    if 'num_expert_trajs' in variant: raise NotImplementedError('Not implemented during the transition away from ExpertReplayBuffer')
    
    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    if env_specs['train_test_env']:
        env, training_env = get_env(env_specs)
    else:
        env, _ = get_env(env_specs)
        training_env, _ = get_env(env_specs)
    env.seed(variant['seed'])
    training_env.seed(variant['seed'])

    # if variant['wrap_absorbing_state']:
    #     assert False, 'Not handling train_test_env'
    #     env = WrappedAbsorbingEnv(env)

    print(env.observation_space)

    if isinstance(env.observation_space, Dict):
        if not variant['gail_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(env.observation_space.spaces['obs'].shape))
            if variant['gail_params']['policy_uses_task_params']:
                if variant['gail_params']['concat_task_params_to_policy_obs']:
                    obs_dim += int(np.prod(env.observation_space.spaces['obs_task_params'].shape))
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    print(obs_dim, action_dim)
    sleep(3)

    if variant['gail_params']['state_only']: print('\n\nUSING STATE ONLY DISC\n\n')
    disc_model = ThirdVersionSingleColorFetchCustomDisc(
        clamp_magnitude=variant['disc_clamp_magnitude'],
        state_only=variant['gail_params']['state_only'],
        wrap_absorbing=variant['gail_params']['wrap_absorbing']
    )
    if variant['gail_params']['use_target_disc']:
        target_disc = disc_model.copy()
    else:
        target_disc = None
    print(disc_model)
    print(disc_model.clamp_magnitude)

    policy_net_size = variant['policy_net_size']
    hidden_sizes = [policy_net_size] * variant['num_hidden_layers']
    qf1 = ObsPreprocessedQFunc(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        hidden_sizes=hidden_sizes,
        input_size=6 + 4 + 4 + 1*variant['gail_params']['wrap_absorbing'],
        output_size=1,
        wrap_absorbing=variant['gail_params']['wrap_absorbing']
    )
    qf2 = ObsPreprocessedQFunc(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        hidden_sizes=hidden_sizes,
        input_size=6 + 4 + 4 + 1*variant['gail_params']['wrap_absorbing'],
        output_size=1,
        wrap_absorbing=variant['gail_params']['wrap_absorbing']
    )
    vf = ObsPreprocessedVFunc(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        hidden_sizes=hidden_sizes,
        input_size=6 + 4 + 1*variant['gail_params']['wrap_absorbing'],
        output_size=1,
        wrap_absorbing=variant['gail_params']['wrap_absorbing']
    )
    policy = ObsPreprocessedReparamTanhMultivariateGaussianPolicy(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        hidden_sizes=hidden_sizes,
        obs_dim=6 + 4,
        action_dim=4,
    )

    policy_optimizer = NewSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        wrap_absorbing=variant['gail_params']['wrap_absorbing'],
        **variant['policy_params']
    )
    algorithm = GAIL(
        env,
        policy,
        disc_model,
        policy_optimizer,
        expert_replay_buffer,
        training_env=training_env,
        target_disc=target_disc,
        **variant['gail_params']
    )
    print(algorithm.use_target_disc)
    print(algorithm.soft_target_disc_tau)
    print(algorithm.exploration_policy)
    print(algorithm.eval_policy)
    print(algorithm.policy_optimizer.policy_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.qf1_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.qf2_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.vf_optimizer.defaults['lr'])
    print(algorithm.disc_optimizer.defaults['lr'])

    if variant['gail_params']['wrap_absorbing']:
        print('\n\nWRAP ABSORBING\n\n')
    
    # assert False, "Have not added new sac yet!"
    if ptu.gpu_enabled():
        algorithm.cuda()
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
    
    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
