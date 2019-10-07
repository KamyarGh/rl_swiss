import numpy as np
import joblib
from os import path as osp
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu
from rlkit.envs import get_env
from rlkit.core import eval_util
from rlkit.samplers import PathSampler
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.envs.wrappers import ScaledEnv


env_specs = {'env_name': 'halfcheetah', 'env_kwargs': {}, 'eval_env_seed': 3562}
env = get_env(env_specs)
env.seed(env_specs['eval_env_seed'])
with open('expert_demos_listing.yaml', 'r') as f:
    listings = yaml.load(f.read())
    expert_demos_path = listings['norm_halfcheetah_32_demos_sub_20']['file_paths'][0]
    buffer_save_dict = joblib.load(expert_demos_path)
    env = ScaledEnv(
        env,
        obs_mean=buffer_save_dict['obs_mean'],
        obs_std=buffer_save_dict['obs_std'],
        acts_mean=buffer_save_dict['acts_mean'],
        acts_std=buffer_save_dict['acts_std'],
    )

bc_policy = joblib.load('/scratch/hdd001/home/kamyar/output/paper-version-hc-bc/paper_version_hc_bc_2019_05_19_00_32_05_0000--s-0/params.pkl')['exploration_policy']
bc_policy = MakeDeterministic(bc_policy)
bc_policy.to(ptu.device)

dagger_policy = joblib.load('/scratch/hdd001/home/kamyar/output/dagger-halfcheetah/dagger_halfcheetah_2019_08_20_16_30_36_0000--s-0/params.pkl')['exploration_policy']
dagger_policy = MakeDeterministic(dagger_policy)
dagger_policy.to(ptu.device)

irl_policy = joblib.load('/scratch/hdd001/home/kamyar/output/hc_airl_ckpt/params.pkl')['exploration_policy']
irl_policy = MakeDeterministic(irl_policy)
irl_policy.to(ptu.device)

fig, ax = plt.subplots(1)

eval_sampler = PathSampler(
    env,
    bc_policy,
    20000,
    1000,
    no_terminal=False,
    render=False,
    render_kwargs={}
)
bc_paths = eval_sampler.obtain_samples()
run_rews = [[t['reward_run'] for t in p['env_info']] for p in bc_paths]
for rr in run_rews:
    rr = np.array(rr)
    rr = np.cumsum(rr)
    ax.plot(np.arange(rr.shape[0]), rr, color='tomato')

eval_sampler = PathSampler(
    env,
    dagger_policy,
    20000,
    1000,
    no_terminal=False,
    render=False,
    render_kwargs={}
)
dagger_paths = eval_sampler.obtain_samples()
run_rews = [[t['reward_run'] for t in p['env_info']] for p in dagger_paths]
for rr in run_rews:
    rr = np.array(rr)
    rr = np.cumsum(rr)
    ax.plot(np.arange(rr.shape[0]), rr, color='purple')

eval_sampler = PathSampler(
    env,
    irl_policy,
    20000,
    1000,
    no_terminal=False,
    render=False,
    render_kwargs={}
)
irl_paths = eval_sampler.obtain_samples()
run_rews = [[t['reward_run'] for t in p['env_info']] for p in irl_paths]
for rr in run_rews:
    rr = np.array(rr)
    rr = np.cumsum(rr)
    ax.plot(np.arange(rr.shape[0]), rr, color='royalblue')

plt.savefig('plots/paper_results/cov_shift.png', bbox_inches='tight')
plt.close()
