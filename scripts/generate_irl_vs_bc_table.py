import joblib
import numpy as np
from collections import defaultdict

def dd0():
    return defaultdict(list)

def get_stats(d):
    expert = 'norm_halfcheetah_{}_demos_20_subsampling'
    means, stds = [], []
    for n in [4, 8, 16, 32, 64, 128, 256]:
        # print('\n')
        # print(n)
        # print(d[expert.format(n)])
        means.append(np.mean(d[expert.format(n)]))
        stds.append(np.std(d[expert.format(n)]))
    return means, stds

def get_line(name, d):
    line = '\\textbf{' + name + '} '
    means, stds = get_stats(d)
    for m, s in zip(means, stds):
        line += '& %.1f $\\pm$ %.1f ' % (m, s)
    line += r' \\ \hline'
    return line

irl_dirs = [
    '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-state-action/all_returns.pkl',
    '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-final-state-only/all_returns.pkl',
    '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc/all_returns.pkl',
    '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc-state-only/all_returns.pkl',
]
bc_dir = '/scratch/gobi2/kamyar/oorl_rlkit/output/normalized-finally-totally-absolute-final-bc-on-varying-data-amounts/all_returns.pkl'

sa_forw_rets = joblib.load(irl_dirs[2])['all_returns']
s_forw_rets = joblib.load(irl_dirs[3])['all_returns']
sa_rev_rets = joblib.load(irl_dirs[0])['all_returns']
s_rev_rets = joblib.load(irl_dirs[1])['all_returns']

bc_rets = joblib.load(bc_dir)['all_returns']

print(r'\begin{table}[]')
print(r'\begin{tabular}{|l|l|l|l|l|l|l|l|}')
print(r'\hline')
print(r'\textbf{Method} & 4 Demos & 8 Demos & 16 Demos & 32 Demos & 64 Demos & 128 Demos & 256 Demos \\ \hline')


# AIRL s,a
print(get_line('AIRL 8.0', sa_rev_rets[8.0]))
print(get_line('AIRL 12.0', sa_rev_rets[12.0]))

# Forw s,a
print(get_line('Forw KL', sa_forw_rets[50.0]))

# AIRL s
print(get_line('s AIRL 8.0', s_rev_rets[8.0]))
print(get_line('s AIRL 12.0', s_rev_rets[12.0]))

# Forw s
print(get_line('s Forw KL', s_forw_rets[50.0]))

# BC 16
print(get_line('BC 16', bc_rets[16]))
# BC 32
print(get_line('BC 32', bc_rets[32]))
# BC 64
print(get_line('BC 64', bc_rets[64]))


print(r'\end{tabular}')
print(r'\end{table}')
