import joblib
from scipy.misc import imsave

d = joblib.load(
    '/ais/gobi6/kamyar/oorl_rlkit/output/'
    + 'gen-simple-meta-reacher-expert-trajs/gen_simple_meta_reacher_expert_trajs_2018_10_25_19_23_47_0000--s-0/extra_data.pkl')

print(d.keys())
print(d['replay_buffer']._observations.keys())

for i in range(1000):
    imsave('plots/test_pixels/%d.png'%i, d['replay_buffer']._observations['pixels'][i])
