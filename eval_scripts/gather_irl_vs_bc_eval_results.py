import pprint
import joblib
import numpy as np
import os.path as osp
from collections import defaultdict

all_dict_paths = [
    '/scratch/hdd001/home/kamyar/output/paper-version-fairl-4-ant-demos-correct-final-with-saving',
    '/scratch/hdd001/home/kamyar/output/paper-version-ant-bc',
    '/scratch/hdd001/home/kamyar/output/paper-version-fairl-4-ant-demos-correct-final-with-saving',
    '/scratch/hdd001/home/kamyar/output/paper-version-fairl-4-hopper-demos-correct-final-with-saving',
    '/scratch/hdd001/home/kamyar/output/paper-version-fairl-4-walker-demos-correct-final-with-saving',
    '/scratch/hdd001/home/kamyar/output/paper-version-hc-bc',
    '/scratch/hdd001/home/kamyar/output/paper-version-walker-bc',



    # '/scratch/hdd001/home/kamyar/output/paper-version-hopper-bc-rerun',
]

'''
exp name
    det / stoch
        setting
            seeds
'''

bc_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
airl_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
fairl_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for p in all_dict_paths:
    d = joblib.load(osp.join(p, 'all_eval_stats.pkl'))
    det_rets = d['all_det_returns']
    stoch_rets = d['all_stoch_returns']
    
    k = list(det_rets.keys())[0]

    # print(k[-1])
    # print(type(k[-1]))
    if type(k[-1]) is float:
        # print('float')
        print(k)
        # continue
        model_dict = bc_dict
    else:
        print(k)
        if k[-1]:
            continue

            print('shdskdh')
            model_dict = fairl_dict
        else:
            continue
            print('kdshksdhkds')
            model_dict = airl_dict

        # model_dict = airl_dict
    
    # if k[-1] == 0.0:
    #     model_dict = bc_dict
    # elif k[-1]:
    #     model_dict = fairl_dict
    # else:
    #     model_dict = airl_dict
    
    for k in det_rets:
        exp_name = k[0]
        rew = k[1]
        gp = k[2]
        model_dict[exp_name][True][(rew, gp)].append(det_rets[k])
    
    for k in stoch_rets:
        exp_name = k[0]
        rew = k[1]
        gp = k[2]
        model_dict[exp_name][False][(rew, gp)].append(stoch_rets[k])



# print(airl_dict)
# 1/0

# pp = pprint.PrettyPrinter(indent=0)
# # pp.pprint(fairl_dict)
# pp.pprint(bc_dict)

# now filter
'''
exp name
    det / stoch
        [mean, std]
'''

print('FAIRL')
print('TRUE IS DETERMINISTIC EVAL')
print('SHOWING MEAN AND STD')

# new_dict = defaultdict(defaultdict)
# for exp_name in airl_dict:
#     best_ret = float('-Inf')
#     d = airl_dict[exp_name][True]
#     for setting in d:
#         ret = np.mean(d[setting])
#         if ret > best_ret:
#             best_setting = setting
#             best_ret = ret
    
#     for det in [True, False]:
#         new_dict[exp_name][det] = [
#             np.mean(airl_dict[exp_name][det][best_setting]),
#             np.std(airl_dict[exp_name][det][best_setting])
#         ]
# airl_dict = new_dict

# pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(airl_dict)

# new_dict = defaultdict(defaultdict)
# for exp_name in fairl_dict:
#     best_ret = float('-Inf')
#     d = fairl_dict[exp_name][True]
#     for setting in d:
#         ret = np.mean(d[setting])
#         if ret > best_ret:
#             best_setting = setting
#             best_ret = ret
    
#     for det in [True, False]:
#         new_dict[exp_name][det] = [
#             np.mean(fairl_dict[exp_name][det][best_setting]),
#             np.std(fairl_dict[exp_name][det][best_setting])
#         ]
# fairl_dict = new_dict
# pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(fairl_dict)


for exp_name in bc_dict:
    print(exp_name)
    print(True)
    print('{} +/- {}'.format(
        np.mean(
            bc_dict[exp_name][True][(-1.0,-1.0)]
        ),
        np.std(
            bc_dict[exp_name][True][(-1.0,-1.0)]
        )
    ))
    print(False)
    print('{} +/- {}'.format(
        np.mean(bc_dict[exp_name][False][(-1.0,-1.0)]),
        np.std(bc_dict[exp_name][False][(-1.0,-1.0)])
    ))
    print('\n')
