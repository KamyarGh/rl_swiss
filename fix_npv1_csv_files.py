import os
from collections import defaultdict
import numpy as np

exp_path = 'output/npv1-trans-regr'
# exp_path = 'output/npv1-trans-regr/npv1_trans_regr_2018_08_13_06_23_35_0001--s-0/debug.log'
ignore_str = '-----'

for sub_exp_dir in os.listdir(exp_path):
    sub_exp_path = os.path.join(exp_path, sub_exp_dir)
    if not os.path.isdir(sub_exp_path): continue
    try:
        with open(os.path.join(sub_exp_path, 'debug.log'), 'r') as f:
            value_dict = defaultdict(list)
            for line in f:
                line = line.split()
                if ignore_str not in line[0]:
                    k, v = line[0], float(line[1])
                    value_dict[k].append(v)
            
            # find the shortest list length
            min_len = min(map(lambda k: len(value_dict[k]), value_dict))
            new_dict = {k: v[:min_len] for k,v in value_dict.items()}

            # convert to numpy array
            arr = np.zeros((min_len, len(new_dict.keys())))
            for i, k in enumerate(sorted(new_dict.keys())):
                v = np.array(new_dict[k])
                arr[:,i] = v
            header = ','.join(sorted(new_dict.keys()))
            
            np.savetxt(os.path.join(sub_exp_path, 'progress.csv'), arr, delimiter=",", header=header)
            # print(os.path.join(sub_exp_path, 'progress.csv'))
    except:
        print('failed ', sub_exp_path)
