import multiprocessing
from multiprocessing import Pool, Process
import yaml
import argparse
import psutil
import os
from queue import Queue
from time import sleep
from rlkit.launchers.launcher_util import setup_logger, build_nested_variant_generator

# from exp_pool_fns.neural_process_v1 import exp_fn


def get_pool_function(exp_fn_name):
    if exp_fn_name == 'neural_processes_v1':
        from exp_pool_fns.neural_process_v1 import exp_fn
    elif exp_fn_name == 'sac':
        from exp_pool_fns.sac import exp_fn
    
    return exp_fn


if __name__ == '__main__':
    # # Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--experiment', help='experiment specification file')
    # args = parser.parse_args()
    # with open(args.experiment, 'r') as spec_file:
    #     spec_string = spec_file.read()
    #     exp_specs = yaml.load(spec_string)

    # # generating the variants
    # vg_fn = build_nested_variant_generator(exp_specs)
    # all_exp_args = []
    # for i, variant in enumerate(vg_fn()):
    #     all_exp_args.append([variant, i])
    
    # # setting up pool and cpu affinity
    # num_total = len(all_exp_args)
    # num_workers = min(exp_specs['meta_data']['num_workers'], num_total)

    # cpu_range = exp_specs['meta_data']['cpu_range']
    # num_available_cpus = cpu_range[1] - cpu_range[0] + 1
    # num_cpu_per_worker = exp_specs['meta_data']['num_cpu_per_worker']
    # assert  num_cpu_per_worker * num_workers <= num_available_cpus

    # affinity_Q = Queue()
    # for i in range(int(num_available_cpus / num_cpu_per_worker)):
    #     affinity_Q.put(
    #         ','.join(
    #             map(
    #                 str,
    #                 [
    #                     cpu_range[0] + num_cpu_per_worker * i + j
    #                     for j in range(num_cpu_per_worker)
    #                 ]
    #             )
    #         )
    #     )

    # pool_function = get_pool_function(exp_specs['meta_data']['exp_fn_name'])
    # running_process = {}
    # args_idx = 0
    # while (args_idx < len(all_exp_args)) or (len(running_process) > 0):
    #     if len(running_process) < num_workers:
    #         aff = affinity_Q.get()
    #         p = Process(target=pool_function, args=([all_exp_args[args_idx]]))
    #         args_idx += 1
    #         p.start()
    #         os.system("taskset -p -c %s %d" % (aff, p.pid))
    #         running_process[p] = aff

    #     new_running_process = {}
    #     for p, aff in running_process.items():
    #         N = p.join()
    #         if N is None:
    #             new_running_process[p] = aff
    #         else:
    #             del new_running_process[p]
    #             affinity_Q.put(aff)
    #     running_process = new_running_process
        
    #     print(running_process)
    #     sleep(2)

    # # # # add the affinity_Q to the experiment params
    # # # for a in all_exp_args:
    # # #     a.append(affinity_Q)
    
    # # print(
    # #     '\n\n\n\n{}/{} experiments ran successfully!'.format(
    # #         sum(p.map(pool_function, all_exp_args)),
    # #         num_total
    # #     )
    # # )














    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    # generating the variants
    vg_fn = build_nested_variant_generator(exp_specs)
    all_exp_args = []
    for i, variant in enumerate(vg_fn()):
        all_exp_args.append([variant, i])
    
    # setting up pool and cpu affinity
    num_total = len(all_exp_args)
    num_workers = min(exp_specs['meta_data']['num_workers'], num_total)

    # cpu_range = exp_specs['meta_data']['cpu_range']
    # num_available_cpus = cpu_range[1] - cpu_range[0] + 1
    # num_cpu_per_worker = exp_specs['meta_data']['num_cpu_per_worker']
    # assert  num_cpu_per_worker * num_workers <= num_available_cpus

    # m = multiprocessing.Manager()
    # affinity_Q = m.Queue()
    # for i in range(int(num_available_cpus / num_cpu_per_worker)):
    #     affinity_Q.put(
    #         [
    #             cpu_range[0] + num_cpu_per_worker * i + j
    #             for j in range(num_cpu_per_worker)
    #         ]
    #     )

    p = Pool(num_workers)
    pool_function = get_pool_function(exp_specs['meta_data']['exp_fn_name'])

    # # add the affinity_Q to the experiment params
    # for a in all_exp_args:
    #     a.append(affinity_Q)
    
    print(
        '\n\n\n\n{}/{} experiments ran successfully!'.format(
            sum(p.map(pool_function, all_exp_args)),
            num_total
        )
    )












# if __name__ == '__main__':
#     # Arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-e', '--experiment', help='experiment specification file')
#     args = parser.parse_args()
#     with open(args.experiment, 'r') as spec_file:
#         spec_string = spec_file.read()
#         exp_specs = yaml.load(spec_string)

#     # generating the variants
#     vg_fn = build_nested_variant_generator(exp_specs)
#     all_exp_args = []
#     for i, variant in enumerate(vg_fn()):
#         all_exp_args.append([variant, i])
    
#     # setting up pool and cpu affinity
#     num_total = len(all_exp_args)
#     num_workers = min(exp_specs['meta_data']['num_workers'], num_total)

#     cpu_range = exp_specs['meta_data']['cpu_range']
#     num_available_cpus = cpu_range[1] - cpu_range[0] + 1
#     num_cpu_per_worker = exp_specs['meta_data']['num_cpu_per_worker']
#     assert  num_cpu_per_worker * num_workers <= num_available_cpus

#     m = multiprocessing.Manager()
#     affinity_Q = m.Queue()
#     for i in range(int(num_available_cpus / num_cpu_per_worker)):
#         affinity_Q.put(
#             [
#                 cpu_range[0] + num_cpu_per_worker * i + j
#                 for j in range(num_cpu_per_worker)
#             ]
#         )

#     p = Pool(num_workers)
#     pool_function = get_pool_function(exp_specs['meta_data']['exp_fn_name'])

#     # add the affinity_Q to the experiment params
#     for a in all_exp_args:
#         a.append(affinity_Q)
    
#     print(
#         '\n\n\n\n{}/{} experiments ran successfully!'.format(
#             sum(p.map(pool_function, all_exp_args)),
#             num_total
#         )
#     )
