import os

num_cpu_per_worker = 1
cpu_range = [0,159]

def get_legal_cpus(cpu_range, num_cpu_per_worker):
    num_available_cpus = cpu_range[1] - cpu_range[0] + 1
    affinities = []
    for i in range(int(num_available_cpus / num_cpu_per_worker)):
        affinities.append(
            [
                cpu_range[0] + num_cpu_per_worker * i + j
                for j in range(num_cpu_per_worker)
            ]
        )
    affinities = [hex(sum(2**i for i in aff)) for aff in affinities]

    legal_cpus = []
    for i, aff in enumerate(affinities):
        command_to_run = 'taskset {} python -c \"x=1\" >/dev/null 2>&1'.format(aff)
        if os.system(command_to_run) == 0: legal_cpus.append(i)
    return legal_cpus

legal_cpus = get_legal_cpus(cpu_range, num_cpu_per_worker)
print(legal_cpus)
