import joblib
from os import path as osp

import numpy as np

NUM_ITEMS = 2
MIN_ANGLE_BETWEEN_ITEMS = np.pi/6
RADIUS = 10
ACCEPT_RADIUS = 1
SAMPLE_DENSITY = 10 # number of samples per unit travel distance

NUM_EPISODES = 100


def angles_check(prev_as, new_a):
    if len(prev_as) == 0:
        return True
    for a in prev_as:
        if abs(a - new_a) < MIN_ANGLE_BETWEEN_ITEMS:
            return False
    return True


# obs: [cur_pos, obj0_pos, ...., wentto0, ...]
all_samples = []
for _ in range(NUM_EPISODES):
    # sample the positions
    angles = []
    for _ in range(NUM_ITEMS):
        new_a = np.random.uniform(high=np.pi)
        while not angles_check(angles, new_a):
            new_a = np.random.uniform(high=np.pi)
        angles.append(new_a)
    
    angles = np.array(angles)
    obj_poses = np.stack([RADIUS*np.cos(angles), RADIUS*np.sin(angles)], axis=1)
    flat_obj_poses = obj_poses.flatten()

    prev_pos = np.zeros(2)
    visitation_state = np.zeros(NUM_ITEMS)
    for i in range(obj_poses.shape[0]):
        next_pos = obj_poses[i]

        # the inter points
        num_inter = SAMPLE_DENSITY * np.linalg.norm(next_pos - prev_pos)
        repeated_prev_pos = np.repeat(prev_pos[None,:], num_inter, axis=0)
        repeated_next_pos = np.repeat(next_pos[None,:], num_inter, axis=0)
        w = np.linspace(0, 1, num=num_inter, endpoint=True)[:,None]
        inter_pos = w*repeated_prev_pos + (1-w)*repeated_next_pos


        # the states
        inter_visits = np.repeat(visitation_state[None,:], num_inter, axis=0)
        within_accept = np.linalg.norm(inter_pos - repeated_next_pos, axis=1) < ACCEPT_RADIUS
        inter_visits[within_accept,i] = 1.0
        # print(inter_visits)
        # print(inter_pos)

        # obj poses
        repeated_obj_poses = np.repeat(flat_obj_poses[None,:], num_inter, axis=0)

        # add the new samples
        all_samples.append(
            np.concatenate((inter_pos, repeated_obj_poses, inter_visits), axis=1)
        )
        # print(all_samples[-1])

        prev_pos = next_pos
        visitation_state[i] = 1.0

all_samples = np.concatenate(all_samples, axis=0)
print(all_samples.shape)

save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/num_items_2_r_20_acc_1_eps_100_no_noise.pkl'
joblib.dump(
    {
        'data': all_samples,
    },
    save_path,
    compress=3
)
