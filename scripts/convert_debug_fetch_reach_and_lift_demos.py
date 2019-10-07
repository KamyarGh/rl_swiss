import numpy as np
import joblib
import os
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer

# get the original
her_demos_path = '/ais/gobi6/kamyar/oorl_rlkit/expert_demos/her_fetch_pick_and_place/fetch_reach_and_lift/data_for_fetch_reach_and_lift_random_100.npz'
rlkit_buffer_save_dir = '/ais/gobi6/kamyar/oorl_rlkit/expert_demos/her_fetch_pick_and_place/fetch_reach_and_lift/'
d = np.load(her_demos_path)

# make the buffer
buffer_size = sum(len(path) for path in d['obs'])
obs_dim = {
    'obs': d['obs'][0][0]['observation'].shape[0],
    'obs_task_params': 3
}
action_dim = len(d['acs'][0][0])
buffer = SimpleReplayBuffer(buffer_size, obs_dim, action_dim)

# fill the buffer
for path_num in range(len(d['obs'])):
    obs = d['obs'][path_num]
    acs = d['acs'][path_num]

    ep_len = len(obs)
    debug_goal = obs[0]['observation'][3:6].copy()
    for j in range(ep_len-1):
        o = {
            'obs': obs[j]['observation'],
            'obs_task_params': debug_goal
        }
        a = acs[j]
        r = 0. # the demons don't come with reward
        terminal = 0 # none of the robotic environments in gym have terminal 1 ever
        next_o = {
            'obs': obs[j+1]['observation'],
            'obs_task_params': debug_goal
        }
        buffer.add_sample(o, a, r, terminal, next_o, agent_info={}, env_info={})
    buffer.terminate_episode()

# save it
file_name = os.path.join(rlkit_buffer_save_dir, 'extra_data.pkl')
joblib.dump({'replay_buffer': buffer}, file_name, compress=3)
