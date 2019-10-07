import numpy as np
import joblib
import os.path as osp
from rlkit.core.vistools import plot_2dhistogram, plot_scatter, plot_seaborn_heatmap


def pusher_to_obj_to_goal(num_episodes):
    obj_Z = -0.275
    init_arm_pos = np.array([8.20999983e-01, -5.99903808e-01, -1.25506088e-04])
    target_pos = np.array([0.45, -0.05, obj_Z])

    all_samples = []
    for _ in range(num_episodes):
        arm_traj = []
        obj_traj = []

        init_obj_pos = np.array([0.45, -0.05, obj_Z])
        while True:
            cylinder_pos = np.concatenate([
                np.random.uniform(low=-0.2, high=0.2, size=1),
                np.random.uniform(low=-0.3, high=0, size=1)
            ])
            if np.linalg.norm(cylinder_pos - np.zeros(2)) > 0.17:
                break
        init_obj_pos[:2] = cylinder_pos + target_pos[:2]

        # move arm to object
        w = np.linspace(0, 1, num=40, endpoint=True)[:,None]
        arm_traj.append(
            w * init_obj_pos[None,:] + (1-w) * init_arm_pos[None,:]
        )
        obj_traj.append(
            np.repeat(init_obj_pos[None,:], 40, axis=0)
        )

        # move arm and obj to target
        w = np.linspace(0, 1, num=40, endpoint=True)[:,None]
        arm_traj.append(
            w * target_pos[None,:] + (1-w) * init_obj_pos[None,:]
        )
        obj_traj.append(
            w * target_pos[None,:] + (1-w) * init_obj_pos[None,:]
        )

        # add samples in the target region
        target_region_pos = np.repeat(target_pos[None,:], 20, axis=0)
        # target region radius is 0.08, so std 0.02 works
        target_region_pos[:,:2] += np.random.normal(scale=0.02, size=(target_region_pos.shape[0], 2))
        arm_traj.append(
            target_region_pos
        )
        obj_traj.append(
            target_region_pos
        )

        # put the samples together
        # arm_traj, obj_traj[:,:2]
        arm_traj = np.concatenate(arm_traj, axis=0)
        obj_traj = np.concatenate(obj_traj, axis=0)
        target_traj = np.repeat(target_pos[None,:], arm_traj.shape[0], axis=0)

        all_samples.append(
            np.concatenate([arm_traj, obj_traj[:,:2]], axis=1)
        )
    
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples



def pusher_to_obj_to_goal_gaussian_target(num_episodes):
    obj_Z = -0.275
    init_arm_pos = np.array([8.20999983e-01, -5.99903808e-01, -1.25506088e-04])
    target_pos = np.array([0.45, -0.05, obj_Z])

    all_samples = []
    for _ in range(num_episodes):
        arm_traj = []
        obj_traj = []

        init_obj_pos = np.array([0.45, -0.05, obj_Z])
        while True:
            cylinder_pos = np.concatenate([
                np.random.uniform(low=-0.2, high=0.2, size=1),
                np.random.uniform(low=-0.3, high=0, size=1)
            ])
            if np.linalg.norm(cylinder_pos - np.zeros(2)) > 0.17:
                break
        init_obj_pos[:2] = cylinder_pos + target_pos[:2]

        # move arm to object
        w = np.linspace(0, 1, num=50, endpoint=True)[:,None]
        arm_traj.append(
            w * init_obj_pos[None,:] + (1-w) * init_arm_pos[None,:]
        )
        obj_traj.append(
            np.repeat(init_obj_pos[None,:], 50, axis=0)
        )

        # gaussian from target center
        target_region_pos = np.repeat(target_pos[None,:], 50, axis=0)
        target_region_pos[:,:2] += np.random.normal(scale=0.02, size=(target_region_pos.shape[0], 2))
        arm_traj.append(
            target_region_pos
        )
        obj_traj.append(
            target_region_pos
        )

        # put the samples together
        # arm_traj, obj_traj[:,:2]
        arm_traj = np.concatenate(arm_traj, axis=0)
        obj_traj = np.concatenate(obj_traj, axis=0)
        target_traj = np.repeat(target_pos[None,:], arm_traj.shape[0], axis=0)

        all_samples.append(
            np.concatenate([arm_traj, obj_traj[:,:2]], axis=1)
        )
    
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples


def pusher_to_obj_to_goal_gaussian_target_with_gaussian_line(num_episodes):
    obj_Z = -0.275
    init_arm_pos = np.array([8.20999983e-01, -5.99903808e-01, -1.25506088e-04])
    target_pos = np.array([0.45, -0.05, obj_Z])

    all_samples = []
    for _ in range(num_episodes):
        arm_traj = []
        obj_traj = []

        init_obj_pos = np.array([0.45, -0.05, obj_Z])
        while True:
            cylinder_pos = np.concatenate([
                np.random.uniform(low=-0.2, high=0.2, size=1),
                np.random.uniform(low=-0.3, high=0, size=1)
            ])
            if np.linalg.norm(cylinder_pos - np.zeros(2)) > 0.17:
                break
        init_obj_pos[:2] = cylinder_pos + target_pos[:2]

        # move arm to object
        delta = init_arm_pos - init_obj_pos
        delta /= np.linalg.norm(delta)
        delta = delta[None, :]
        w = 0.1 * np.abs(np.random.normal(size=(50,1)))
        # print(w)
        # print(delta.shape)
        # print(w.shape)
        arm_traj.append(
            w * delta + init_obj_pos[None,:]
        )
        obj_traj.append(
            np.repeat(init_obj_pos[None,:], 50, axis=0)
        )

        # gaussian from target center
        target_region_pos = np.repeat(target_pos[None,:], 50, axis=0)
        target_region_pos[:,:2] += np.random.normal(scale=0.02, size=(target_region_pos.shape[0], 2))
        arm_traj.append(
            target_region_pos
        )
        obj_traj.append(
            target_region_pos
        )

        # put the samples together
        # arm_traj, obj_traj[:,:2]
        arm_traj = np.concatenate(arm_traj, axis=0)
        obj_traj = np.concatenate(obj_traj, axis=0)
        target_traj = np.repeat(target_pos[None,:], arm_traj.shape[0], axis=0)

        all_samples.append(
            np.concatenate([arm_traj, obj_traj[:,:2]], axis=1)
        )
    
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples



if __name__ == '__main__':
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/pusher_task_states_200_eps_correct.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/test.pkl'
    # pusher_points = pusher_to_obj_to_goal(200)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/pusher_task_states_no_inbetween.pkl'
    save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/corl_pusher_push_states.pkl'
    pusher_points = pusher_to_obj_to_goal_gaussian_target(200)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/with_gaussian_line_pusher_task_states_no_inbetween.pkl'
    # pusher_points = pusher_to_obj_to_goal_gaussian_target_with_gaussian_line(200)


    plot_scatter(
        pusher_points[:,0], pusher_points[:,1], 30, 'test', 'plots/data_gen/pusher_arm_x_y.png', [[-0.5,1], [-1,0.5]]
    )
    plot_scatter(
        pusher_points[:,1], pusher_points[:,2], 30, 'test', 'plots/data_gen/pusher_arm_y_z.png', [[-1,0.5], [-0.5,0.5]]
    )
    plot_scatter(
        pusher_points[:,3], pusher_points[:,4], 30, 'test', 'plots/data_gen/pusher_obj.png', [[-0.5,1], [-1,0.5]]
    )
    print(pusher_points.shape)

    joblib.dump(
        {
            'data': pusher_points,
        },
        save_path,
        compress=3
    )
