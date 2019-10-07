import numpy as np
import joblib
import os.path as osp
from rlkit.core.vistools import plot_2dhistogram, plot_scatter, plot_seaborn_heatmap


# def task_0(num_points, noise_scale):
#     '''
#     Make the puck go left, encourage fixed heigth for gripper
#     '''
#     grip_z = np.ones(num_points) * 0.435
#     X = np.ones(num_points)*0.98694312
#     Y = np.linspace(0.74910049, 0.74910049+0.35, num=num_points, endpoint=True)

#     data = np.stack([grip_z, X, Y], axis=1)
#     data[:,0] += np.random.normal(scale=noise_scale, size=data[:,0].shape)
#     data[:,1] += np.random.normal(scale=noise_scale, size=data[:,1].shape)
#     return data


# def task_1(num_points):
#     '''
#     Make the puck go left, encourage fixed heigth for gripper
#     '''
#     X = np.random.uniform(0.98697072 - 0.2, 0.98697072 + 0.05, size=num_points)
#     Y = np.random.uniform(0.74914774, 0.74914774 + 0.25, size=num_points)

#     data = np.stack([X, Y], axis=1)
#     return data

def task(num_points):
    '''
    Make the puck go left, encourage fixed heigth for gripper
    '''
    X = np.random.uniform(0.98697072 - 0.2, 0.98697072 + 0.05, size=num_points)
    Y = np.random.uniform(0.74914774, 0.74914774 + 0.25, size=num_points)

    obj_pos = np.stack([X, Y], axis=1)
    noise = np.random.normal(size=obj_pos.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    noise *= np.random.uniform(low=0.03, high=0.06, size=noise.shape)
    grip_pos = obj_pos + noise

    data = np.concatenate([obj_pos, grip_pos], axis=1)
    return data


def column(num_points):
    X = np.random.uniform(0.98697072 - 0.05, 0.98697072 + 0.05, size=num_points)
    Y = np.random.uniform(0.74914774, 0.74914774 + 0.35, size=num_points)

    obj_pos = np.stack([X, Y], axis=1)
    noise = np.random.normal(size=obj_pos.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    noise *= np.random.uniform(low=0.03, high=0.06, size=noise.shape)
    grip_pos = obj_pos + noise

    data = np.concatenate([obj_pos, grip_pos], axis=1)
    return data


def uniform_box(num_points):
    center = np.array([1.34196849, 0.74910081])

    X = np.random.uniform(center[0] - 0.2, center[0] + 0.15, size=num_points)
    Y = np.random.uniform(center[1] - 0.25, center[1] + 0.25, size=num_points)

    obj_pos = np.stack([X, Y], axis=1)
    dist = np.linalg.norm(obj_pos - center[None, :], axis=1)
    obj_pos = obj_pos[dist > 0.08, :]

    noise = np.random.normal(size=obj_pos.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    noise *= np.random.uniform(low=0.04, high=0.06, size=noise.shape)
    grip_pos = obj_pos + noise

    data = np.concatenate([obj_pos, grip_pos], axis=1)
    return data


# 0.3, 0.6


if __name__ == '__main__':
    
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/square_fixed_height.pkl'
    # data = task_1(4000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/square_fixed_height_with_grip_pos.pkl'
    # data = task(8000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/slide_column.pkl'
    # data = column(8000)

    save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/corl_box_pushing_from_center.pkl'
    data = uniform_box(10000)




    X = data[:,0]
    Y = data[:,1]
    plot_scatter(
        X, Y, 30, 'test', 'plots/data_gen/obj_scatter.png',
        [
            [1.34196849 - 0.3, 1.34196849 + 0.2],
            [0.74910081 - 0.35, 0.74910081 + 0.35]
        ]
    )
    X = data[:,2]
    Y = data[:,3]
    plot_scatter(
        X, Y, 30, 'test', 'plots/data_gen/grip_scatter.png',
        [
            [1.34196849 - 0.3, 1.34196849 + 0.2],
            [0.74910081 - 0.35, 0.74910081 + 0.35]
        ]
    )

    print(data.shape)
    joblib.dump(
        {
            'data': data,
        },
        save_path,
        compress=3
    )





    # X = data[:,0]
    # Y = data[:,1]
    # plot_scatter(
    #     X, Y, 30, 'test', 'plots/data_gen/obj_scatter.png',
    #     [
    #         [0.98697072 - 0.3, 0.98697072 + 0.175],
    #         [0.74914774 - 0.35, 0.74914774 + 0.45]
    #     ]
    # )
    # X = data[:,2]
    # Y = data[:,3]
    # plot_scatter(
    #     X, Y, 30, 'test', 'plots/data_gen/grip_scatter.png',
    #     [
    #         [0.98697072 - 0.3, 0.98697072 + 0.175],
    #         [0.74914774 - 0.35, 0.74914774 + 0.45]
    #     ]
    # )

    # print(data.shape)
    # joblib.dump(
    #     {
    #         'xy_data': data,
    #     },
    #     save_path,
    #     compress=3
    # )
