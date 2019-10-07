import numpy as np
import joblib
import os.path as osp
from rlkit.core.vistools import plot_2dhistogram, plot_scatter, plot_seaborn_heatmap

def line_data(length, width_scale, num_points):
    X = np.random.normal(loc=0.0, scale=width_scale, size=num_points)
    Y = 2*length*np.random.uniform(size=num_points) - length
    return X, Y


def ant_spiral(length, noise_scale, num_points):
    X = np.concatenate(
        [
            np.linspace(0.0, -length, num=int(num_points/8), endpoint=True),
            np.ones(int(num_points/8)) * -length,
            np.linspace(-length, length, num=int(num_points/4), endpoint=True),
            np.ones(int(num_points/4)) * length,
            np.linspace(length, -length, num=int(num_points/4), endpoint=True),
        ]
    )
    Y = np.concatenate(
        [
            np.zeros(int(num_points/8)),
            np.linspace(0.0, -length, num=int(num_points/8), endpoint=True),
            np.ones(int(num_points/4)) * (-length),
            np.linspace(-length, length, num=int(num_points/4), endpoint=True),
            np.ones(int(num_points/4)) * length,
        ]
    )
    X += np.random.normal(scale=noise_scale, size=num_points)
    Y += np.random.normal(scale=noise_scale, size=num_points)
    return X, Y


def ant_line(length, noise_scale, num_points):
    X = np.linspace(0.0, length, num=num_points, endpoint=True)
    Y = np.random.normal(scale=noise_scale, size=num_points)
    return X, Y

def ant_disc(radius, num_points):
    a = np.random.uniform(0, 2*np.pi, size=num_points)
    r = radius * np.random.uniform(size=num_points)**0.5
    X = r*np.cos(a)
    Y = r*np.sin(a)
    return X, Y

def two_gaussians(distance, scale, num_points, lower_only=False):
    X, Y = [], []
    for _ in range(num_points):
        if lower_only:
            c = np.array([0.0, -distance])
        else:
            if np.random.uniform() > 0.5:
                c = np.array([0.0, distance])
            else:
                c = np.array([0.0, -distance])
        noise = np.random.normal(scale=scale, size=2)
        X.append(c[0] + noise[0])
        Y.append(c[1] + noise[1])
    return np.array(X), np.array(Y)


def four_gaussians(distance, scale, num_points):
    X, Y = [], []
    for _ in range(num_points):
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                c = np.array([0.0, distance])
            else:
                c = np.array([0.0, -distance])
        else:
            if np.random.uniform() > 0.5:
                c = np.array([distance, 0.0])
            else:
                c = np.array([-distance, 0.0])
        noise = np.random.normal(scale=scale, size=2)
        X.append(c[0] + noise[0])
        Y.append(c[1] + noise[1])
    return np.array(X), np.array(Y)


def waypoints(points, scale, num_points):
    num_per_point = int(float(num_points) / len(points))
    points = np.repeat(points, num_per_point, axis=0)
    points = points + np.random.normal(scale=scale, size=points.shape)
    return points[:,0], points[:,1]


def Lpath(scale, num_points):
    X = np.concatenate(
        (
            np.random.uniform(low=-5.0 + 3*scale, high=5.0, size=int(num_points/2)),
            5.0*np.ones(int(num_points/2))
        )
    )
    Y = np.concatenate(
        (
            -5.0*np.ones(int(num_points/2)),
            np.random.uniform(low=-5.0, high=5.0, size=int(num_points/2))
        )
    )

    X += np.random.normal(scale=scale, size=X.shape)
    Y += np.random.normal(scale=scale, size=Y.shape)
    return X, Y


def AntLpath(scale, num_points):
    X = np.concatenate(
        (
            # np.random.uniform(low=0.0 + 3*scale, high=5.0, size=int(num_points/2)),
            np.random.uniform(low=0.0, high=5.0, size=int(num_points/2)),
            5.0*np.ones(int(num_points/2)) + np.random.normal(scale=scale, size=int(num_points/2))
        )
    )
    Y = np.concatenate(
        (
            np.random.normal(scale=scale, size=int(num_points/2)),
            # 0.0*np.ones(int(num_points/2)),
            np.random.uniform(low=0.0, high=5.0, size=int(num_points/2))
        )
    )

    X += np.random.normal(scale=scale, size=X.shape)
    Y += np.random.normal(scale=scale, size=Y.shape)
    return X, Y


def AntUpath(scale, num_points):
    X = np.concatenate(
        (
            # np.random.uniform(low=0.0 + 3*scale, high=5.0, size=int(num_points/2)),
            np.random.uniform(low=-2.5, high=2.5, size=int(num_points/2)),
            2.5*np.ones(int(num_points/2)) + np.random.normal(scale=scale, size=int(num_points/2)),
            -2.5*np.ones(int(num_points/2)) + np.random.normal(scale=scale, size=int(num_points/2))
        )
    )
    Y = np.concatenate(
        (
            np.random.normal(scale=scale, size=int(num_points/2)),
            # 0.0*np.ones(int(num_points/2)),
            np.random.uniform(low=0.0, high=5.0, size=int(num_points/2)),
            np.random.uniform(low=0.0, high=5.0, size=int(num_points/2))
        )
    )

    X += np.random.normal(scale=scale, size=X.shape)
    Y += np.random.normal(scale=scale, size=Y.shape)
    return X, Y


def AntHpath(scale, num_points):
    X = np.concatenate(
        (
            # np.random.uniform(low=0.0 + 3*scale, high=5.0, size=int(num_points/2)),
            np.random.uniform(low=-5, high=5, size=int(num_points/2)),
            5*np.ones(int(num_points/2)) + np.random.normal(scale=scale, size=int(num_points/2)),
            -5*np.ones(int(num_points/2)) + np.random.normal(scale=scale, size=int(num_points/2))
        )
    )
    Y = np.concatenate(
        (
            np.random.normal(scale=scale, size=int(num_points/2)),
            # 0.0*np.ones(int(num_points/2)),
            np.random.uniform(low=-5.0, high=5.0, size=int(num_points/2)),
            np.random.uniform(low=-5.0, high=5.0, size=int(num_points/2))
        )
    )

    X += np.random.normal(scale=scale, size=X.shape)
    Y += np.random.normal(scale=scale, size=Y.shape)
    return X, Y


def AntXpath(scale, num_points):
    X = np.concatenate(
        (
            np.linspace(-5.0, 5.0, num=int(num_points/2.0), endpoint=True),
            np.linspace(-5.0, 5.0, num=int(num_points/2.0), endpoint=True)
        )
    )
    Y = np.concatenate(
        (
            np.linspace(-5.0, 5.0, num=int(num_points/2.0), endpoint=True),
            np.linspace(5.0, -5.0, num=int(num_points/2.0), endpoint=True)
        )
    )

    X += np.random.normal(scale=scale, size=X.shape)
    Y += np.random.normal(scale=scale, size=Y.shape)
    return X, Y


def sinusoidal(start_X, num_periods, period, amplitude, noise_scale, num_points):
    X = np.random.uniform(low=0.0, high=2*np.pi*num_periods, size=num_points)
    Y = amplitude * np.sin(X)
    Y += np.random.normal(scale=noise_scale, size=Y.shape)
    X *= (period / (2*np.pi))
    X += start_X
    return X, Y

def other_sinusoidal(start_X, num_periods, period, amplitude, noise_scale, num_points):
    X = np.linspace(0.0, 2*np.pi*num_periods, num=num_points, endpoint=True)
    Y = amplitude * np.sin(X)
    Y += np.random.normal(scale=noise_scale, size=Y.shape)
    X *= (period / (2*np.pi))
    X += start_X
    X += np.random.normal(scale=noise_scale, size=X.shape)
    return X, Y

def circle(radius, noise_scale, num_points):
    a = np.random.uniform(low=0.0, high=2*np.pi, size=num_points)
    r = np.ones(a.shape) * radius
    r += np.random.normal(scale=noise_scale, size=r.shape)
    return np.cos(a)*r, np.sin(a)*r


def spiral(num_rotations, radius, noise_scale, num_points):
    a = np.linspace(0.0, 2*np.pi*num_rotations, num=num_points, endpoint=True)
    r = np.linspace(0.0, radius, num=num_points, endpoint=True)
    X = r*np.cos(a)
    Y = r*np.sin(a)
    X += np.random.normal(scale=noise_scale, size=X.shape)
    Y += np.random.normal(scale=noise_scale, size=Y.shape)
    return X, Y


def pusher_sin_trace(noise, num_points):
    center = np.array([0.0, -0.6])
    amp = 0.3
    a = np.linspace(-np.pi, np.pi, num=num_points, endpoint=False)
    Z = amp*np.sin(2*(a+np.pi))
    r = np.random.uniform(0.7, 0.8, size=num_points)
    X = r*np.cos(a) + center[0]
    Y = r*np.sin(a) + center[1]
    return np.stack([X,Y,Z], axis=1), a


def infty(r, noise_scale, num_points):
    a = np.linspace(0.0, 2*np.pi, num=num_points, endpoint=True)
    X = r*(2**0.5)*np.cos(a) / (np.sin(a)**2 + 1)
    Y = X * np.sin(a)
    X += np.random.normal(scale=noise_scale, size=X.shape)
    Y += np.random.normal(scale=noise_scale, size=Y.shape)
    return X, Y


def ant_s_maze():
    points = []

    N = 300
    # # left direction
    # # down
    # X = np.zeros(N)
    # Y = -2.5*np.random.uniform(size=N)
    # points.append(np.stack([X,Y], axis=1))
    # # left
    # Y = -2.5*np.ones(6*N)
    # X = -15*np.random.uniform(size=6*N)
    # points.append(np.stack([X,Y], axis=1))
    # # up
    # X = -15*np.ones(2*N)
    # Y = 5*np.random.uniform(size=2*N) - 2.5
    # points.append(np.stack([X,Y], axis=1))
    # # right
    # Y = 2.5*np.ones(4*N)
    # X = -10*np.random.uniform(size=4*N) - 5
    # points.append(np.stack([X,Y], axis=1))

    # right direction
    # down
    X = np.zeros(N)
    Y = 2.5*np.random.uniform(size=N)
    points.append(np.stack([X,Y], axis=1))
    # left
    Y = 2.5*np.ones(6*N)
    X = 15*np.random.uniform(size=6*N)
    points.append(np.stack([X,Y], axis=1))
    # up
    X = 15*np.ones(2*N)
    Y = 5*np.random.uniform(size=2*N) - 2.5
    points.append(np.stack([X,Y], axis=1))
    # right
    Y = -2.5*np.ones(4*N)
    X = 10*np.random.uniform(size=4*N) + 5
    points.append(np.stack([X,Y], axis=1))

    points = np.concatenate(points)
    points += np.random.normal(scale=0.6, size=points.shape)
    return points[:,0], points[:,1]



if __name__ == '__main__':
    # length = 3.0
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/line_len_3_scale_0p2_4000_points.pkl'
    # X, Y = line_data(length, 0.2, 4000)

    # dist = 3.0
    # scale = 0.2
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/lower_only_two_gaussians_dist_3_scale_0p2_4000_points.pkl'
    # X, Y = two_gaussians(dist, scale, 4000, lower_only=True)

    # dist = 3.0
    # scale = 0.2
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/four_gaussians_dist_3_scale_0p2_4000_points.pkl'
    # X, Y = four_gaussians(dist, scale, 4000)

    # points = np.array(
    #     [
    #         [-5.0, -5.0],
    #         [-3.0, -5.0],
    #         [-1.0, -5.0],
    #         [1.0, -5.0],
    #         [3.0, -5.0],
    #         [5.0, -5.0],
    #         [5.0, -3.0],
    #         [5.0, -1.0],
    #         [5.0, 1.0],
    #         [5.0, 3.0],
    #         [5.0, 5.0],
    #     ]
    # )

    # points = np.array(
    #     [
    #         [-5.0, -5.0],
    #         [-2.5, -5.0],
    #         [0.0, -5.0],
    #         [2.5, -5.0],
    #         [5.0, -5.0],
    #         [5.0, -2.5],
    #         [5.0, 0.0],
    #         [5.0, 2.5],
    #         [5.0, 5.0],
    #     ]
    # )
    # scale = 0.2
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/right_down_waypoints.pkl'
    # X, Y = waypoints(points, scale, 2000)

    # scale = 0.2
    # scale = 0.1
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_L_path_noise_0p1.pkl'
    # X, Y = Lpath(scale, 4000)
    # X, Y = AntLpath(scale, 2000)


    # scale = 0.1
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_U_path_noise_0p1.pkl'
    # X, Y = AntUpath(scale, 2000)

    # scale = 0.2
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_H_path_noise_0p2.pkl'
    # X, Y = AntHpath(scale, 8000)

    # scale = 0.1
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_X_path_noise_0p1.pkl'
    # X, Y = AntXpath(scale, 8000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/corl_pm_infinity.pkl'
    # X, Y = infty(12.0, 0.3, 4000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_line_noise_scale_0p1.pkl'
    # X, Y = ant_line(7.0, 0.1, 1000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_5line_noise_0p1.pkl'
    # X, Y = ant_line(5.0, 0.1, 1000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_disc_radius_4.pkl'
    # X, Y = ant_disc(4.0, 8000)

    save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/corl_pusher_sin_trace.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/test.pkl'
    pusher_points, a = pusher_sin_trace(0.2, 8000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_spiral_3_0p1.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_spiral_3_0p3.pkl'
    # X, Y = ant_spiral(3.0, 0.3, 16000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_s_maze.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/ant_half_s_maze_noise_scale_0p6.pkl'
    # X, Y = ant_s_maze()

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/test_other_easier_sin_scale_0p6.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/test_sin.pkl'
    # X, Y = sinusoidal(-2*np.pi, 1.0, 4*np.pi, 4.0, 0.6, 4000)
    # X, Y = other_sinusoidal(-2*np.pi, 1.0, 4*np.pi, 4.0, 0.6, 4000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/circle.pkl'
    # X, Y = circle(5.0, 0.4, 4000)

    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/spiral_16.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/final_spiral.pkl'
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/corl_pm_spiral_16.pkl'
    # X, Y = spiral(2.0, 16.0, 0.3, 16000)






    # # bound = 18
    # # bound = 6
    # bound = 20
    # plot_2dhistogram(
    #     X, Y, 30, 'test', 'plots/data_gen/hist.png', [[-bound,bound], [-bound,bound]]
    # )
    # plot_scatter(
    #     X, Y, 30, 'test', 'plots/data_gen/scatter.png', [[-bound,bound], [-bound,bound]]
    # )
    # plot_seaborn_heatmap(
    #     X, Y, 30, 'test', 'plots/data_gen/heatmap.png', [[-bound,bound], [-bound,bound]]
    # )

    # XY_DATA = np.array([X,Y]).T
    # print(XY_DATA.shape)
    # print(np.mean(XY_DATA, axis=0).shape)
    # print(np.std(XY_DATA, axis=0).shape)
    # joblib.dump(
    #     {
    #         'data': XY_DATA,
    #         'xy_mean': np.mean(XY_DATA, axis=0),
    #         'xy_std': np.std(XY_DATA, axis=0)
    #     },
    #     save_path,
    #     compress=3
    # )



    bound = 2
    plot_scatter(
        pusher_points[:,0], pusher_points[:,1], 30, 'test', 'plots/data_gen/pusher_top_down.png', [[-1,1], [-1.6,0.4]]
    )
    plot_scatter(
        a,
        # np.arctan2(pusher_points[:,1], pusher_points[:,0]),
        pusher_points[:,2], 30, 'test', 'plots/data_gen/pusher_a_z.png', [[-np.pi,np.pi], [-1,1]]
    )
    print(pusher_points.shape)

    joblib.dump(
        {
            'data': pusher_points,
        },
        save_path,
        compress=3
    )
