import numpy as np

WALL_LEN = 20.0
NUM_PATHS = 8.0
WALL_FROM_CENTER = 8.0
TARGET_DIST = 25.0

xml_block = '<body mocap=\"true\" pos=\"{x} {y} 1\" euler="0 0 {z_rot}\">\n\
\t<geom type=\"box\" size=\"{wall_len} 0.5 2\" group=\"1\" condim=\"3\" conaffinity=\"1\"/>\n\
</body>'

increments = 2*np.pi / NUM_PATHS
half_increments = np.pi / NUM_PATHS
angles = np.linspace(0.0, 2*np.pi, num=NUM_PATHS, endpoint=False)

for a in angles:
    left_and_right = []
    for inc in [-half_increments, half_increments]:
        xy = np.array([np.cos(a+inc), np.sin(a+inc)]) * WALL_FROM_CENTER
        left_and_right.append(
            xy + np.array([np.cos(a), np.sin(a)]) * WALL_LEN
        )
        xy += np.array([np.cos(a), np.sin(a)]) * WALL_LEN / 2.0
        print(
            xml_block.format(
                **{
                    'x': '%.2f' % xy[0],
                    'y': '%.2f' % xy[1],
                    'z_rot': '%.2f' % (a * 360.0 / (2*np.pi)),
                    'wall_len': WALL_LEN / 2.0
                }
            )
        )
    
    end_block_len = np.linalg.norm(left_and_right[0] - left_and_right[1]) + 1.0
    mid_point = sum(left_and_right) / 2.0
    print(
        xml_block.format(
            **{
                'x': '%.2f' % mid_point[0],
                'y': '%.2f' % mid_point[1],
                'z_rot': '%.2f' % ((a + np.pi/2.0) * 360.0 / (2*np.pi)),
                'wall_len': end_block_len / 2.0
            }
        )
    )


# print('\n')

for i, a in enumerate(angles):
    xy = np.array([np.cos(a), np.sin(a)]) * TARGET_DIST
    print(
        "<site name=\"target{num}\" pos=\"{x} {y} .01\" rgba=\"0.75 0 0.75 1\" type=\"sphere\" size=\"1\"/>".format(
            num=i,
            x='%.2f' % xy[0],
            y='%.2f' % xy[1]
        )
    )
