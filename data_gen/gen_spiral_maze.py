import numpy as np

PATH_WIDTH = 4.0
S_LEN = 15.0
WALL_THICKNESS = 1.0

def print_wall(start, end, angle):
    xml_block = '<body mocap=\"true\" pos=\"{x} {y} 1\" euler="0 0 {z_rot}\">\n\
    \t<geom type=\"box\" size=\"{wall_len} 0.5 2\" group=\"1\" condim=\"3\" conaffinity=\"1\"/>\n\
    </body>'
    mid = (start + end)/2.0
    print(
        xml_block.format(
            **{
                'x': '%.2f' % mid[0],
                'y': '%.2f' % mid[1],
                'z_rot': '%.2f' % angle,
                'wall_len': np.linalg.norm(end - start)/2.0
            }
        )
    )

prev_point = np.array([-PATH_WIDTH/2.0, -(PATH_WIDTH+WALL_THICKNESS)/2.0])
next_point = np.array([WALL_THICKNESS + PATH_WIDTH/2.0, -(PATH_WIDTH+WALL_THICKNESS)/2.0])
print_wall(prev_point, next_point, 0)

prev_point = np.array([0.5*PATH_WIDTH + 0.5*WALL_THICKNESS, -0.5*PATH_WIDTH - WALL_THICKNESS])
next_point = np.array([0.5*PATH_WIDTH + 0.5*WALL_THICKNESS, 0.5*PATH_WIDTH + WALL_THICKNESS])
print_wall(prev_point, next_point, 90)

prev_point = np.array([0.5*PATH_WIDTH + WALL_THICKNESS, 0.5*PATH_WIDTH + 0.5*WALL_THICKNESS])
next_point = np.array([-1.5*PATH_WIDTH - WALL_THICKNESS, 0.5*PATH_WIDTH + 0.5*WALL_THICKNESS])
print_wall(prev_point, next_point, 0)

prev_point = np.array([-1.5*PATH_WIDTH - 0.5*WALL_THICKNESS, 0.5*PATH_WIDTH + WALL_THICKNESS])
next_point = np.array([-1.5*PATH_WIDTH - 0.5*WALL_THICKNESS, -1.5*PATH_WIDTH - 2*WALL_THICKNESS])
print_wall(prev_point, next_point, 90)

prev_point = np.array([-1.5*PATH_WIDTH - WALL_THICKNESS, -1.5*PATH_WIDTH - 1.5*WALL_THICKNESS])
next_point = np.array([1.5*PATH_WIDTH + 2*WALL_THICKNESS, -1.5*PATH_WIDTH - 1.5*WALL_THICKNESS])
print_wall(prev_point, next_point, 0)

prev_point = np.array([1.5*PATH_WIDTH + 1.5*WALL_THICKNESS, -1.5*PATH_WIDTH - 2*WALL_THICKNESS])
next_point = np.array([1.5*PATH_WIDTH + 1.5*WALL_THICKNESS, 1.5*PATH_WIDTH + 2*WALL_THICKNESS])
print_wall(prev_point, next_point, 90)

prev_point = np.array([1.5*PATH_WIDTH + 2*WALL_THICKNESS, 1.5*PATH_WIDTH + 1.5*WALL_THICKNESS])
next_point = np.array([-2.5*PATH_WIDTH - 2*WALL_THICKNESS, 1.5*PATH_WIDTH + 1.5*WALL_THICKNESS])
print_wall(prev_point, next_point, 0)

prev_point = np.array([-2.5*PATH_WIDTH - 1.5*WALL_THICKNESS, 1.5*PATH_WIDTH + 2*WALL_THICKNESS])
next_point = np.array([-2.5*PATH_WIDTH - 1.5*WALL_THICKNESS, -2.5*PATH_WIDTH - 3*WALL_THICKNESS])
print_wall(prev_point, next_point, 90)

prev_point = np.array([-2.5*PATH_WIDTH - 2*WALL_THICKNESS, -2.5*PATH_WIDTH - 2.5*WALL_THICKNESS])
next_point = np.array([1.5*PATH_WIDTH + 2*WALL_THICKNESS, -2.5*PATH_WIDTH - 2.5*WALL_THICKNESS])
print_wall(prev_point, next_point, 0)
