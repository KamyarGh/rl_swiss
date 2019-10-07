import numpy as np

PATH_WIDTH = 4.0
S_LEN = 15.0
WALL_THICKNESS = 1.0

xml_block = '<body mocap=\"true\" pos=\"{x} {y} 1\" euler="0 0 {z_rot}\">\n\
\t<geom type=\"box\" size=\"{wall_len} 0.5 2\" group=\"1\" condim=\"3\" conaffinity=\"1\"/>\n\
</body>'

# build the outer box wall
width = 2*S_LEN + PATH_WIDTH + 2*WALL_THICKNESS
height = 3*WALL_THICKNESS + 2*PATH_WIDTH
print(
    xml_block.format(
        **{
            'x': '%.2f' % 0,
            'y': '%.2f' % (-height/2.0 + WALL_THICKNESS/2.0),
            'z_rot': '%.2f' % 0,
            'wall_len': width/2
        }
    )
)
print(
    xml_block.format(
        **{
            'x': '%.2f' % 0,
            'y': '%.2f' % (height/2.0 - WALL_THICKNESS/2.0),
            'z_rot': '%.2f' % 0,
            'wall_len': width/2
        }
    )
)
print(
    xml_block.format(
        **{
            'x': '%.2f' % (width/2.0 - WALL_THICKNESS/2.0),
            'y': '%.2f' % 0,
            'z_rot': '%.2f' % 90,
            'wall_len': height/2
        }
    )
)
print(
    xml_block.format(
        **{
            'x': '%.2f' % (-width/2.0 + WALL_THICKNESS/2.0),
            'y': '%.2f' % 0,
            'z_rot': '%.2f' % 90,
            'wall_len': height/2
        }
    )
)

# build the S-barriers
print(
    xml_block.format(
        **{
            'x': '%.2f' % ((-PATH_WIDTH - WALL_THICKNESS)/2.0),
            'y': '%.2f' % ((WALL_THICKNESS + PATH_WIDTH)/2.0),
            'z_rot': '%.2f' % 90,
            'wall_len': ((WALL_THICKNESS + PATH_WIDTH + WALL_THICKNESS)/2.0)
        }
    )
)
print(
    xml_block.format(
        **{
            'x': '%.2f' % ((PATH_WIDTH + WALL_THICKNESS)/2.0),
            'y': '%.2f' % ((-WALL_THICKNESS - PATH_WIDTH)/2.0),
            'z_rot': '%.2f' % 90,
            'wall_len': ((WALL_THICKNESS + PATH_WIDTH + WALL_THICKNESS)/2.0)
        }
    )
)

print(
    xml_block.format(
        **{
            'x': '%.2f' % ((-0.5*width + WALL_THICKNESS + PATH_WIDTH - 0.5*PATH_WIDTH)/2.0),
            'y': '%.2f' % 0,
            'z_rot': '%.2f' % 0,
            'wall_len': ((0.5*width - 0.5*PATH_WIDTH - PATH_WIDTH - WALL_THICKNESS)/2.0)
        }
    )
)
print(
    xml_block.format(
        **{
            'x': '%.2f' % ((0.5*width - WALL_THICKNESS - PATH_WIDTH + 0.5*PATH_WIDTH)/2.0),
            'y': '%.2f' % 0,
            'z_rot': '%.2f' % 0,
            'wall_len': ((0.5*width - 0.5*PATH_WIDTH - PATH_WIDTH - WALL_THICKNESS)/2.0)
        }
    )
)
