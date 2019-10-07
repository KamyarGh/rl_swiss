import os
import os.path as osp
from subprocess import call

import numpy as np
from scipy.misc import imread

import joblib

# out%02d.png
command_format = 'convert -coalesce {} {}'

def do_the_thing(object_dir):
    print(object_dir)
    all_files = list(os.listdir(object_dir))
    # print(all_files)
    num = 0
    for fname in all_files:
        if 'gif' not in fname: continue
        # print(fname)
        # print(num)
        num += 1
        
        # split the gif
        splitting_command = command_format.format(
            osp.join(object_dir, fname),
            osp.join(object_dir, r'out%02d.png')
        )
        call(splitting_command.split())

        # now read the frames, join them, and save
        frames = []
        for i in range(100):
            img = imread(osp.join(object_dir, 'out%02d.png'%i))
            # img = imread(osp.join(object_dir, 'out%02d.png'%i)).astype(np.float32)
            # img /= 255.0
            if img.shape[-1] == 4:
                img = img[:,:,:3]
            frames.append(img)
        frames = np.array(frames)
        frames = np.transpose(frames, (3,0,1,2))
        np.save(osp.join(object_dir, 'demo_%d' % num), frames)
        # joblib.dump(
        #     {'frames': frames},
        #     osp.join(object_dir, 'demo_%d' % num),
        #     compress=3
        # )


# do_the_thing('/scratch/ssd001/home/kamyar/mil/data/sim_push/object_0')
base_path = '/scratch/ssd001/home/kamyar/mil/data/sim_push/object_%d'
for obj_num in range(700,769):
    do_the_thing(base_path % obj_num)
