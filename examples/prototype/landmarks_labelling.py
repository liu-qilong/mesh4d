import os
import sys
from inspect import getsourcefile

from mesh4d import utils

# include ../.. to the system path
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
parent_dir = current_dir[:parent_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)


# parameters setting
folder = 'data/meshes/6kmh_softbra_8markers/'
num = 26

# detect mesh file names
files = os.listdir(folder)
files = [os.path.join(folder, f) for f in files if '.obj' in f]
files.sort()

# landmarks labelling
for idx in range(num):
    print("="*10)
    print("labelling landmarks No. {}".format(idx))

    points = utils.obj_pick_points(
        filedir='speed_6km_soft_bra.000001.obj',
        use_texture=True,
        is_save=False,
    )

    confirm_str = input("confirm labelling? (y/n): ")
    if confirm_str == 'y':
        pass

    elif confirm_str == 'n':
        pass

    else:
        print("invalid input. extracted landmarks are regarded")