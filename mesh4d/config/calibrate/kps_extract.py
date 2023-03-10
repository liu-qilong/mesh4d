# include ../../.. to the system path
from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
parent_dir = current_dir[:parent_dir.rfind(os.path.sep)]
parent_dir = current_dir[:parent_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from mesh4d import utils

# pick points
utils.obj_pick_points(
    filedir='data/meshes/static_3markers/static_standing.000001.obj',
    use_texture=True,
    is_save=False,
    save_folder='mesh4d/config/calibrate/',
    save_name='points_3dmd',
)