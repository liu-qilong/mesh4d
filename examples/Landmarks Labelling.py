# include ../../.. to the system path
from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from mesh4d import utils

utils.landmarks_labelling(
    # mesh_folder = 'data/meshes/6kmh_braless_26markers/',
    mesh_folder = 'data/meshes/6kmh_softbra_8markers/',
    mesh_fps = 120,
    point_num = 8,
    start = 120,
    end = 120,
    stride = 1,
    is_save = True,
    export_path = 'examples/output/landmarks.pkl',
    )