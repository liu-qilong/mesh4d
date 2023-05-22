# include ../../.. to the system path
from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

# parameter settings
is_plot = True
is_export = True

landmarks_path = 'data/landmarks/refine_6kmh_braless_18markers_12fps.pkl'
meshes_path = 'data/meshes/6kmh_braless_26markers/'
test_landmarks_path = 'data/test/braless_random_landmarks.pkl'

start=0
stride = 12
end=120

export_folder = "output/data/"

# data loading
from mesh4d import obj3d

mesh_ls, texture_ls = obj3d.load_mesh_series(
    folder=meshes_path,
    start=start,
    stride=stride,
    end=end,
)

from mesh4d import utils

landmarks = utils.load_pkl_object(landmarks_path)
landmarks.interp_field()

from mesh4d.analyse.crave import clip_with_contour

contour = landmarks.extract(('marker 0', 'marker 2', 'marker 3', 'marker 14', 'marker 15', 'marker 17'))
mesh_clip_ls = clip_with_contour(mesh_ls, start_time=0, fps=120/stride, contour=contour, clip_bound='xy', margin=30)

body_ls = obj3d.init_obj_series(
    mesh_ls, 
    obj_type=obj3d.Obj3d_Deform
    )

breast_ls = obj3d.init_obj_series(
    mesh_clip_ls, 
    obj_type=obj3d.Obj3d_Deform
    )

from mesh4d import obj4d

body4d = obj4d.Obj4d_Deform(
    fps=120 / stride,
    enable_rigid=False,
    enable_nonrigid=False,
)
body4d.add_obj(*body_ls)
body4d.load_markerset('landmarks', landmarks)

from mesh4d import obj4d

breast4d = obj4d.Obj4d_Deform(
    fps=120 / stride,
    enable_rigid=False,
    enable_nonrigid=False,
)
breast4d.add_obj(*breast_ls)
breast4d.load_markerset('landmarks', landmarks)

# landmarks index

import pyvista as pv

if is_plot:
    scene = pv.Plotter()
    
    width = body_ls[1].get_width()
    body_ls[1].add_mesh_to_scene(scene)
    body_ls[1].add_kps_to_scene(scene, radius=0.02*width)

    kps_points = body_ls[1].kps_group['landmarks'].get_points_coord()
    poly = pv.PolyData(kps_points)
    poly["idx"] = [f"#{i}" for i in range(poly.n_points)]
    scene.add_point_labels(poly, "idx", point_size=20, font_size=36)
    
    scene.camera_position = 'xy'
    scene.show()