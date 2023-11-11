import os
import numpy as np
import pyvista as pv
from mesh4d import obj3d, obj4d, kps, utils
from mesh4d.analyse.crave import clip_with_contour

os.environ['DISPLAY'] = ':99.0'
os.environ['PYVISTA_OFF_SCREEM'] = 'true'

# settings
is_export = True
off_screen = True
mesh_path = '/mnt/d/knpob/4-data/20211229-DynaBreast4D/3dmd/6kmh_27marker_2/meshes'
landmark_path = '/mnt/d/knpob/4-data/20211229-DynaBreast4D/vicon/6kmh_27marker_2'
start=0
stride = 12
end=120
export_folder = "output/visual"

# load data
mesh_ls, texture_ls = obj3d.load_mesh_series(
    folder=mesh_path,
    start=start,
    stride=stride,
    end=end,
)

vicon_arr = np.load(os.path.join(landmark_path, 'vicon_arr.npy'))
vicon_start = utils.load_pkl_object(os.path.join(landmark_path, 'vicon_start.pkl'))
vicon_cab = utils.load_pkl_object(os.path.join(landmark_path, 'vicon>>3dmd.pkl'))

landmarks = kps.MarkerSet()
landmarks.load_from_array(vicon_arr, start_time=vicon_start, fps=100, trans_cab=vicon_cab)
landmarks.interp_field()

# data processing
contour = landmarks.extract((0, 1, 10, 17, 25, 26))
mesh_clip_ls = clip_with_contour(mesh_ls, start_time=0, fps=120/stride, contour=contour, clip_bound='xy', margin=30)

body_ls = obj3d.init_obj_series(
    mesh_ls, 
    obj_type=obj3d.Obj3d_Deform
    )

breast_ls = obj3d.init_obj_series(
    mesh_clip_ls, 
    obj_type=obj3d.Obj3d_Deform
    )

body4d = obj4d.Obj4d_Deform(
    fps=120 / stride,
    enable_rigid=False,
    enable_nonrigid=False,
)
body4d.add_obj(*body_ls)
body4d.load_markerset('landmarks', landmarks)

breast4d = obj4d.Obj4d_Deform(
    fps=120 / stride,
    enable_rigid=False,
    enable_nonrigid=False,
)
breast4d.add_obj(*breast_ls)
breast4d.load_markerset('landmarks', landmarks)

# visualization

if is_export:
    body4d.animate(export_folder=export_folder, filename='body4d')

body4d.show(elements='m', off_screen=off_screen, is_export=is_export, export_folder=export_folder, export_name='body4d')

body4d.show(elements='mk', off_screen=off_screen, is_export=is_export, export_folder=export_folder, export_name='body4d_kps')

if is_export:
    breast4d.animate(export_folder=export_folder, filename='breast4d')
    breast4d.animate(export_folder=export_folder, filename='breast4d_raw', elements='m')

# stack view
stack_dist = 1000
zoom_rate = 3.5
window_size = [2000, 800]

scene = pv.Plotter(off_screen=off_screen)
plot_num = len(body4d.obj_ls)

for idx in range(0, plot_num):
    body = body4d.obj_ls[idx]
    breast = breast4d.obj_ls[idx]
    width = body.get_width()

    body.add_mesh_to_scene(scene, location=[0, 0, idx*stack_dist], opacity=0.1)
    breast.add_mesh_to_scene(scene, location=[0, 0, idx*stack_dist])
    breast.add_kps_to_scene(scene, location=[0, 0, idx*stack_dist], radius=0.02*width)
    
scene.camera_position = 'zy'
scene.camera.azimuth = 45
scene.camera.zoom(zoom_rate)
scene.window_size = window_size
scene.enable_parallel_projection()
# scene.show(interactive_update=True)
scene.show()

if is_export:
    export_path = os.path.join(export_folder, 'crop_stack.png')
    scene.update()
    scene.screenshot(export_path)