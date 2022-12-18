# run at file path

# add root folder of the project to path
import sys
sys.path.insert(0, '../..')

# load kps
from UltraMotionCapture import kps, obj3d

vicon = kps.MarkerSet()
vicon.load_from_vicon('../../UltraMotionCapture/data/6kmh_softbra_8markers_1.csv')
vicon.interp_field()

points_vicon = vicon.get_frame_coord(0)
pvpcd_vicon = obj3d.np2pvpcd(points_vicon, radius=10, phi_resolution=10, theta_resolution=10)

# load parameters
import numpy as np
from UltraMotionCapture import field

r = np.load('../../UltraMotionCapture/config/calibrate/r.npy')
s = np.load('../../UltraMotionCapture/config/calibrate/s.npy')
t = np.load('../../UltraMotionCapture/config/calibrate/t.npy')
s, m = field.transform_rst2sm(r, s, t)

# load mesh
import pyvista as pv
pv.set_jupyter_backend('pythreejs')

mesh = pv.read('../../UltraMotionCapture/data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj')
texture = pv.read_texture('../../UltraMotionCapture/data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.jpg')

mesh.transform(m)
mesh.scale(s)

# plot mesh and kps
scene = pv.Plotter()
scene.add_mesh(mesh, show_edges=True)
scene.add_mesh(pvpcd_vicon, color='Blue')

for i in range(len(points_vicon)):
    point = points_vicon[i, :]
    scene.add_point_labels(point, str(i), font_size=25)

scene.show()