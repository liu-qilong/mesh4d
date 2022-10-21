import numpy as np
import pyvista as pv

# load obj mesh with texture
mesh = pv.read('dataset/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj')
texture = pv.read_texture('dataset/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.jpg')
point_list = []

def callback(point):
    # create a cube and a label at the click point
    mesh = pv.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
    pl.add_mesh(mesh, style='wireframe', color='r')
    pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])

    # store picked point
    point_list.append(np.expand_dims(point, axis=0))

# launch point picking
pl = pv.Plotter()
pl.add_mesh(mesh, texture=texture, show_edges=True)
pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=True)
pl.show()

# print picked points
points = np.concatenate(point_list, axis=0)
np.save('dataset/calibrate/points_3dmd', points)
print("save picked points:\n{}".format(points))