import numpy as np
import pyvista as pv

# load obj mesh with texture
mesh = pv.read('dataset/45kmh_26markers_12fps/speed_45km_h_26_marker_set_1.000001.obj')
texture = pv.read_texture('dataset/45kmh_26markers_12fps/speed_45km_h_26_marker_set_1.000001.jpg')
points = []

def callback(point):
    # create a cube and a label at the click point
    mesh = pv.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
    pl.add_mesh(mesh, style='wireframe', color='r')
    pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])

    # store picked point
    points.append(np.expand_dims(point, axis=0))

# launch point picking
pl = pv.Plotter()
pl.add_mesh(mesh, texture=texture, show_edges=True)
pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=True)
pl.show()

# print picked points
print(np.concatenate(points, axis=0))