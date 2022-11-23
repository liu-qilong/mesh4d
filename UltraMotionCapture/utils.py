from __future__ import annotations
from typing import Type, Union, Iterable

import os
import imageio
import numpy as np
import pyvista as pv

def images_to_gif(path: Union[str, None] = None, remove: bool = False):
    """Convert images in a folder into a gif.
    
    Parameters
    ---
    path
        the directory of the folder storing the images.
    remove
        after generating the :code:`.gif` image, whether remove the original static images or not.

    Example
    ---
    ::

        import UltraMotionCapture as umc
        umc.utils.images_to_gif(path="output/", remove=True)
    """
    files = os.listdir(path)
    files.sort()
    images = []

    for file in files:
        if ('png' in file or 'jpg' in file) and ('gif-' in file):
            images.append(imageio.imread(path + file))
            if remove:
                os.remove(path + file)

    if len(images) == 0:
        print("No images in folder")
    else:
        imageio.mimsave(path + 'output.gif', images)


def obj_pick_points(filedir: str, has_texture: bool = False, save_folder: str = 'output/', save_name: str = 'points'):
    """Manually pick points from 3D mesh loaded from a :code:`.obj` file. The picked points are stored in a (N, 3) :class:`numpy.array` and saved as :code:`.npy` :mod:`numpy` binary file.

    Parameters
    ---
    filedir
        The directory of the :code:`.obj` file.
    has_texture
        Whether the :code:`.obj` file has texture file or not. If set as :code:`True`, the texture will be loaded and rendered.
    save_folder
        The folder for saving :code:`.npy` binary file.

        Attention
        ---
        The folder directory shall be ended with :code:`/`, e.g. :code:`output/`.
    save_name
        The name of the saved :code:`.npy` binary file.

    Example
    ---
    One application of this function is preparing data for **calibration between 3dMD scanning system and the Vicon motion capture system**: Firstly, we acquire the markers' coordinates from the 3dMD scanning image. Then it can be compared with the Vicon data, leading to the reveal of the transformation parameters between two system's coordinates. ::

        import UltraMotionCapture as umc
        umc.utils.obj_pick_points(
            filedir='dataset/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj'
            has_texture=True,
            save_folder='conf/calibrate/',
            save_name='points_3dmd',
        )

    Dragging the scene to adjust perspective and clicking the marker points in the scene. Press :code:`q` to quite the interactive window and then the picked point's coordinates will be stored in a (N, 3) :class:`numpy.array` and saved as :code:`conf/calibrate/points_3dmd.npy`. Terminal will also print the saved :class:`numpy.array` for your reference.

        The remaining procedure to completed the calibration is realised in the following Jupyter notebook script:

        :code:`config/calibrate/calibrate_vicon_3dmd.ipynb`

    .. seealso::

        About the :code:`.npy` :mod:`numpy` binary file: 
        `numpy.save <https://numpy.org/doc/stable/reference/generated/numpy.save.html>`_ 
        `numpy.load <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_ 

        About point picking feature provided by the :mod:`pyvista` package: 
        `Picking a Point on the Surface of a Mesh - PyVista <https://docs.pyvista.org/examples/02-plot/surface-picking.html>`_
    """
    # load obj mesh with texture
    mesh = pv.read(filedir)

    if has_texture:
        texture = pv.read_texture(filedir.replace('.obj', '.jpg'))

    point_list = []

    # call back function for point picking
    def callback(point):
        # create a cube and a label at the click point
        mesh = pv.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
        pl.add_mesh(mesh, style='wireframe', color='r')
        pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])

        # store picked point
        point_list.append(np.expand_dims(point, axis=0))

    # launch point picking
    pl = pv.Plotter()
    
    if has_texture:
        pl.add_mesh(mesh, texture=texture, show_edges=True)
    else:
        pl.add_mesh(mesh, show_edges=True)

    pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=True)
    pl.show()

    # print picked points
    points = np.concatenate(point_list, axis=0)
    np.save(save_folder + save_name, points)
    print("save picked points:\n{}".format(points))