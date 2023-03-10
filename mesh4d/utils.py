from __future__ import annotations
from typing import Type, Union, Iterable

import os
import sys
import pickle
import imageio
import numpy as np
import pyvista as pv

import mesh4d.config.param
from mesh4d import kps

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

        import mesh4d as umc
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


def obj_pick_points(filedir: str, use_texture: bool = False, is_save: bool = False, save_folder: str = 'output/', save_name: str = 'points') -> np.array:
    """Manually pick points from 3D mesh loaded from a :code:`.obj` file. The picked points are stored in a (N, 3) :class:`numpy.array` and saved as :code:`.npy` :mod:`numpy` binary file.

    Parameters
    ---
    filedir
        The directory of the :code:`.obj` file.
    use_texture
        Whether use the :code:`.obj` file's texture file or not. If set as :code:`True`, the texture will be loaded and rendered.
    is_save
        save the points as local :code:`.npy` file or not. Default as :code:`False`.
    save_folder
        The folder for saving :code:`.npy` binary file.
    save_name
        The name of the saved :code:`.npy` binary file.

    Returns
    ---
    :class:`numpy.array`
        (N, 3) :class:`numpy.array` storing the picked points' coordinates.

    Example
    ---
    One application of this function is preparing data for **calibration between 3dMD scanning system and the Vicon motion capture system**: Firstly, we acquire the markers' coordinates from the 3dMD scanning image. Then it can be compared with the Vicon data, leading to the reveal of the transformation parameters between two system's coordinates. ::

        from mesh4d import utils

        utils.obj_pick_points(
            filedir='mesh4d/data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj',
            use_texture=True,
            is_save=True,
            save_folder='mesh4d/config/calibrate/',
            save_name='points_3dmd_test',
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
    # load obj mesh
    mesh = pv.read(filedir)
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
    
    if use_texture:
        texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
        pl.add_mesh(mesh, texture=texture, show_edges=True)
    else:
        pl.add_mesh(mesh, show_edges=True)

    pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=True)
    pl.camera_position = 'xy'
    pl.show()

    # save and return the picked points
    points = np.concatenate(point_list, axis=0)

    if is_save:
        np.save(os.path.join(save_folder, save_name), points)
        print("save picked points:\n{}".format(points))

    return points


def landmarks_labelling(
    mesh_folder: str,
    mesh_fps: int = 120,
    point_num: int = 1,
    stride: int = 1,
    is_save: bool = True,
    export_path: str = 'output/landmarks.pkl',
    ) -> kps.MarkerSet:
    """
    Label landmarks on a set of 3D meshes in the given folder.

    Parameters
    ----------
    mesh_folder : str
        The folder path containing the 3D meshes to label.
    mesh_fps : int, optional (default=120)
        The original frame rate of the mesh files.
    point_num : int, optional (default=1)
        The number of points to label on each mesh.
    stride : int, optional (default=1)
        The stride used to skip over meshes when labeling.
    is_save : bool, optional (default=True)
        Whether to save the labelled landmarks to disk.
    export_path : str, optional (default='output/landmarks.pkl')
        The file path to save the labelled landmarks to, if :code:`is_save` is True.

    Returns
    -------
    kps.MarkerSet
        A MarkerSet object containing the labelled landmarks.
    """
    files = os.listdir(mesh_folder)
    files = [os.path.join(mesh_folder, f) for f in files if '.obj' in f]
    files.sort()

    # landmarks labelling
    landmarks = kps.MarkerSet()
    landmarks.fps = mesh_fps/stride
    landmarks.scale_rate = 1
    landmarks.markers = {}

    for point_idx in range(point_num):
        point_name = 'marker {}'.format(point_idx)
        landmarks.markers[point_name] = kps.Marker(
            name=point_name,
            fps=landmarks.fps,
            scale_rate=landmarks.scale_rate,
        )

    file_idx = 0

    while file_idx < len(files):
        file = files[file_idx]
        print("labelling mesh file: {}".format(file))
        points = obj_pick_points(filedir=file, use_texture=True)

        if len(points) == point_num:
            # if successfully label point_num points
            # update file_idx to label the next mesh
            for point_idx in range(point_num):
                point_name = 'marker {}'.format(point_idx)
                landmarks.markers[point_name].append_data(points[point_idx], convert=False)
            
            print("extracted points:\n{}\n".format(points))
            file_idx = file_idx + stride

        else:
            # otherwise continue to label the same mesh
            # p.s. when realise that the order of labeling is wrong
            # we can press P immediately to break the labelling and trigger the relabelling procedure
            file_idx = file_idx
            

    # save landmarks object
    if is_save:
        save_pkl_object(landmarks, export_path)


def progress_bar(percent: float, bar_len: int = 20):
    """Print & refresh the progress bar in terminal.

    Parameters
    ---
    percent
        percentage from 0 to 1.
    bar_len
        length of the progress bar
    """
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.1%}".format("=" * int(bar_len * percent), bar_len, percent))
    sys.stdout.flush()
    # avoiding '%' appears when progress completed
    if percent == 1:
        print()


def save_pkl_object(obj, filepath: str):
    """Save an object to an :code:`.pkl` file.
    
    .. seealso::
        `Saving an Object (Data persistence) - StackOverflow <https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence>`_
    
    Parameters
    ---
    obj
        the object for storing locally.
    filepath
        the path to stored the :code:`.pkl` file.


    Example
    ---
    ::

         save_pkl_object(vicon, 'data/vkps/vicon.pkl')
    """
    with open(filepath, 'wb') as outp:  # overwrites any existing file
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filepath: str):
    """Load an object stored in :code:`.pkl` file.

    .. seealso::
        `Saving an Object (Data persistence) - StackOverflow <https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence>`_

    Parameters
    ---
    filepath
        the path of the :code:`.pkl` file.


    Example
    ---
    ::

         vicon = load_pkl_object('data/vkps/vicon.pkl')
    """
    with open(filepath, 'rb') as inp:
        return pickle.load(inp)