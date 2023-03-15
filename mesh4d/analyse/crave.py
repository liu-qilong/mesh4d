from __future__ import annotations
from typing import Type, Union, Iterable

import os
import numpy as np
import pyvista as pv

import mesh4d
import mesh4d.config.param
from mesh4d import kps, utils, obj4d
from mesh4d.analyse import measure

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
        `Picking a Point on the Surface of a Mesh - PyVista <https://docs.pyvista.org/examples/02-sceneot/surface-picking.html>`_
    """
    # load obj mesh
    mesh = pv.read(filedir)
    point_list = []

    # call back function for point picking
    def callback(point):
        # point id
        point_id = len(point_list)

        # create a cube and a label at the click point
        mesh = pv.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
        scene.add_mesh(mesh, style='wireframe', color='r')
        scene.add_point_labels(point, ["#{} ({:.2f}, {:.2f}, {:.2f})".format(point_id, point[0], point[1], point[2])])

        # store picked point
        point_list.append(np.expand_dims(point, axis=0))

    # launch point picking
    scene = pv.Plotter()
    
    if use_texture:
        texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
        scene.add_mesh(mesh, texture=texture, show_edges=True)
    else:
        scene.add_mesh(mesh, show_edges=True)

    scene.enable_surface_picking(callback=callback, left_clicking=True, show_point=True)
    scene.camera_position = 'xy'
    scene.show()

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
    start: int = 0,
    end: int = 1,
    stride: int = 1,
    is_save: bool = True,
    export_folder: str = 'output/',
    export_name: str = 'landmarks',
    ) -> kps.MarkerSet:
    """
    Label landmarks on a set of 3D meshes in the given folder.

    Tip
    ---
    When finished labelling a frame, press :code:`P` to proceed. The algorithm will check if the point number is the same as :code:`point_num`. If matched, the labelling procedure will proceed to next frame of mesh , otherwise the same frame will be reopened for labelling.

    Therefore, if we realise that the locations or the order of labelling is wrong, we can click on random positions to make sure that the selected landmarks number doesn't match :code:`point_num`, and then press :code:`P` to break the labelling and trigger the relabelling procedure of the same frame of mesh.

    Parameters
    ----------
    mesh_folder : str
        The folder path containing the 3D meshes to label.
    mesh_fps : int, optional (default=120)
        The original frame rate of the mesh files.
    point_num : int, optional (default=1)
        The number of points to label on each mesh.
    start: int, optional (default=0)
        begin loading from the :code:`start`-th image.
        
        Attention
        ---
        Index begins from 0. The :code:`start`-th image is included in the loaded images.
        Index begins from 0.
    end: int, optional (default=1)
        end loading at the :code:`end`-th image.
        
        Attention
        ---
        Index begins from 0. The :code:`end`-th image is included in the loaded images.
    stride : int, optional (default=1)
        The stride used to skip over meshes when labelling.
    is_save : bool, optional (default=True)
        Whether to save the labelled landmarks to disk.
    export_folder : str, optional (default='output/')
        The folder to save the labelled landmarks to, if :code:`is_save` is True.
    export_name : str, optional (default='landmarks')
        The filename to save the labelled landmarks, if :code:`is_save` is True.

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

    for point_idx in range(point_num):
        point_name = 'marker {}'.format(point_idx)
        landmarks.markers[point_name] = kps.Marker(name=point_name, fps=landmarks.fps)

    file_idx = start

    while file_idx <= end:
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
            # p.s. when realising that the order of labelling is wrong
            # we can press P immediately to break the labelling and trigger the relabelling procedure
            file_idx = file_idx
            
    # save landmarks object
    if is_save:
        utils.save_pkl_object(landmarks, export_folder, "{}.pkl".format(export_name))


def fix_pvmesh_disconnect(mesh: pv.core.pointset.PolyData) -> pv.core.pointset.PolyData():
    """Fix disconnection problem in :mod:`pyvista` mesh.

    - Split the mesh into variously connected meshes.
    - Return the connected mesh with biggest point number.

    Parameters
    ---
    mesh
        :mod:`pyvista` mesh.

    Returns
    ---
    :mod:`pyvista`
        the fully connected mesh.
    """
    # split the mesh into different bodies according to the connectivity
    clean = mesh.clean()
    bodies = clean.split_bodies()

    # get the index of body with maximum number of points 
    point_nums = [len(body.points) for body in bodies]
    max_index = point_nums.index(max(point_nums))

    # return the body with maximum number of points 
    return bodies[max_index].extract_surface()


def clip_with_contour(mesh_ls: Type[obj4d.Obj4d], start_time: float, fps: float, contour: Type[kps.MarkerSet], margin: float = 0, invert: bool = False, clip_bound: str = '') -> Iterable[pv.core.pointset.PolyData]:
    """tbf"""
    mesh_clip_ls = []

    for idx in range(len(mesh_ls)):
        mesh = mesh_ls[idx]

        # estimate contour plane
        time = start_time + idx / fps
        contour_points = contour.get_time_coord(time).get_points_coord()
        norm, center = measure.estimate_plane_from_points(contour_points)

        # clip the mesh with contour plane
        mesh_clip = mesh.clip(
            norm, 
            origin=center - margin * norm,
            invert=invert,
            )
        
        # estimate contour bound
        max_bound = measure.points_get_max_bound(contour_points)
        min_bound = measure.points_get_min_bound(contour_points)
        max_margin = margin * (max_bound - center)
        min_margin = margin * (center - min_bound)

        for bound_symbol in clip_bound:
            mesh_clip = mesh_clip.clip(bound_symbol, origin=max_bound + max_margin, invert=True)
            mesh_clip = mesh_clip.clip(bound_symbol, origin=min_bound - min_margin, invert=False)

        # remove disconnected parts
        mesh_clip = fix_pvmesh_disconnect(mesh_clip)
        mesh_clip_ls.append(mesh_clip)

    return mesh_clip_ls