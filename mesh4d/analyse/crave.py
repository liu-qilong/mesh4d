from __future__ import annotations
from typing import Type, Union, Iterable

import os
import numpy as np
import pyvista as pv

import mesh4d
import mesh4d.config.param
from mesh4d import kps, utils, obj4d
from mesh4d.analyse import measure

def mesh_pick_points(
        filedir: str, 
        point_names: Union[Iterable[str], None] = None,
        use_texture: bool = False, 
        show_coord: bool = False, 
        show_edges: bool = False,
        is_save: bool = False, 
        data_type: str = '.obj',
        save_folder: str = 'output/', 
        save_name: str = 'points', 
        pre_points: Union[None, np.array] = None) -> np.array:
    """Manually pick points from 3D mesh loaded from a :code:`.obj` file. The picked points are stored in a (N, 3) :class:`numpy.array` and saved as :code:`.npy` :mod:`numpy` binary file.

    Parameters
    ---
    filedir
        The directory of the :code:`.obj` file.
    point_names
        tbf
    use_texture
        Whether use the :code:`.obj` file's texture file or not. If set as :code:`True`, the texture will be loaded and rendered.
    is_save
        save the points as local :code:`.npy` file or not. Default as :code:`False`.
    save_folder
        The folder for saving :code:`.npy` binary file.
    save_name
        The name of the saved :code:`.npy` binary file.
    pre_points
        tbf

    Returns
    ---
    :class:`numpy.array`
        (N, 3) :class:`numpy.array` storing the picked points' coordinates.

    Example
    ---
    One application of this function is preparing data for **calibration between 3dMD scanning system and the Vicon motion capture system**: Firstly, we acquire the markers' coordinates from the 3dMD scanning image. Then it can be compared with the Vicon data, leading to the reveal of the transformation parameters between two system's coordinates. ::

        from mesh4d import utils

        utils.mesh_pick_points(
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
    scene = pv.Plotter()

    if use_texture:
        texture = pv.read_texture(filedir.replace(data_type, '.jpg'))
        scene.add_mesh(mesh, texture=texture, show_edges=show_edges)
    else:
        scene.add_mesh(mesh, show_edges=show_edges)

    # function for drawing point label
    def draw_label(point_idx, point):
        if (point_names is not None) and (point_idx < len(point_names)):
            point_name = point_names[point_idx]
        else:
            point_name = point_idx

        if show_coord:
            label = [f"{point_name}@({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"]
        else:
            label = [f"{point_name}"]

        scene.add_point_labels(
            point, label, 
            font_size=30, point_size=30, shape='rounded_rect',
            point_color='goldenrod', render_points_as_spheres=True, always_visible=True,
            )

    # draw pre_points if exists
    if pre_points is not None:
        point_list = [list(point) for point in pre_points]
        
        for point_idx, point in enumerate(point_list):
            draw_label(point_idx, point)

    else:
        point_list = []

    # call back function for point picking
    def callback(point):
        point_idx = len(point_list)
        draw_label(point_idx, point)
        point_list.append(point)

    # launch point picking
    scene.enable_surface_point_picking(callback=callback, left_clicking=True, show_point=True)
    scene.camera_position = 'xy'
    scene.show()

    # save and return the picked points
    points = np.array(point_list)

    if is_save:
        np.save(os.path.join(save_folder, save_name), points)
        print("save picked points:\n{}".format(points))

    return points


def mesh_pick_points_with_check(
        file: str,
        point_names: str = None,
        **kwargs,
        ):
    """manual labeling of landmarks on a mesh sequence"""

    # labelling
    point_num = len(point_names)
    pre_points = None

    while(True):
        points = mesh_pick_points(filedir=file, point_names=point_names, pre_points=pre_points, **kwargs)

        if len(points) == point_num:
            # if successfully label point_num points, break loop
            
            print(f"extracted points:\n{points}")
            return points

        elif len(points) < point_num:
            # if the number of labelled points is less than point_num
            # relabel the same mesh with the last labelled points being undone
            print("undo the last labelled points")
            print("-"*20)
            pre_points = points[:-1]

        else:
            # otherwise continue to label the same mesh
            # with all labelled points being undone
            # p.s. when realizing that the order of labelling is wrong
            # we can press Q immediately to break the labelling and trigger the relabelling procedure
            print("undo all labelled points")
            pre_points = None


def markerset_labelling(
    mesh_folder: str,
    mesh_fps: int = 120,
    point_names: Iterable[str] = None,
    point_num: int = None,
    start: int = None,
    end: int = None,
    stride: int = 1,
    file_type: str = '.obj',
    is_save: bool = True,
    export_folder: str = 'output/',
    export_name: str = 'landmarks',
    **kwargs,
    ) -> kps.MarkerSet:
    """
    Label landmarks on a set of 3D meshes in the given folder.

    Tip
    ---
    When finished labelling a frame, press :code:`Q` to proceed. The algorithm will check if the point number is the same as :code:`point_num`. If matched, the labelling procedure will proceed to next frame of mesh , otherwise the same frame will be reopened for labelling.

    Therefore, if we realize that the locations or the order of labelling is wrong, we can click on random positions to make sure that the selected landmarks number doesn't match :code:`point_num`, and then press :code:`Q` to break the labelling and trigger the relabelling procedure of the same frame of mesh.

    Parameters
    ----------
    mesh_folder : str
        The folder path containing the 3D meshes to label.
    mesh_fps : int, optional (default=120)
        The original frame rate of the mesh files.
    point_names: 
        A list of names of the points to label.
    point_num : int, optional (default=1)
        The number of points to label on each mesh.

        Attention
        ---
        Give either :code:`point_num` or :code:`point_names`. If both are given, make sure that they are consistent.

    start: int, optional (default=0)
        begin loading from the :code:`start`-th image.
        
        Attention
        ---
        Index begins from 0. The :code:`start`-th image is included in the loaded images.
    end: int, optional (default=0)
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
    files = [os.path.join(mesh_folder, f) for f in files if file_type in f]
    files.sort()

    # default values
    if start == None:
        start = 0

    if end == None:
        end = len(files) - 1

    if point_names is None:
        point_names = [i for i in range(point_num)]

    if point_num is None:
        point_num = len(point_names)
    
    # landmarks labelling
    landmarks = kps.MarkerSet()
    landmarks.fps = mesh_fps/stride

    for point_idx in range(point_num):
        if point_names is not None:
            point_name = point_names[point_idx]

        else:
            point_name = point_idx

        landmarks.markers[point_name] = kps.Marker(name=point_name, fps=landmarks.fps)

    file_idx = start
    files_labeled = []

    for file_idx in range(start, end+1, stride):
        file = files[file_idx]
        files_labeled.append(file)
        print("-"*20)
        print(f"labelling mesh file: {file}")

        points = mesh_pick_points_with_check(filedir=file, point_names=point_names, **kwargs)

        for point_name in point_names:
            landmarks.markers[point_name].append_data(points[point_idx])
            
    # save landmarks object
    if is_save:
        utils.save_pkl_object(landmarks, export_folder, export_name)

    return landmarks, files_labeled


def fix_pvmesh_disconnect(mesh: pv.core.pointset.PolyData, selector_points: np.array = None) -> pv.core.pointset.PolyData:
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

    if selector_points is None:
        # if there is no selector points
        # return the body with maximum number of points 
        return bodies[max_index].extract_surface()
    
    else:
        # otherwise, return the body that is closest to the selector points
        dist_ls = []

        for body in bodies:
            nearest_points = measure.nearest_points_from_plane(body.extract_surface(), selector_points)
            dist_ls.append(np.linalg.norm(selector_points - nearest_points, axis=1).mean())

        min_index = dist_ls.index(min(dist_ls))
        return bodies[min_index].extract_surface()


def clip_meshes_with_contour(
    mesh_ls: Type[obj4d.Obj4d], 
    start_time: float, 
    contour: Type[kps.MarkerSet], 
    fps: float = None, 
    margin: float = 0, 
    invert: bool = False, 
    clip_bound: str = '',
    ) -> Iterable[pv.core.pointset.PolyData]:
    """Clip meshes with a given contour plane and bounding box.

    Parameters
    ---
    mesh_ls
        The list of meshes to be clipped.
    start_time
        The starting time to use for the clipping process.
    fps
        The frames per second used for the clipping process.
        If set as None, then the mesh and landmarks will be considered corresponding to each other frame-by-frame
    contour
        The contour (:class:`~mesh4d.kps.MarkerSet`) used to clip the meshes.
    margin
        The margin factor used to extend the bounding box. Defaults to 0.
    invert
        Whether to invert the clipping plane. Defaults to :code:`False`.
    clip_bound
        The clip bound axis used to clip the meshes. For example, :code:`xy` represents bounding the meshes with the highest and lowest positions of the contour in x and y axis. Defaults to ''.

    Returns
    ---
    Iterable[pv.core.pointset.PolyData]
        The list of clipped meshes.
    """
    mesh_clip_ls = []

    for idx in range(len(mesh_ls)):
        mesh = mesh_ls[idx]

        # estimate contour plane
        if fps is None:
            contour_points = contour.get_frame_coord(idx).get_points_coord()
        else:
            time = start_time + idx / fps
            contour_points = contour.get_time_coord(time).get_points_coord()

        mesh_clip = clip_mesh_with_contour(mesh, contour_points, margin, invert, clip_bound)
        mesh_clip_ls.append(mesh_clip)

    return mesh_clip_ls

def clip_mesh_with_contour(
    mesh: pv.core.pointset.PolyData, 
    contour_points: np.array,   # (N, 3) points
    margin: float = 0, 
    invert: bool = False, 
    clip_bound: str = '',
    ) -> Iterable[pv.core.pointset.PolyData]:
    """TK"""
    # estimate contour plane
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
    max_margin = margin * (max_bound - center) / np.linalg.norm(max_bound - center)
    min_margin = margin * (center - min_bound) / np.linalg.norm(center - min_bound)

    for bound_symbol in clip_bound:
        mesh_clip = mesh_clip.clip(bound_symbol, origin=max_bound + max_margin, invert=True)
        mesh_clip = mesh_clip.clip(bound_symbol, origin=min_bound - min_margin, invert=False)

    # remove disconnected parts and return
    return fix_pvmesh_disconnect(mesh_clip)

def clip_mesh_with_plane(
    mesh: pv.core.pointset.PolyData, 
    norm: np.array,
    center: np.array,
    margin: float = 0, 
    invert: bool = False,
    ) -> Iterable[pv.core.pointset.PolyData]:
    """TK"""
    # clip the mesh with contour plane
    mesh_clip = mesh.clip(
        norm, 
        origin=center - margin * norm,
        invert=invert,
        )

    # remove disconnected parts and return
    return fix_pvmesh_disconnect(mesh_clip)