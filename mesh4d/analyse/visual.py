from __future__ import annotations
from typing import Type, Union, Iterable

import os
import copy
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d

def np2pvpcd(points: np.array, **kwargs) -> pv.core.pointset.PolyData:
    """Transform the points coordinates stored in a :class:`numpy.array` to a a :mod:`pyvista` point cloud (:class:`pyvista.PolyData`).

    Parameters
    ---
    points
        the points coordinates data stored in a (N, 3) :class:`numpy.array`.
        
    Return
    ---
    :class:`pyvista.PolyData`
        the point cloud (:class:`pyvista.PolyData`).

    Attention
    ---
    Acutally, :mod:`pyvista` package doesn't have a specific class to represent point cloud. The returned :class:`pyvista.PolyData` object is a point collection mainly for illustration purpose.

    Tip
    ---
    More configuration parameters can be passed in via :code:`**kwargs`. Please refer to `pyvista.PolyData <https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.html#pyvista.PolyData>`_ for the accepted parameters.
    """
    # create many spheres from the point cloud
    pdata = pv.PolyData(points)
    pdata['orig_sphere'] = np.arange(len(points))
    sphere = pv.Sphere(**kwargs)
    pvpcd = pdata.glyph(scale=False, geom=sphere, orient=False)
    return pvpcd


def show_obj3d_diff(obj1: Type[obj3d.Obj3d], obj2: Type[obj3d.Obj3d], kps_names: Union[None, tuple, list] = None, cmap: str = "cool", op2: float = 0.2):
    """Illustrate the difference of two 3D object:

    :attr:`obj1` mesh will be coloured according to each of its points' distance to :attr:`obj2` mesh. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
    
    Parameters
    ---
    obj1
        the first 3D object.
    obj2
        the second 3D object.
    kps_names
        a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
    cmap
        the color map name. 
        
        .. seealso::
            For full list of supported color map, please refer to `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
    op2
        the opacity of the second 3D object.
    """
    scene = pv.Plotter()
    
    tree = KDTree(obj2.mesh.points)
    d_kdtree, _ = tree.query(obj1.mesh.points)
    obj1.mesh["distances"] = d_kdtree

    obj1.add_mesh_to_scene(scene, show_edges=False, cmap=cmap)
    obj2.add_mesh_to_scene(scene, show_edges=False, opacity=op2)

    # if 3d objects are derived from Obj3d_Kps, also show the key points attached to them
    if isinstance(obj1, obj3d.Obj3d_Kps):
        width = obj1.get_width()
        obj1.add_kps_to_scene(scene, kps_names, color="gold", radius=0.02*width)

    if isinstance(obj2, obj3d.Obj3d_Kps):
        width = obj1.get_width()
        obj2.add_kps_to_scene(scene, kps_names, color="green", radius=0.02*width)

    scene.camera_position = 'xy'
    scene.show()


def show_mesh_value_mask(mesh: pv.core.pointset.PolyData, points: Iterable, values: Iterable, k_nbr: int = 10, distance_upper_bound: float = 50, max_threshold: Union[float, None] = None, min_threshold: Union[float, None] = None, cmap: str = "cool", is_save: bool = False, export_folder: str = '', export_name: str = 'screeenshot', **kwargs):
    """Show the 3D mesh with a value mask.

    Parameters
    ----------
    mesh : pyvista.core.pointset.PolyData
        The mesh to be shown with value mask.
    points : Iterable
        An iterable containing the points to use for assigning values to the mesh vertices.
    values : Iterable
        An iterable containing the values to assign to the mesh vertices based on their nearest point in `points`.
    k_nbr : int, optional
        The number of nearest neighbors to consider when assigning values to the mesh vertices. Default is 10.
    distance_upper_bound
        Maximum distance when masking the values to the mesh cell. Default is 50.
    max_threshold : float or None, optional
        The maximum value to include in the mask. Any values greater than this threshold will be replaced with the threshold value. Default is None.
    min_threshold : float or None, optional
        The minimum value to include in the mask. Any values less than this threshold will be replaced with the threshold value. Default is None.
    cmap
        the color map name. 
        
        .. seealso::
            For full list of supported color map, please refer to `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
    is_save
        weather save the generated figure or not.
    export_folder
        The folder to save the figure.
    export_name
        The filename to save the figure.

    **kwargs
        arguments to be passed to :meth:`pyvista.Plotter.add_mesh`.

        .. seealso::
            `pyvista.Plotter.add_mesh <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh>`_
    """
    mesh = copy.deepcopy(mesh)

    # assign each vertex with a value based on rbf interpolation
    value_field = RBFInterpolator(points, values, neighbors=k_nbr)
    value_mask = value_field(mesh.points)

    # filter out the values out of the threshold
    if mesh4d.output_msg:
        print("original value range: {} - {}".format(np.min(value_mask), np.max(value_mask)))

    for idx in range(len(value_mask)):
        if min_threshold is not None:
            if value_mask[idx] < min_threshold:
                value_mask[idx] = min_threshold

        if max_threshold is not None:
            if value_mask[idx] > max_threshold:
                value_mask[idx] = max_threshold

    if mesh4d.output_msg:
        print("after thresholding: {} - {}".format(np.min(value_mask), np.max(value_mask)))

    # plot the mesh mask with the values
    mesh["distances"] = value_mask

    scene = pv.Plotter()
    scene.add_mesh(mesh, cmap=cmap, **kwargs)
    scene.camera_position = 'xy'

    if is_save:
        export_path = os.path.join(export_folder, '{}.png'.format(export_name))
        scene.show(screenshot=export_path)
        if mesh4d.output_msg:
            print("export image: {}".format(export_path))

    else:
        scene.show()

    # filter out the values out of the threshold
    if mesh4d.output_msg:
        print("original value range: {} - {}".format(np.min(value_mask), np.max(value_mask)))

    for idx in range(len(value_mask)):
        if min_threshold is not None:
            if value_mask[idx] < min_threshold:
                value_mask[idx] = min_threshold

        if max_threshold is not None:
            if value_mask[idx] > max_threshold:
                value_mask[idx] = max_threshold

    if mesh4d.output_msg:
        print("after thresholding: {} - {}".format(np.min(value_mask), np.max(value_mask)))