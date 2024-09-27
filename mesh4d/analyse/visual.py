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
    # create many spheres from the point cloud
    pdata = pv.PolyData(points)
    pdata['orig_sphere'] = np.arange(len(points))
    sphere = pv.Sphere(**kwargs)
    pvpcd = pdata.glyph(scale=False, geom=sphere, orient=False)
    return pvpcd


def show_obj3d_diff(obj1: Type[obj3d.Obj3d], obj2: Type[obj3d.Obj3d], kps_names: Union[None, tuple, list] = None, cmap: str = "cool", op2: float = 0.2, off_screen: bool = False) -> pv.Plotter:
    scene = pv.Plotter(off_screen=off_screen)
    
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
    scene.show(interactive_update=True)

    return scene


def show_mesh_value_mask( mesh: pv.core.pointset.PolyData, points: Iterable, values: Iterable, k_nbr: int = 10, max_threshold: Union[float, None] = None, min_threshold: Union[float, None] = None, cmap: str = "cool", off_screen: bool = False, is_export: bool = False, export_folder: str = '', export_name: str = 'screeenshot', **kwargs) -> pv.Plotter:
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

    scene = pv.Plotter(off_screen=off_screen)
    scene.add_mesh(mesh, cmap=cmap, **kwargs)
    scene.camera_position = 'xy'

    if is_export:
        export_path = os.path.join(export_folder, f'{export_name}.png')
        scene.show(screenshot=export_path, interactive_update=True)

        if mesh4d.output_msg:
            print("export image: {}".format(export_path))

    else:
        scene.show(interactive_update=True)

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

    return scene