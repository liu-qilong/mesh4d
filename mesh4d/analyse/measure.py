from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

import mesh4d
import mesh4d.config.param
from mesh4d import kps
from mesh4d.analyse import crave


def points_get_center(points: np.array) -> np.array:
    return np.mean(points, 0)


def points_get_max_bound(points: np.array) -> np.array:
    return np.ndarray.max(points, 0)


def points_get_min_bound(points: np.array) -> np.array:
    return np.ndarray.min(points, 0)


def nearest_points_from_plane(mesh: pv.core.pointset.PolyData, points: np.array) -> np.array:
    _, closest_points = mesh.find_closest_cell(points, return_closest_point=True)
    return closest_points


def estimate_plane_from_points(points: np.array) -> tuple:
    center = points_get_center(points)

    # synthesis A matrix 
    A_up = points - center
    A_down = np.expand_dims(
        np.ones(len(center)), 
        axis=0
        )
    A = np.concatenate((A_up, A_down))

    # synthesis b vector
    b_up = np.zeros(len(points))
    b_down = np.ones(1)
    b = np.concatenate((b_up, b_down))

    # solve norm vector of the plane $\bm n$ with least-squares method
    norm = np.linalg.lstsq(A, b)[0]
    norm = norm / np.linalg.norm(norm)

    return norm, center


def marker_trace_length(marker: kps.Marker, start_frame: int = 0, end_frame: Union[int, None] = None) -> dict:
    coord_clip = marker.coord[:, start_frame:end_frame]
    coord_front = coord_clip[:, :-1]
    coord_back = coord_clip[:, 1:]

    disp = coord_front - coord_back
    dist = np.linalg.norm(disp, axis=0)
    trace = np.sum(dist)

    trace_dict = {}
    trace_dict['disp'] = disp
    trace_dict['dist'] = dist
    trace_dict['trace'] = trace

    return trace_dict


def markerset_trace_length(markerset: kps.MarkerSet, start_frame: int = 0, end_frame: Union[int, None] = None) -> tuple:
    trace_dict = {}

    for name, marker in markerset.markers.items():
        trace_dict[name] = marker_trace_length(marker, start_frame, end_frame)

    starts = []
    traces = []

    for name in markerset.markers.keys():
        starts.append(markerset.markers[name].get_frame_coord(start_frame))
        traces.append(trace_dict[name]['trace'])

    return trace_dict, starts, traces


def mesh_density(mesh):
    mesh = crave.fix_pvmesh_disconnect(mesh)
    tree = KDTree(mesh.points)
    d, _ = tree.query(mesh.points, k=3)
    print("{:.2f} \pm {:.2f}".format(np.mean(d[:, 1]), np.std(d[:, 1])))