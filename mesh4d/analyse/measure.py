from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np
import pyvista as pv

import mesh4d
import mesh4d.config.param
from mesh4d import kps

def points_get_center(points: np.array) -> np.array:
    """Get the center point of a set of points.
    
    The center point is defined as the geometric average point of all points:

    .. math::
        \\boldsymbol c = \\frac{\sum_{i} \\boldsymbol p_i}{N}

    where :math:`N` denotes the total number of points.

    Parameters
    ---
    points
        the points' coordinates stored in a (N, 3) :class:`numpy.array`.

    Return
    ---
    :class:`numpy.array`
        The center point coordinates represented in a (3, ) :class:`numpy.array`.
    """
    return np.mean(points, 0)


def points_get_max_bound(points: np.array) -> np.array:
    """Get the maximum boundary of a set of points.

    Parameters
    ---
    points
        the points' coordinates stored in a (N, 3) :class:`numpy.array`.

    Return
    ---
    :class:`numpy.array`
        a list containing the maximum value of :math:`x, y, z`-coordinates: :code:`[max_x, max_y, max_z]`.
    """
    return np.ndarray.max(points, 0)


def points_get_min_bound(points: np.array) -> np.array:
    """Get the minimum boundary of a set of points.

    Parameters
    ---
    points
        the points' coordinates stored in a (N, 3) :class:`numpy.array`.

    Return
    ---
    :class:`numpy.array`
        a list containing the minimum value of :math:`x, y, z`-coordinates: :code:`[min_x, min_y, min_z]`.
    """
    return np.ndarray.min(points, 0)


def search_nearest_points_plane(mesh: pv.core.pointset.PolyData, points: np.array) -> np.array:
    """tbf"""
    _, closest_points = mesh.find_closest_cell(points, return_closest_point=True)
    return closest_points


def estimate_plane_from_points(points: np.array) -> tuple:
    """Estimate the approximate plane contains the points.
    
    Parameters
    ---
    points
        points coordinates stored in (N, 3) array.

    Returns
    ---
    :class:`numpy.array`
        normal vector of the plane
    :class:`numpy.array`
        one point of the plane


    Note
    ---
    Denote the contour points as :math:`\\boldsymbol p_i, i = 0, 1, ..., N`. There mean coordinate is regarded as their center point :math:`\\boldsymbol c`:

    .. math::
        \\boldsymbol c = \sum_i^N \\boldsymbol p_i

    Then, an ideal plane that contains all of them should have a normal vector :math:`\\boldsymbol n` that satisfies:
    
    .. math::
        \\begin{cases}
        (\\boldsymbol p_i - \\boldsymbol c)^T \\boldsymbol n = 0, i = 1, 2, ..., N \\\\
        \|\\boldsymbol n\|_2 = 1
        \end{cases}

    Noted that the last equation is not linear. It's then be replace with linear constrain :math:`\|\\boldsymbol n\|_1 = 1`. Then the norm vector of the plane can be solved with least-squares method. Together with the center point :math:`\\boldsymbol c`, the (approximate) contour plane is estimated.
    """
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
    """Get the trace length information.

    Parameters
    ---
    marker
        the marker object
    start_frame
        the start frame index. Noted that frame index starts at 0.
    end_frame
        the end frame index. Default as :code:`None`, meaning using the last frame as the end frame.
    
    Return
    ---
    A dictionary contains:

    :code:`'disp'` a list of displacement vectors of each frame.
    :code:`'dist'` a list of displacement distance of each frame.
    :code:`'trace'` the sum of trace length.
    """
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
    """Get the trace length information.

    Parameters
    ---
    Parameters
    ---
    marker
        the marker set object
    start_frame
        the start frame index. Noted that frame index starts at 0.
    end_frame
        the end frame index. Default as :code:`None`, meaning using the last frame as the end frame.
    
    Return
    ---
    :class:`dict`
        a dictionary of the trace dictionary of each marker. See :meth:`Marker.get_trace_length`.
    :class:`list`
        a list of the start points of each marker.
    :class:`list`
        a list of whole period trace length of each marker.
    """
    trace_dict = {}

    for name, marker in markerset.markers.items():
        trace_dict[name] = marker_trace_length(marker, start_frame, end_frame)

    starts = []
    traces = []

    for name in markerset.markers.keys():
        starts.append(markerset.markers[name].get_frame_coord(start_frame))
        traces.append(trace_dict[name]['trace'])

    return trace_dict, starts, traces