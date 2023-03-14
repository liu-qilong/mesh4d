from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np

import mesh4d
import mesh4d.config.param

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