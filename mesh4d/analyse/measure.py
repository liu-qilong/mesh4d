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


def estimate_plane_from_points(points: np.array) -> tuple:
    """Estimate the approximate plane contains the points.
    
    Parameters
    ---
    points
        points coordinates stored in (N, 3) array.

    Note
    ---
    Note the contour points as :math:`\\boldsymbol p_i, i = 0, 1, ..., N`. There mean coordinate is regarded as their center point :math:`\\boldsymbol c`:

    .. math::
        \\boldysmbol c = \sum_i^N \\boldsymbol p_i

    Then, an ideal plane that contains all of them should have a normal vector :math:`\\boldsymbol c` that satisfies:
    
    .. math::
        \\begin{cases}
        (\\boldsymbol p_i - \\boldsymbol c)^T \\boldsymbol n = 0, i = 1, 2, ..., N\\
        \|\\boldsymbol n\|_2 = 1
        \end{cases}

    Noted that the last equation is not linear. It's then be replace with linear constrain :math:`\|\\boldsymbol n\|_1 = 1`. Then the norm vector of the plane can be solved with least-squares method. Together with the center point $p_{center}$, the (approximate) contour plane is estimated.
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