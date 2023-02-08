"""The 3dMD 4D scanning device can record 3D images series in very high time- and space-resolution, which provides very rich information of the dynamic movement and deformation of human body during active activities. However, there is a crucial lack of inner-relationship information between different frames of 3D image:

.. important::
    For example, with the 5-th and 6-th frames of 3D images, we know that the former one transforms to the next one. However, for any specific point on the 5-th frame, we don't know which point in the 6-th frame it transfers to.

    *Such lack of information blocks the way of any sophisticated and thematic analysis of the 4D data*, such as tracing the movement of the nipple points and tracking the variation of the upper arm area during some kind of sports activity.

The :mod:`UltraMotionCapture.field` aims at revealing the so-called inner-relationship information between different frames. In the context of mathematical, the most meticulous level of such information can be represented as *displacement field* and other kinds of transformation. Actually, the whole :mod:`UltraMotionCapture` project is motivated and centred around this bottleneck problem.
"""

from __future__ import annotations
from typing import Type, Union, Iterable

import copy
import numpy as np
from probreg import cpd

import UltraMotionCapture
import UltraMotionCapture.config.param
from UltraMotionCapture import obj3d

class Trans(object):
    """The base class of transformation. Different types of transformation, such as rigid and non-rigid transformation, are further defined in the children classes like :class:`Trans_Rigid` and :class:`Trans_Nonrigid`.

    Parameters
    ---
    source_obj
        The source 3D object of the transformation. Any object of the class derived from :class:`UltraMotionCapture.obj3d.Obj3d` is valid.
    target_obj
        The target 3D object of the transformation. Any object of the class derived from :class:`UltraMotionCapture.obj3d.Obj3d` is valid.

    Note
    ---
    `Class Attributes`

    self.source
        The source point cloud (:class:`open3d.geometry.PointCloud`) of the transformation.
    self.target
        The target point cloud (:class:`open3d.geometry.PointCloud`) of the transformation.
    """
    def __init__(self, source_obj: Type[obj3d.Obj3d], target_obj: Type[obj3d.Obj3d], **kwargs):
        self.source = source_obj.pcd
        self.target = target_obj.pcd


class Trans_Rigid(Trans):
    """The rigid transformation, which can be expressed in the form of :math:`\mathcal{T}`:
    
    .. math:: \mathcal{T}(\\boldsymbol x) = s \\boldsymbol R \\boldsymbol x + \\boldsymbol t
    
    where :math:`s \in \mathbb R`, :math:`\\boldsymbol R \in \mathbb R^{3 \\times 3}`, :math:`\\boldsymbol t \in \mathbb R^{3}`, :math:`\\boldsymbol x \in \mathbb R^{3}` stand for the scaling rate, the rotation matrix, the translation vector, and an arbitrary point under transformation, respectively.

    Note
    ---
    `Class Attributes`

    self.scale
        the scaling rate :math:`s`.
    self.rot
        the rotation matrix :math:`\\boldsymbol R`.
    self.t
        the translation vector :math:`\\boldsymbol t`.
    
    Attention
    ---
    After initialisation, the registration method :meth:`regist` must be called to estimate the rigid transformation between the source and target point cloud.

    Example
    ---
    After loading and registration, the rigid transformation parameters can then be accessed, including the scaling rate, the rotation matrix, and the translation vector: ::

        from UltraMotionCapture import obj3d

        o3_1 = obj3d.Obj3d('data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj')
        o3_2 = obj3d.Obj3d('data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000002.obj')

        trans = field.Trans_Rigid(o3_1, o3_2)
        trans.regist()
        print(trans.scale, trans.rot, trans.t)
    """
    def regist(self, method=cpd.registration_cpd, **kwargs):
        """The registration method.

        Parameters
        ---
        method
            At current stage, only methods from :mod:`probreg` package are supported. Default as :func:`probreg.cpd.registration_cpd`.
        **kwargs
            Configurations parameters of the registration.
            
            See Also
            --------
            `probreg.cpd.registration_cpd <https://probreg.readthedocs.io/en/latest/probreg.html?highlight=registration_cpd#probreg.cpd.registration_cpd>`_
        """
        tf_param, _, _ = method(
            self.source, self.target, 'rigid', **kwargs
        )
        self.__parse(tf_param)
        self.__fix()
        
        if UltraMotionCapture.output_msg:
            print("registered 1 rigid transformation")

    def __parse(self, tf_param: Type[cpd.CoherentPointDrift]):
        """Parse the registration result to provide :attr:`self.s`, :attr:`self.rot`, and :attr:`self.t`. Called by :meth:`regist`.
        
        Parameters
        ---
        tf_param
            Attention
            ---
            At current stage, the default registration method is Coherent Point Drift (CPD) method realised by :mod:`probreg` package. Therefore the accepted transformation object to be parse is derived from :class:`cpd.CoherentPointDrift`. Transformation object provided by other registration method shall be tested in future development.
        """
        self.rot = tf_param.rot
        self.scale = tf_param.scale
        self.t = tf_param.t

    def __fix(self):
        """Fix the registration result. Called by :meth:`regist`.
        
        Attention
        ---
        At current stage, the fixing logic only checks the scaling rate and raises a warning in terminal. The underline assumption is that since :mod:`UltraMotionCapture` focuses on human body which doesn't scale a lot, the scaling rate shall be closed to 1.
        """
        if np.abs(self.scale - 1) > 0.05 and UltraMotionCapture.output_msg:
            print("warnning: large rigid scale {}".format(self.scale))

    def shift_points(self, points: np.array) -> np.array:
        """Implement the transformation to set of points.

        Parameters
        ---
        points
            :math:`N` points in 3D space that we want to implement the transformation on. Stored in a (N, 3) :class:`numpy.array`.

        Return
        ---
        :class:`numpy.array`
            (N, 3) :class:`numpy.array` stores the points after transformation.

        Warning
        ---
        This method will be realised in future development.
        """
        pass

    def show(self):
        """Illustrate the estimated transformation.

        Warning
        ---
        This method will be realised in future development.
        """
        pass


class Trans_Nonrigid(Trans):
    """The non-rigid transformation, under which points in different locations may be transformed in different directions and distances. Such an idea can be expressed in the form of :math:`\mathcal{T}`:
    
    .. math:: \mathcal{T}(\\boldsymbol S) = \\boldsymbol S + \\boldsymbol T
    
    where :math:`\\boldsymbol S \in \mathbb R^{N \\times 3}` and :math:`\\boldsymbol T \in \mathbb R^{N \\times 3}` stand for the original point cloud and the translation matrix, all stored in the form of :math:`N \\times 3` matrix.

    Note
    ---
    `Class Attributes`
    
    self.source_points
        The source points :math:`\\boldsymbol S \in \mathbb R^{N \\times 3}` stored in (N, 3) :class:`numpy.array`.
    self.deform_points
        The deformed points :math:`\\boldsymbol S + \\boldsymbol T \in \mathbb R^{N \\times 3}` stored in (N, 3) :class:`numpy.array`.
    self.disp
        The displacement matrix :math:`\\boldsymbol T \in \mathbb R^{N \\times 3}` stored in (N, 3) :class:`numpy.array`.
    
    Attention
    ---
    After initialisation, the registration method :meth:`regist` must be called to estimate the non-rigid transformation between the source and target point cloud.

    Example
    ---
    After loading and registration, the rigid transformation parameters can then be accessed, including the scaling rate, the rotation matrix, and the translation vector: ::

        from UltraMotionCapture import obj3d, field

        o3_1 = obj3d.Obj3d('data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj')
        o3_2 = obj3d.Obj3d('data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000002.obj')

        trans = field.Trans_Nonrigid(o3_1, o3_2)
        trans.regist()
        print(trans.deform_points, trans.disp)
    """
    def regist(self, method=cpd.registration_cpd, **kwargs):
        """The registration method.

        Parameters
        ---
        method
            At current stage, only methods from :mod:`probreg` package are supported. Default as :func:`probreg.cpd.registration_cpd`.
        **kwargs
            Configurations parameters of the registration.
            
            See Also
            --------
            `probreg.cpd.registration_cpd <https://probreg.readthedocs.io/en/latest/probreg.html?highlight=registration_cpd#probreg.cpd.registration_cpd>`_
        """
        tf_param, _, _ = method(
            self.source, self.target, 'nonrigid', **kwargs
        )
        self.__parse(tf_param)
        self.__fix()
        
        if UltraMotionCapture.output_msg:
            print("registered 1 nonrigid transformation")

    def __parse(self, tf_param):
        """Parse the registration result to provide :attr:`self.source_points`, :attr:`self.deform_points`, and :attr:`self.disp`. Called by :meth:`regist`.
        
        Parameters
        ---
        tf_param
            Attention
            ---
            At current stage, the default registration method is Coherent Point Drift (CPD) method realised by :mod:`probreg` package. Therefore the accepted transformation object to be parse is derived from :class:`cpd.CoherentPointDrift`. Transformation object provided by other registration method shall be tested in future development.
        """
        deform = copy.deepcopy(self.source)
        deform.points = tf_param.transform(deform.points)
        
        self.deform_points = obj3d.pcd2np(deform)
        self.source_points = obj3d.pcd2np(self.source)
        self.disp = self.deform_points - self.source_points

    def __fix(self):
        """Fix the registration result. Called by :meth:`regist`.
        
        Attention
        ---
        At current stage, the fixing logic aligns the deformed points to their closest points in the target point cloud, to avoid distortion effect after long-chain registration procedure. This logic may be discarded or replaced by better scheme in future development.
        tbf
        """
        pass
    
        """
        deform_fix_points = []
        target_points = obj3d.pcd2np(self.target)

        for n in range(len(self.deform_points)):
            deform_fix_points.append(
                obj3d.search_nearest_point(self.deform_points[n], target_points)
            )

        self.deform_points = deform_fix_points
        self.disp = self.deform_points - self.source_points
        """

    def shift_points(self, points: np.array) -> np.array:
        """Implement the transformation to set of points.

        To apply proper transformation to an arbitrary point :math:`\\boldsymbol x`:

        - Find the closest point :math:`\\boldsymbol s_{\\boldsymbol x}` and its displacement :math:`\\boldsymbol t_{\\boldsymbol x}`.
        - Use :math:`\\boldsymbol t_{\\boldsymbol x}` as :math:`\\boldsymbol x`'s displacement: :math:`\\boldsymbol x' = \\boldsymbol x + \\boldsymbol t_{\\boldsymbol x}`

        Warning
        ---
        This logic may be replaced by better scheme in future development.

        Parameters
        ---
        points
            :math:`N` points in 3D space that we want to implement the transformation on. Stored in a (N, 3) :class:`numpy.array`.

        Return
        ---
        :class:`numpy.array`
            (N, 3) :class:`numpy.array` stores the points after transformation.
        """
        points_shift = []
        for point in points:
            idx = obj3d.search_nearest_point_idx(point, self.source_points)
            points_shift.append(self.deform_points[idx])
        return np.array(points_shift)

    def shift_disp_dist(self, points: np.array) -> Iterable[np.array, np.array]:
        """Evaluate the displacement and distance of the transformation implemented to a set of points.

        Parameters
        ---
        points
            :math:`N` points in 3D space that we want to implement the transformation on. Stored in a (N, 3) :class:`numpy.array`.

        Return
        ---
        :class:`numpy.array`
            the displacement vectors stored in (N, 3) array.
        :class:`numpy.array`
            the displacement distances stored in (N, ) array.
        """
        points_deform = self.shift_points(points)
        disp = points_deform - points
        dist = np.linalg.norm(disp, axis=1)
        return disp, dist

def transform_rst2sm(R: np.array, s: float, t: np.array) -> tuple[float, np.array]:
    """Transform rigid transformation representation from
        
        rotation matrix :math:`\\boldsymbol R \in \mathbb{R}^{3 \\times 3}`, scaling rate :math:`s \in \mathbb{R}`, and translation vector :math:`\\boldsymbol t \in \mathbb{R}^{3}`
    
    to
    
        homogeneous transformation matrix :math:`\\boldsymbol M \in \mathbb{R}^{4 \\times 4}` and scaling rate :math:`s \in \mathbb{R}`.

    .. math::
        \mathcal T(\\boldsymbol x) = s \\boldsymbol R \\boldsymbol x + \\boldsymbol t = s \\boldsymbol M \\boldsymbol x

    .. seealso::
        Homogeneous transformation matrix is a very popular representation of rigid transformation, adopted by :mod:`OpenGL` and other computer vision packages. It applies rotation and translation in one :math:`4 \\times 4` matrix.
        
        More information: `Spatial Transformation Matrices - Rainer Goebel <https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html>`_

    Parameters
    ---
    R
        rotation matrix :math:`\\boldsymbol R \in \mathbb{R}^{3 \\times 3}` stored in (3, 3) :class:`numpy.array`.
    s
        scaling rate :math:`s \in \mathbb{R}` stored in a :class:`float` variable.
    t
        translation vector :math:`\\boldsymbol t \in \mathbb{R}^{3}` stored in (3, ) :class:`numpy.array`.

    Return
    ---
    :class:`float`
        scaling rate :math:`s \in \mathbb{R}` stored in a :class:`float` variable.
    :class:`numpy.array`
        homogeneous transformation matrix :math:`\\boldsymbol M \in \mathbb{R}^{4 \\times 4}` stored in (4, 4) :class:`numpy.array`.
    """
    M = np.diag(np.full(4, 1, dtype='float64'))
    M[0:3, 0:3] = R
    M[0:3, 3] = t/s
    return float(s), M


def transform_sm2rst(s: float, M: np.array) -> tuple[np.array, float, np.array]:
    """Transform rigid transformation representation from
        
        homogeneous transformation matrix :math:`\\boldsymbol M \in \mathbb{R}^{4 \\times 4}` and scaling rate :math:`s \in \mathbb{R}`.  
    
    to

        rotation matrix :math:`\\boldsymbol R \in \mathbb{R}^{3 \\times 3}`, scaling rate :math:`s \in \mathbb{R}`, and translation vector :math:`\\boldsymbol t \in \mathbb{R}^{3}`

    .. math::
        \mathcal T(\\boldsymbol x) = s \\boldsymbol R \\boldsymbol x + \\boldsymbol t = s \\boldsymbol M \\boldsymbol x

    .. seealso::
        Homogeneous transformation matrix is a very popular representation of rigid transformation, adopted by :mod:`OpenGL` and other computer vision packages. It applies rotation and translation in one :math:`4 \\times 4` matrix.
        
        More information: `Spatial Transformation Matrices - Rainer Goebel <https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html>`_

    Parameters
    ---
    s
        scaling rate :math:`s \in \mathbb{R}` stored in a :class:`float` variable.
    M
        homogeneous transformation matrix :math:`\\boldsymbol M \in \mathbb{R}^{4 \\times 4}` stored in (4, 4) :class:`numpy.array`.

    Return
    ---
    :class:`numpy.python`
        rotation matrix :math:`\\boldsymbol R \in \mathbb{R}^{3 \\times 3}` stored in (3, 3) :class:`numpy.array`.
    :class:`float`
        scaling rate :math:`s \in \mathbb{R}` stored in a :class:`float` variable.
    :class:`numpy.python`
        translation vector :math:`\\boldsymbol t \in \mathbb{R}^{3}` stored in (3, ) :class:`numpy.array`.
    """
    R = M[0:3, 0:3]
    t = M[0:3, 3]*s
    return R, s, t