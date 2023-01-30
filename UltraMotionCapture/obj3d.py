"""The 4D object is consist of a series of 3D objects. In :mod:`UltraMotionCapture.obj3d`, 3D object classes with different features and capabilities are developed, serving for different analysis needs and scenarios. At current stage, there are 3 types of 3D object:

- Static 3D object :class:`Obj3d`

  It loads :code:`.obj` 3D mesh image and sampled it as the point cloud.

- Static 3D object :class:`Obj3d_Kps` with key points

  It's derived from :class:`Obj3d` and attach the key points (:class:`UltraMotionCapture.kps.Kps`) to it.

- Dynamic/Deformable 3D object :class:`Obj3d_Deform`

  It's derived from :class:`Obj3d_Kps` and attach the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) and non-rigid deformation (:class:`UltraMotionCapture.field.Trans_Nonrigid`) to it.

Moreover, a wide range of utils functions are provided, serving for 3D images loading, processing, format transformation, ect.
"""

from __future__ import annotations
from typing import Type, Union, Iterable

import os
import numpy as np
import open3d as o3d
import pyvista as pv

from UltraMotionCapture import kps
from UltraMotionCapture import field

class Obj3d(object):
    """
    The basic 3D object class. Loads :code:`.obj` 3D mesh image and sampled it as the point cloud.

    Parameters
    ---
    filedir
        the direction of the 3D object.
    scale_rate
        the scaling rate of the 3D object.

        .. seealso::
            Reason for providing :code:`scale_rate` parameter is explained in :class:`Obj3d_Deform`.

    sample_num
        the number of the points sampled from the mesh to construct the point cloud.

    Note
    ---
    `Class Attributes`

    self.mesh_o3d
        3D mesh (:class:`open3d.geometry.TriangleMesh`) loaded with :mod:`open3d` .
    self.pcd
        3D point cloud (:class:`open3d.geometry.PointCloud`) sampled from :attr:`self.mesh_o3d`.

    Attention
    ---
    In future development, mesh may also be loaded with :mod:`pyvista` as :attr:`self.mesh_pv`, for its advantages in visualisation and some specific geometric analysis features.

    Example
    ---
    ::

        from UltraMotionCapture import obj3d
        o3 = obj3d.Obj3d(
            filedir = 'data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj',
        )
        o3.show()
    """
    def __init__(
        self,
        filedir: str,
        scale_rate: float = 0.001,
        sample_num: int = 1000,
    ):
        self.mesh = pvmesh_fix_disconnect(pv.read(filedir))
        self.texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
        self.mesh.scale(scale_rate, inplace=True)

        self.pcd = pvmesh2pcd_pro(self.mesh, sample_num)

    def show(self):
        """Show the loaded mesh and the sampled point cloud.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()

        # plot mesh
        scene.add_mesh(self.mesh, show_edges=True)

        # plot sampled point cloud
        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        scene.add_points(pcd2np(self.pcd) + [1.5*width, 0, 0])

        scene.show()


class Obj3d_Kps(Obj3d):
    """
    The 3D object with key points attached to it. Derived from :class:`Obj3d` and attach the key points (:class:`UltraMotionCapture.kps.Kps`) to it.

    Parameters
    ---
    **kwargs
        parameters can be passed in via keywords arguments. Please refer to :class:`Obj3d` for accepted parameters.

    Note
    ---
    `Class Attributes`

    self.kps
        key points (:class:`UltraMotionCapture.kps.Kps`) attached to the 3D object.
    """
    def __init__(self, **kwargs):
        Obj3d.__init__(self, **kwargs)
        self.kps = kps.Kps_Deform()

    def load_kps(self, markerset: Type[kps.MarkerSet], time: float = 0.0):
        """Load key points from a :class:`kps.MarkerSet` object.
        
        Parameters
        ---
        markerset
            the :class:`kps.MarkerSet` object.
        time
            the time from the :class:`kps.MarkerSet`'s recording period to be loaded.
        """
        self.kps.load_from_markerset_time(markerset, time)

    def show(self):
        """Show the loaded mesh, the sampled point cloud, and the key points attached to it.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()
        
        # plot mesh
        scene.add_mesh(self.mesh, show_edges=True)

        # plot sampled point cloud
        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        lateral_move = [1.5 * width, 0, 0]
        scene.add_points(pcd2np(self.pcd) + lateral_move, point_size=0.001*width)

        # plot key points
        pvpcd_kps = np2pvpcd(self.kps.get_kps_source_points(), radius=0.02*width)
        scene.add_mesh(pvpcd_kps, color='Gold')
        scene.add_mesh(pvpcd_kps.translate(lateral_move, inplace=False), color='Gold')

        scene.show()


class Obj3d_Deform(Obj3d_Kps):
    """
    The dynamic/deformable 3D object with key points and transformations attached to it. Derived from :class:`Obj3d_Kps` and attach the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) and non-rigid deformation (:class:`UltraMotionCapture.field.Trans_Nonrigid`) to it.

    Parameters
    ---
    **kwargs
        parameters can be passed in via keywords arguments. Please refer to :class:`Obj3d` and :class:`Obj3d_Kps` for accepted parameters.

        Attention
        ---
        The transformations (:mod:`UltraMotionCapture.field`) are estimated via registration. For effective registration iteration, as an empirical suggestion, the absolute value of coordinates shall falls into or near :math:`(-1, 1)`. That's why we provide a :code:`scale_rate` parameter defaulted as :math:`10^{-2}` in the initialisation method of the base class (:class:`Obj3d`).

    Note
    ---
    `Class Attributes`

    self.trans_rigid
        the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) of the 3D object.
    self.trans_nonrigid
        the non-rigid transformation (:class:`UltraMotionCapture.field.Trans_Nonrigid`) of the 3D object.
    """
    def __init__(self, **kwargs):
        Obj3d_Kps.__init__(self, **kwargs)
        self.trans_rigid = None
        self.trans_nonrigid = None

    def set_trans_rigid(self, trans_rigid: field.Trans_Rigid):
        """Set rigid transformation.

        Parameters
        ---
        trans_rigid
            the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`).
        """
        self.trans_rigid = trans_rigid

    def set_trans_nonrigid(self, trans_nonrigid: field.Trans_Nonrigid):
        """Set non-rigid transformation.

        Parameters
        ---
        trans_nonrigid
            the non-rigid transformation (:class:`UltraMotionCapture.field.Trans_Nonrigid`).
        """
        self.trans_nonrigid = trans_nonrigid
        self.kps.set_trans(trans_nonrigid)

    def offset_rotate(self):
        """Offset the rotation according to the estimated rigid transformation.

        Tip
        ---
        This method usually serves for reorientate all 3D objects to a referencing direction, since that the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) is usually estimated according to the difference of two different 3D object.
        """
        if self.trans_rigid is None:
            print("no rigid transformation")
            return

        rot = self.trans_rigid.rot
        center = pcd_get_center(self.pcd)

        self.mesh_o3d.rotate(rot, center)
        self.pcd.rotate(rot, center)

        print("reorientated 1 3d object")


# utils for data & object transform

def mesh2pcd(mesh: o3d.geometry.TriangleMesh, sample_num: int) -> o3d.geometry.PointCloud:
    """Sampled a :mod:`open3d` mesh (:class:`open3d.geometry.TriangleMesh`) to a :mod:`open3d` point cloud (:class:`open3d.geometry.PointCloud`).

    .. seealso::

        The sampling method is :func:`open3d.geometry.sample_points_poisson_disk` (`link <http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.sample_points_poisson_disk.html#open3d-geometry-sample-points-poisson-disk>`_) [#]_.
       
        .. [#] Cem Yuksel. "Sample elimination for generating poisson disk sample sets". Computer Graphics Forum. 2015, 34(2): 25–32.

    Parameters
    ---
    mesh
        the mesh (:class:`open3d.geometry.TriangleMesh`) being sampled.
    sample_num
        the number of sampling points.
    
    Return
    ---
    :class:`o3d.geometry.PointCloud`
        The sampled point cloud.
    """
    return mesh.sample_points_poisson_disk(number_of_points=sample_num, init_factor=5)


def pcd2np(pcd: o3d.geometry.PointCloud) -> np.array:
    """Extracted the points coordinates data from a :mod:`open3d` point cloud (:class:`open3d.geometry.PointCloud`).

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.PointCloud`).
    
    Return
    ---
    :class:`numpy.array`
        the points coordinates data stored in a (N, 3) :class:`numpy.array`.
    """
    return np.asarray(pcd.points)


def np2pcd(points: np.array) -> o3d.cpu.pybind.geometry.PointCloud:
    """Transform the points coordinates stored in a :class:`numpy.array` to a a :mod:`open3d` point cloud (:class:`open3d.geometry.PointCloud`).

    Parameters
    ---
    points
        the points coordinates data stored in a (N, 3) :class:`numpy.array`.
        
    Return
    ---
    :class:`open3d.geometry.PointCloud`
        The point cloud (:class:`open3d.geometry.PointCloud`).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


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


def pvmesh2pcd(mesh: pv.core.pointset.PolyData, sample_num: int = 1000) -> o3d.cpu.pybind.geometry.PointCloud:
    """Transform the :mod:`pyvista` mesh to a :mod:`open3d` point cloud with uniform sampling method
    
    Parameters
    ---
    mesh
        the :mod:`pyvista` mesh.
    sample_num
        the number of sampling points.

    See Also
    ---
    The sampling is realised with decimation function provided by :mod:`pyvista`: `Decimation - PyVista <https://docs.pyvista.org/examples/01-filter/decimate.html>`_.
    `pyvista.PolyData.decimate <https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.decimate.html#pyvista.PolyData.decimate>`_ is used for uniform sampling.
    """
    dec_ratio = 1 - sample_num / len(mesh.points)
    dec_mesh = mesh.decimate(dec_ratio)
    return np2pcd(dec_mesh.points)


def pvmesh2pcd_pro(mesh: pv.core.pointset.PolyData, sample_num: int = 1000) -> o3d.cpu.pybind.geometry.PointCloud:
    """Transform the :mod:`pyvista` mesh to a :mod:`open3d` point cloud with curation sampling method
    
    Parameters
    ---
    mesh
        the :mod:`pyvista` mesh.
    sample_num
        the number of sampling points.

    See Also
    ---
    The sampling is realised with decimation function provided by :mod:`pyvista`: `Decimation - PyVista <https://docs.pyvista.org/examples/01-filter/decimate.html>`_.
    `pyvista.PolyData.decimate_pro <https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.decimate_pro.html>`_ is used for uniform sampling.
    """
    dec_ratio = 1 - sample_num / len(mesh.points)
    dec_mesh = mesh.decimate_pro(dec_ratio)
    return np2pcd(dec_mesh.points)



# utils for object cropping and other operations

def mesh_crop(mesh: o3d.geometry.TriangleMesh, min_bound: list = [-1000, -1000, -1000], max_bound: list = [1000, 1000, 1000]) -> o3d.geometry.TriangleMesh:
    """Crop the mesh (:class:`open3d.geometry.TriangleMesh`) according the maximum and minimum boundaries.

    Parameters
    ---
    mesh
        the mesh (:class:`open3d.geometry.TriangleMesh`) being cropped.
    max_bound
        a list containing the maximum value of :math:`x, y, z`-coordinates: :code:`[max_x, max_y, max_z]`.
    min_bound
        a list containing the minimum value of :math:`x, y, z`-coordinates: :code:`[min_x, min_y, min_z]`.

    Return
    ---
    :class:`o3d.geometry.TriangleMesh`
        The cropped mesh.
    """
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return mesh.crop(bbox)


def pcd_crop(pcd: o3d.geometry.PointCloud, min_bound: list = [-1000, -1000, -1000], max_bound: list = [1000, 1000, 1000]) -> o3d.geometry.PointCloud:
    """Crop the point cloud (:class:`open3d.geometry.PointCloud`) according the maximum and minimum boundaries.

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.PointCloud`) being cropped.
    max_bound
        a list containing the maximum value of :math:`x, y, z`-coordinates: :code:`[max_x, max_y, max_z]`.
    min_bound
        a list containing the minimum value of :math:`x, y, z`-coordinates: :code:`[min_x, min_y, min_z]`.

    Return
    ---
    :class:`o3d.geometry.PointCloud`
        The cropped point cloud.
    """
    points = pcd2np(pcd)
    points_crop = []

    for point in points:
        min_to_point = point - min_bound
        point_to_max = max_bound - point
        less_than_zero = np.sum(min_to_point < 0) + np.sum(point_to_max < 0)
        if less_than_zero == 0:
            points_crop.append(point)

    return np2pcd(np.array(points_crop))


def pcd_crop_front(pcd: o3d.geometry.PointCloud, ratio: float = 0.5) -> o3d.geometry.PointCloud:
    """Crop the front side of a point cloud (:class:`open3d.geometry.PointCloud`) with a adjustable ratio.

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.TriangleMesh`) being cropped.
    ratio
        the ratio of the cropped front part.

    Return
    ---
    :class:`o3d.geometry.PointCloud`
        the cropped point cloud.
    """
    max_bound = pcd_get_max_bound(pcd)
    min_bound = pcd_get_min_bound(pcd)
    min_bound[2] = (1-ratio)*max_bound[2] + ratio*min_bound[2]
    return pcd_crop(pcd, min_bound)


def pvmesh_fix_disconnect(mesh: pv.core.pointset.PolyData) -> pv.core.pointset.PolyData():
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

# utils for object estimation

def pcd_get_center(pcd: o3d.geometry.PointCloud) -> np.array:
    """Get the center point of a point cloud. The center point is defined as the geometric average point of all points:

    .. math::
        \\boldsymbol c = \\frac{\sum_{i} \\boldsymbol p_i}{N}

    where :math:`N` denotes the total number of points.

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.TriangleMesh`).

    Return
    ---
    :class:`numpy.array`
        The center point coordinates represented in a (3, ) :class:`numpy.array`.
    """
    points = pcd2np(pcd)
    return np.mean(points, 0)


def pcd_get_max_bound(pcd: o3d.geometry.PointCloud) -> np.array:
    """Get the maximum boundary of a point cloud.

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.TriangleMesh`).

    Return
    ---
    :class:`numpy.array`
        a list containing the maximum value of :math:`x, y, z`-coordinates: :code:`[max_x, max_y, max_z]`.
    """
    points = pcd2np(pcd)
    return np.ndarray.max(points, 0)


def pcd_get_min_bound(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Get the minimum boundary of a point cloud.

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.TriangleMesh`).

    Return
    ---
    :class:`numpy.array`
        a list containing the minimum value of :math:`x, y, z`-coordinates: :code:`[min_x, min_y, min_z]`.
    """
    points = pcd2np(pcd)
    return np.ndarray.min(points, 0)


def search_nearest_point_idx(point: np.array, target_points: np.array) -> int:
    """Search the index of the nearest point from a collection of target points.

    Parameters
    ---
    point
        the source point coordinates stores in a (3, ) :class:`numpy.array`.
    target_points
        the target points collection stores in a (N, 3) :class:`numpy.array`.

    Return
    ---
    :class:`int`
        the index of the nearest point.
    """
    dist = np.linalg.norm(
        target_points - point, axis=1
    )
    idx = np.argmin(dist)
    return idx


def search_nearest_point(point: np.array, target_points: np.array) -> np.array:
    """Search the nearest point from a collection of target points.

    Parameters
    ---
    point
        the source point coordinates stores in a (3, ) :class:`numpy.array`.
    target_points
        the target points collection stores in a (N, 3) :class:`numpy.array`.

    Return
    ---
    :class:`numpy.array`
        the nearest point coordinates stores in a (3, ) :class:`numpy.array`.
    """
    idx = search_nearest_point_idx(point, target_points)
    return target_points[idx]


# utils for 3D objects loading

def load_obj_series(
        folder: str,
        start: int = 0,
        end: int = 1,
        stride: int = 1,
        obj_type: Type[Obj3d] = Obj3d,
        **kwargs
    ) -> Iterable[Type[Obj3d]]:
    """ Load a series of point cloud obj files from a folder.
    
    Parameters
    ---
    folder
        the directory of the folder storing the 3D images.
    start
        begin loading from the :code:`start`-th image.
        
        Attention
        ---
        Index begins from 0. The :code:`start`-th image is included in the loaded images.
        Index begins from 0.
    end
        end loading at the :code:`end`-th image.
        
        Attention
        ---
        Index begins from 0. The :code:`end`-th image is included in the loaded images.
    stride
        the stride of loading. For example, setting :code:`stride=5` means load one from every five 3D images.
    obj_type
        The 3D object class. Any class derived from :class:`Obj3d` is accepted.
    **kwargs
        Configuration parameters for initialisation of the 3D object can be passed in via :code:`**kwargs`.

    Return
    ---
    Iterable[Type[Obj3d]]
        A list of 3D object.

    Example
    ---
    The :func:`load_obj_series` is usually used for getting a list of 3D object and then loading to the 4D object: ::

        from UltraMotionCapture import obj3d, obj4d

        o3_ls = obj3d.load_obj_series(
            folder='data/6kmh_softbra_8markers_1/',
            start=0,
            end=1,
            sample_num=1000,
        )

        o4 = obj4d.Obj4d()
        o4.add_obj(*o3_ls)
    """
    files = os.listdir(folder)
    files = [folder + f for f in files if '.obj' in f]
    files.sort()

    o3_ls = []
    for n in range(start, end + 1, stride):
        filedir = files[n]
        o3_ls.append(obj_type(filedir=filedir, **kwargs))
        print("loaded 1 mesh file: {}".format(filedir))

    return o3_ls