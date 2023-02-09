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
import copy
import random
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial import KDTree
import matplotlib.colors as mcolors

import UltraMotionCapture
import UltraMotionCapture.config.param
from UltraMotionCapture import kps, field

class Obj3d(object):
    """
    The basic 3D object class. Loads :code:`.obj` 3D mesh image and sampled it as the point cloud.

    Parameters
    ---
    filedir
        the direction of the 3D object.
    scale_rate
        the scaling rate of the 3D object.

        .. attention::
            Noted that the original unit of 3dMD raw data is millimetre (mm). The default :attr:`scale_rate` remains this unit.

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
    self.scale_rate
        the scaling rate of the 3dMD model.

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
        scale_rate: float = 1,
    ):
        self.mesh = pvmesh_fix_disconnect(pv.read(filedir))
        self.texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
        self.scale_rate = scale_rate

        self.mesh.scale(self.scale_rate, inplace=True)
        self.pcd = np2pcd(self.mesh.points)

    def show(self, scene: Union[None, pv.Plotter] = None):
        """Show the loaded mesh and the sampled point cloud.
        
        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()
        self.add_to_scene(scene)
        scene.show()

    def add_to_scene(self, scene: pv.Plotter, shift: np.array = np.array((0, 0, 0))) -> pv.Plotter:
        """Add the visualisation of current object to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        shift
            shift the displace location by a (3, ) vector stored in :class:`list`, :class:`tuple`, or :class:`numpy.array`.

        Returns
        ---
        :class:`pyvista.Plotter`
            :class:`pyvista.Plotter` scene added the visualisation.
        """
        # plot mesh
        scene.add_mesh(self.mesh.translate(shift, inplace=False), show_edges=True)

        # plot sampled point cloud
        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        scene.add_points(pcd2np(self.pcd) + [1.5*width, 0, 0] + shift)

        return scene


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

    self.kps_group
        a dictionary of various :class:`UltraMotionCapture.kps.Kps` object (set of key points) attached to the 3D object, used for different purposes, such as measurement, registration guiding, virtue key point tracking, registration error estimation, etc.
    """
    def __init__(self, **kwargs):
        Obj3d.__init__(self, **kwargs)
        self.kps_group = {}

    def load_kps(self, name: str, markerset: Type[kps.MarkerSet], time: float = 0.0):
        """Load key points as :attr:`self.kps` from a :class:`kps.MarkerSet` object.
        
        Parameters
        ---
        name
            name of the kps as its key in :attr:`self.kps_group`.
        markerset
            the :class:`kps.MarkerSet` object.
        time
            the time from the :class:`kps.MarkerSet`'s recording period to be loaded.
        """
        self.kps_group[name] = markerset.get_time_coord(time, kps_class=kps.Kps)

    def show(self, kps_names: Union[None, list, tuple] = None):
        """Show the loaded mesh, the sampled point cloud, and the key points attached to it.

        Parameters
        ---
        kps_names
            a list of names of the :class:`~UltraMotionCapture.kps.Kps` objects to be shown. Noted that a :class:`~UltraMotionCapture.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()
        self.add_to_scene(scene, kps_names)
        scene.show()

    def add_to_scene(self, scene: pv.Plotter, kps_names: Union[None, tuple, list] = None, shift: np.array = np.array((0, 0, 0))) -> pv.Plotter:
        """Add the visualisation of current object to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        kps_names
            a list of names of the :class:`~UltraMotionCapture.kps.Kps` objects to be shown. Noted that a :class:`~UltraMotionCapture.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        shift
	        shift the displace location by a (3, ) vector stored in :class:`list`, :class:`tuple`, or :class:`numpy.array`.

        Returns
        ---
        :class:`pyvista.Plotter`
            :class:`pyvista.Plotter` scene added the visualisation.
        """        
        # plot mesh
        scene.add_mesh(self.mesh.translate(shift, inplace=False), show_edges=True)

        # plot sampled point cloud
        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        lateral_move = [1.5 * width, 0, 0]
        scene.add_points(pcd2np(self.pcd) + lateral_move + shift, point_size=0.001*width)

        # plot key points
        if kps_names is None:
            kps_names = self.kps_group.keys()

        seed = 26
        color_ls = list(mcolors.CSS4_COLORS.keys())

        for name in kps_names:
            # random color select
            random.seed(seed)
            color = random.choice(color_ls)
            seed = seed + 1

            self.kps_group[name].add_to_scene(scene, shift=shift, radius=0.02*width, color=color)
            self.kps_group[name].add_to_scene(scene, shift=lateral_move + shift, radius=0.02*width, color=color)
            
        return scene


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

    def load_kps(self, name: str, markerset: Type[kps.MarkerSet], time: float = 0.0):
        """Load a set of key points as into :attr:`self.kps_group` dictionary from a :class:`kps.MarkerSet` object with a name.
        
        Parameters
        ---
        name
            the name of the :class:`~UltraMotionCapture.kps.Kps` objects. Noted that this name will also been used as its keyword in :attr:`self.kps_group`.
        markerset
            the :class:`kps.MarkerSet` object.
        time
            the time from the :class:`kps.MarkerSet`'s recording period to be loaded.
        """
        self.kps_group[name] = markerset.get_time_coord(time, kps_class=kps.Kps)

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

    def show(self, kps_names: Union[None, list, tuple] = None, mode: str = 'raw', obj3d_gt: Union[None, Type[Obj3d_Kps]] = None, **kwargs):
        """Show the loaded mesh, the sampled point cloud, and the key points attached to it.

        Parameters
        ---
        kps_names
            a list of names of the :class:`~UltraMotionCapture.kps.Kps` objects to be shown. Noted that a :class:`~UltraMotionCapture.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        mode

            - :code:`raw` calls :meth:`add_to_scene` inherited from :class:`Obj3d_Kps` to illustrate the mesh, sampled points, and key points.
            - :code:`deform` calls :meth:`add_to_scene_deform` to illustrate the deformed mesh, displacement field and key points.
            - :code:`diff_deform_gt` calls :meth:`add_to_scene_diff_deform_gt` to illustrate the difference between the estimated deformed mesh and sampling points with the ground-truth.

        obj3d_gt
            if set :code:`mode='diff_deform_gt'`, the ground-truth 3D object is needed to be provided.
        **kwargs
            parameters passed to downstream methods.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()

        if mode == 'raw':
            self.add_to_scene(scene, kps_names, **kwargs)
        elif mode == 'deform':
            self.add_to_scene_deform(scene, kps_names, **kwargs)
        elif mode == 'diff_deform_gt':
            self.add_to_scene_diff_deform_gt(scene, obj3d_gt, kps_names, **kwargs)
        
        scene.show()

    def add_to_scene_deform(self, scene: pv.Plotter, kps_names: Union[None, tuple, list] = None, shift: np.array = np.array((0, 0, 0)), cmap: str = "cool"):
        """Illustrate the mesh, the sampled point cloud, and the key points after the estimated deformation.
        
        - The mesh will be coloured with the distance of deformation. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
        - The sampled points will be attached with displacement vectors to illustrate the displacement field.

        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        kps_names
            a list of names of the :class:`~UltraMotionCapture.kps.Kps` objects to be shown. Noted that a :class:`~UltraMotionCapture.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        shift
			shift the displace location by a (3, ) vector stored in :class:`list`, :class:`tuple`, or :class:`numpy.array`.
        cmap
            the color map name. 
            
            .. seealso::
                For full list of supported color map, please refer to `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        # plot mesh with displacement distance
        mesh_deform = self.trans_nonrigid.shift_mesh(self.mesh)
        dist = np.linalg.norm(self.mesh.points - mesh_deform.points, axis = 1)
        mesh_deform["distances"] = dist
        scene.add_mesh(mesh_deform.translate(shift, inplace=False), scalars="distances", cmap=cmap)

        if UltraMotionCapture.output_msg:
            print("average displacemnt: {:.3} (mm)".format(np.average(dist)/self.scale_rate))

        # plot displacement of the sampled point cloud
        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        lateral_move = [1.5 * width, 0, 0]
        self.trans_nonrigid.add_to_scene(scene, shift=lateral_move + shift)

        # plot key points
        if kps_names is None:
            kps_names = self.kps_group.keys()

        seed = 26
        color_ls = list(mcolors.CSS4_COLORS.keys())

        for name in kps_names:
            # random color select
            random.seed(seed)
            color = random.choice(color_ls)
            seed = seed + 1

            kps_deform = self.trans_nonrigid.shift_kps(self.kps_group[name])
            kps_deform.add_to_scene(scene, shift=shift, radius=0.02*width, color=color)
            kps_deform.add_to_scene(scene, shift=lateral_move + shift, radius=0.02*width, color=color)

        return scene

    def add_to_scene_diff_deform_gt(self, scene, obj3d_gt: Type[Obj3d_Kps], kps_names: Union[None, tuple, list] = None, shift: np.array = np.array((0, 0, 0)), cmap: str = "cool", opacity: float = 0.2):
        """Illustrate the distance between the deformed 3d object under revealed deformation and the ground-truth deformed 3d object.
        
        - The deformed mesh will be coloured with the distance of ground-truth mesh. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
        - The deformed key points will be coloured in gold while the ground-truth key points will be coloured in green.
        - The deformed sampled points will be illustrated to the right.
        - The ground-truth mesh will be displayed in low opacity with the deformed mesh and sampled points.

        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        obj3d_gt
            the ground-truth 3D object.
        kps_names
            a list of names of the :class:`~UltraMotionCapture.kps.Kps` objects to be shown. Noted that a :class:`~UltraMotionCapture.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        shift
            shift the displace location by a (3, ) vector stored in :class:`list`, :class:`tuple`, or :class:`numpy.array`.
        obj3d_gt

        cmap
            the color map name. 
            
            .. seealso::
                For full list of supported color map, please refer to `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
        opacity
            the opacity of the ground-truth 3D object.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')"""
        # plot the distance between the deformed mesh and the ground truth
        mesh_gt = obj3d_gt.mesh
        mesh_deform = self.trans_nonrigid.shift_mesh(self.mesh)
        
        tree = KDTree(mesh_gt.points)
        d_kdtree, _ = tree.query(mesh_deform.points)
        mesh_deform["distances"] = d_kdtree
        
        scene.add_mesh(mesh_deform.translate(shift, inplace=False), scalars="distances", cmap=cmap)
        scene.add_mesh(mesh_gt.translate(shift, inplace=False), opacity=opacity)

        if UltraMotionCapture.output_msg:
            print("average distance between the deformed mesh and the ground truth: {:.3} (mm)".format(np.mean(d_kdtree)/self.scale_rate))

        # plot the deformed sampled point cloud
        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        lateral_move = [1.5 * width, 0, 0]

        pcd_points = pcd2np(self.pcd)
        pcd_deform_points = self.trans_nonrigid.shift_points(pcd_points)
        scene.add_points(pcd_deform_points + lateral_move + shift, point_size=0.001*width, color='Gold')
        scene.add_mesh(mesh_gt.translate(lateral_move + shift, inplace=False), opacity=opacity)

        # plot the difference between key points
        if kps_names is None:
            kps_names = self.kps_group.keys()

        for name in kps_names:
            kps_deform = self.trans_nonrigid.shift_kps(self.kps_group[name])
            kps_deform.add_to_scene(scene, shift=shift, radius=0.02*width, color='gold')

            kps_gt = obj3d_gt.kps_group[name]
            kps_gt.add_to_scene(scene, shift=shift, radius=0.02*width, color='green')

        return scene

    def offset_rotate(self):
        """Offset the rotation according to the estimated rigid transformation.

        Tip
        ---
        This method usually serves for reorientate all 3D objects to a referencing direction, since that the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) is usually estimated according to the difference of two different 3D object.
        """
        if self.trans_rigid is None:
            if UltraMotionCapture.output_msg:
                print("no rigid transformation")

            return

        rot = self.trans_rigid.rot
        center = pcd_get_center(self.pcd)

        self.mesh_o3d.rotate(rot, center)
        self.pcd.rotate(rot, center)
        
        if UltraMotionCapture.output_msg:
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

    Attention
    ---
    The changing of the extracted data won't affect the original one.

    Parameters
    ---
    pcd
        the point cloud (:class:`open3d.geometry.PointCloud`).
    
    Return
    ---
    :class:`numpy.array`
        the points coordinates data stored in a (N, 3) :class:`numpy.array`.
    """
    pcd_copy = copy.deepcopy(pcd)
    return np.asarray(pcd_copy.points)


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
        
        if UltraMotionCapture.output_msg:
            print("loaded 1 mesh file: {}".format(filedir))

    return o3_ls