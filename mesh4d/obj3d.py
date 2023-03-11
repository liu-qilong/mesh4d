"""The 4D object is consist of a series of 3D objects. In :mod:`mesh4d.obj3d`, 3D object classes with different features and capabilities are developed, serving for different analysis needs and scenarios. At current stage, there are 3 types of 3D object:

- Static 3D object :class:`Obj3d`

  It loads :code:`.obj` 3D mesh image and sampled it as the point cloud.

- Static 3D object :class:`Obj3d_Kps` with key points

  It's derived from :class:`Obj3d` and attach the key points (:class:`mesh4d.kps.Kps`) to it.

- Dynamic/Deformable 3D object :class:`Obj3d_Deform`

  It's derived from :class:`Obj3d_Kps` and attach the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) and non-rigid deformation (:class:`mesh4d.field.Trans_Nonrigid`) to it.

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

import mesh4d
import mesh4d.config.param
from mesh4d import kps, field

class Obj3d(object):
    """
    The basic 3D object class. Loads :code:`.obj` 3D mesh image and sampled it as the point cloud.

    Parameters
    ---
    filedir
        the direction of the 3D object.

    mode
        
        - :code:`load` the default mode is load from a file.
        - :code:`empty` create a 3D object without any 3D data.

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

        from mesh4d import obj3d
        o3 = obj3d.Obj3d(
            filedir = 'data/6kmh_softbra_8markers_1/speed_6km_soft_bra.000001.obj',
        )
        o3.show()
    """
    def __init__(
        self,
        filedir: str = '',
        mode: str = "load"
    ):
        if mode == "load":
            self.mesh = pvmesh_fix_disconnect(pv.read(filedir))
            self.texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
            self.pcd = np2pcd(self.mesh.points)

        elif mode == "empty":
            self.mesh = None
            self.texture = None
            self.pcd = None

    def show(self):
        """Show the loaded mesh and the sampled point cloud.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()

        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]
        self.add_mesh_to_scene(scene)
        self.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-6*width)


        scene.camera_position = 'xy'
        scene.show()

    def show_diff(obj1: Type[Obj3d], obj2: Type[Obj3d], cmap: str = "cool", op2: float = 0.2):
        """Illustrate the difference of two 3D object:

        :attr:`obj1` mesh will be coloured according to each of its points' distance to :attr:`obj2` mesh. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
        
        Parameters
        ---
        obj1
            the first 3D object.
        obj2
            the second 3D object.
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

        scene.camera_position = 'xy'
        scene.show()

    def add_mesh_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), show_edges: bool =True, **kwargs) -> pv.Plotter:
        """Add the visualisation of the mesh to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        location
            the displace location represented in a (3, ) :class:`numpy.array`.
        show_edges
            show mesh edges or not
        **kwargs
            other visualisation parameters.

            .. seealso::
                `pyvista.Plotter.add_mesh <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.BasePlotter.add_mesh.html>`_
                `pyvista.Plotter.add_points <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.BasePlotter.add_points.html>`_

        Returns
        ---
        :class:`pyvista.Plotter`
            :class:`pyvista.Plotter` scene added the visualisation.
        """
        scene.add_mesh(self.mesh.translate(location, inplace=False), show_edges=show_edges, **kwargs)

    def add_pcd_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), **kwargs) -> pv.Plotter:
        """Add the visualisation of the sampled key points to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        location
            the displace location represented in a (3, ) :class:`numpy.array`.
        **kwargs
            other visualisation parameters.

            .. seealso::
                `pyvista.Plotter.add_mesh <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.BasePlotter.add_mesh.html>`_
                `pyvista.Plotter.add_points <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.BasePlotter.add_points.html>`_

        Returns
        ---
        :class:`pyvista.Plotter`
            :class:`pyvista.Plotter` scene added the visualisation.
        """
        scene.add_points(pcd2np(self.pcd) + location, **kwargs)


class Obj3d_Kps(Obj3d):
    """
    The 3D object with key points attached to it. Derived from :class:`Obj3d` and attach the key points (:class:`mesh4d.kps.Kps`) to it.

    Parameters
    ---
    **kwargs
        parameters can be passed in via keywords arguments. Please refer to :class:`Obj3d` for accepted parameters.

    Note
    ---
    `Class Attributes`

    self.kps_group
        a dictionary of various :class:`mesh4d.kps.Kps` object (set of key points) attached to the 3D object, used for different purposes, such as measurement, registration guiding, virtue key point tracking, registration error estimation, etc.
    """
    def __init__(self, **kwargs):
        Obj3d.__init__(self, **kwargs)
        self.kps_group = {}

    def attach_kps(self, name: str, kps: Type[kps.Kps]):
        """attach key points.
        
        Parameters
        ---
        name
            name of the key points as its keyword in :attr:`self.kps_group`.
        kps
            the key points stored in a :class:`mesh4d.kps.Kps` object.
        """
        self.kps_group[name] = kps
    
    def load_kps_from_markerset(self, name: str, markerset: Type[kps.MarkerSet], time: float = 0.0):
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
        kps = markerset.get_time_coord(time)
        self.attach_kps(name, kps)

    def show(self, kps_names: Union[None, list, tuple] = None):
        """Show the loaded mesh, the sampled point cloud, and the key points attached to it.

        Parameters
        ---
        kps_names
            a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the `pythreejs <https://docs.pyvista.org/user-guide/jupyter/pythreejs.html>`_ backend of :mod:`pyvista` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('pythreejs')
        """
        scene = pv.Plotter()

        width = pcd_get_max_bound(self.pcd)[0] - pcd_get_min_bound(self.pcd)[0]

        self.add_mesh_to_scene(scene)
        self.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-6*width)
        self.add_kps_to_scene(scene, kps_names, radius=0.02*width)
        self.add_kps_to_scene(scene, kps_names, radius=0.02*width, location=[1.5*width, 0, 0])

        scene.camera_position = 'xy'
        scene.show()

    def show_diff(obj1: Type[Obj3d_Kps], obj2: Type[Obj3d_Kps], kps_names: Union[None, tuple, list] = None, cmap: str = "cool", op1: float = 0.8, op2: float = 0.2):
        """Illustrate the difference of two 3D object:

        - :attr:`obj1` mesh will be coloured according to each of its points' distance to :attr:`obj2` mesh. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
        - :attr:`obj1` key points will be coloured in gold while :attr:`obj1` key points will be coloured in green.

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
        op1
            the opacity of the first 3D object.
        op2
            the opacity of the second 3D object.
        """
        scene = pv.Plotter()
        
        tree = KDTree(obj2.mesh.points)
        d_kdtree, _ = tree.query(obj1.mesh.points)
        obj1.mesh["distances"] = d_kdtree
        width = pcd_get_max_bound(obj1.pcd)[0] - pcd_get_min_bound(obj1.pcd)[0]

        obj1.add_mesh_to_scene(scene, show_edges=False, opacity=op1, cmap=cmap)
        obj2.add_mesh_to_scene(scene, show_edges=False, opacity=op2)
        obj1.add_kps_to_scene(scene, kps_names, color="gold", radius=0.02*width)
        obj2.add_kps_to_scene(scene, kps_names, color="green", radius=0.02*width)

        scene.camera_position = 'xy'
        scene.show()

    def add_kps_to_scene(self, scene: pv.Plotter, kps_names: Union[None, tuple, list] = None, location: np.array = np.array((0, 0, 0)), color: Union[None, str] = None, **kwargs) -> pv.Plotter:
        """Add the visualisation of key points to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        kps_names
            a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        shift
	        shift the displace location by a (3, ) vector stored in :class:`list`, :class:`tuple`, or :class:`numpy.array`.
        color
            color of the key points. If not set, a unique color will be automatically assigned to each group of key points.
        **kwargs
            other visualisation parameters.

            .. seealso::
                `pyvista.Plotter.add_mesh <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.BasePlotter.add_mesh.html>`_
                `pyvista.Plotter.add_points <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.BasePlotter.add_points.html>`_

        Returns
        ---
        :class:`pyvista.Plotter`
            :class:`pyvista.Plotter` scene added the visualisation.
        """
        if kps_names is None:
            kps_names = self.kps_group.keys()

        if color is None:
            # prepare random color select
            random_color = True
            seed = 26
            color_ls = list(mcolors.CSS4_COLORS.keys())
        else:
            random_color = False

        for name in kps_names:
            # random color select
            if random_color:
                random.seed(seed)
                color = random.choice(color_ls)
                seed = seed + 1

            self.kps_group[name].add_to_scene(scene, location=location, color=color, **kwargs)


class Obj3d_Deform(Obj3d_Kps):
    """
    The dynamic/deformable 3D object with key points and transformations attached to it. Derived from :class:`Obj3d_Kps` and attach the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) and non-rigid deformation (:class:`mesh4d.field.Trans_Nonrigid`) to it.

    Parameters
    ---
    **kwargs
        parameters can be passed in via keywords arguments. Please refer to :class:`Obj3d` and :class:`Obj3d_Kps` for accepted parameters.

    Note
    ---
    `Class Attributes`

    self.trans_rigid
        the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) of the 3D object.
    self.trans_nonrigid
        the non-rigid transformation (:class:`mesh4d.field.Trans_Nonrigid`) of the 3D object.
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
            the rigid transformation (:class:`mesh4d.field.Trans_Rigid`).
        """
        self.trans_rigid = trans_rigid

    def set_trans_nonrigid(self, trans_nonrigid: field.Trans_Nonrigid):
        """Set non-rigid transformation.

        Parameters
        ---
        trans_nonrigid
            the non-rigid transformation (:class:`mesh4d.field.Trans_Nonrigid`).
        """
        self.trans_nonrigid = trans_nonrigid

    def get_deform_obj3d(self, mode: str = "nonrigid"):
        """Get the deformed 3D object.
        
        Parameters
        ---
        mode
            
            - :code:`nonrigid`: the non-rigid transformation will be used to deform the object.
            - :code:`rigid`: the rigid transformation will be used to deform the object.

        Warning
        ---
        Before calling this method, please make sure that the transformation has been estimated.
        """
        if mode == 'nonrigid' and self.trans_nonrigid is not None:
            trans = self.trans_nonrigid
        elif mode == 'rigid' and self.trans_rigid is not None:
            trans = self.trans_rigid
        else:
            if mesh4d.output_msg:
                print("fail to provide deformed object")
            
            return

        deform_obj = type(self)(mode='empty')
        deform_obj.mesh = trans.shift_mesh(self.mesh)
        deform_obj.pcd = trans.shift_pcd(self.pcd)
        
        for name in self.kps_group.keys():
            deform_obj.kps_group[name] = trans.shift_kps(self.kps_group[name])

        return deform_obj

    def show_deform(self, kps_names: Union[None, list, tuple] = None, mode: str = 'nonrigid', cmap: str = "cool"):
        """Illustrate the mesh, the sampled point cloud, and the key points after the estimated deformation.
        
        - The mesh will be coloured with the distance of deformation. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
        - The sampled points will be attached with displacement vectors to illustrate the displacement field.
        - The deformed key points will be shown attached to the mesh and point cloud.

        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        kps_names
            a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        mode
            
            - :code:`nonrigid`: the non-rigid transformation will be used to deform the object.
            - :code:`rigid`: the rigid transformation will be used to deform the object.

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
        if mode == 'nonrigid' and self.trans_nonrigid is not None:
            trans = self.trans_nonrigid
        elif mode == 'rigid' and self.trans_rigid is not None:
            trans = self.trans_rigid
        else:
            if mesh4d.output_msg:
                print("fail to provide deformed object")

            return

        scene = pv.Plotter()

        deform_obj = self.get_deform_obj3d(mode=mode)
        dist = np.linalg.norm(self.mesh.points - deform_obj.mesh.points, axis = 1)
        width = pcd_get_max_bound(deform_obj.pcd)[0] - pcd_get_min_bound(deform_obj.pcd)[0]

        deform_obj.mesh["distances"] = dist
        deform_obj.add_mesh_to_scene(scene, cmap=cmap)

        if mode == 'nonrigid' and self.trans_nonrigid is not None:
            trans.add_to_scene(scene, location=[1.5*width, 0, 0], cmap=cmap)
        elif mode == 'rigid' and self.trans_rigid is not None:
            trans.add_to_scene(scene, location=[1.5*width, 0, 0], cmap=cmap, original_length=width)
        
        deform_obj.add_kps_to_scene(scene, kps_names, radius=0.02*width)
        deform_obj.add_kps_to_scene(scene, kps_names, radius=0.02*width, location=[1.5*width, 0, 0])
        
        scene.camera_position = 'xy'
        scene.show()

    def offset_rotate(self):
        """Offset the rotation according to the estimated rigid transformation.

        Tip
        ---
        This method usually serves for reorientate all 3D objects to a referencing direction, since that the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) is usually estimated according to the difference of two different 3D object.
        """
        if self.trans_rigid is None:
            if mesh4d.output_msg:
                print("no rigid transformation")

            return

        rot = self.trans_rigid.rot
        center = pcd_get_center(self.pcd)

        self.mesh_o3d.rotate(rot, center)
        self.pcd.rotate(rot, center)
        
        if mesh4d.output_msg:
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

        from mesh4d import obj3d, obj4d

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
    files = [os.path.join(folder, f) for f in files if '.obj' in f]
    files.sort()

    o3_ls = []
    for n in range(start, end + 1, stride):
        filedir = files[n]
        o3_ls.append(obj_type(filedir=filedir, **kwargs))
        
        if mesh4d.output_msg:
            print("loaded 1 mesh file: {}".format(filedir))

    return o3_ls