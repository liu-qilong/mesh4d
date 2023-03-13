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
import math
import random
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial import KDTree
import matplotlib.colors as mcolors

import mesh4d
import mesh4d.config.param
from mesh4d import kps, field, utils

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

    self.mesh
        3D mesh (:class:`open3d.geometry.TriangleMesh`) loaded with :mod:`pyvista`.

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
            self.mesh = pv.read(filedir)
            self.texture = pv.read_texture(filedir.replace('.obj', '.jpg'))

        elif mode == "empty":
            self.mesh = None
            self.texture = None

    def get_vertices(self) -> np.array:
        """Get the vertices of :attr:`self.mesh`.

        Returns
        ---
        :class:`numpy.array`
            (N, 3) array storing the coordinates of the mesh vertices
        """
        return np.array(self.mesh.points)
    
    def get_sample_points(self, sample_num: int) -> np.array:
        """Get sampled points from :attr:`self.mesh`.

        Parameters
        ---
        sample_num
            the number of the points sampled from the mesh.
        """
        vertices_num = len(self.get_vertices())

        if sample_num < vertices_num:
            dec_ratio = 1 - sample_num / vertices_num
            dec_mesh = self.mesh.decimate_pro(dec_ratio)
            return np.array(dec_mesh.points)
        
        else:
            sub_time = math.ceil(np.log2(sample_num / vertices_num))
            sub_mesh = self.mesh.subdivide(sub_time, 'loop')
            return np.array(sub_mesh.points)
        
    def get_sample_kps(self, sample_num: int) -> kps.Kps:
        """Get sampled key points object (:class:`mesh4d.kps.Kps`) from :attr:`self.mesh`.

        Parameters
        ---
        sample_num
            the number of the points sampled from the mesh.
        """
        full_kps = kps.Kps()
        points = self.get_sample_points(sample_num)

        for idx in range(len(points)):
            full_kps.add_point("point {}".format(idx), points[idx])

        return full_kps
        
    def get_width(self) -> float:
        """Return the lateral width of the mesh
        """
        left = points_get_max_bound(self.get_vertices())[0]
        right = points_get_min_bound(self.get_vertices())[0]
        return left - right

    def show(self):
        """Show the loaded mesh and the sampled point cloud.

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the :code:`static` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('static')
        """
        scene = pv.Plotter()

        width = self.get_width()

        self.add_mesh_to_scene(scene)
        self.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-6*width)

        scene.camera_position = 'xy'
        scene.show()

    @staticmethod
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

    def show_with_value_mask(self, points: Iterable, values: Iterable, k_nbr: int = 20, max_threshold: Union[float, None] = None, min_threshold: Union[float, None] = None):
        """Show the 3D mesh with a value mask.

        Parameters
        ----------
        points : Iterable
            An iterable containing the points to use for assigning values to the mesh vertices.
        values : Iterable
            An iterable containing the values to assign to the mesh vertices based on their nearest point in `points`.
        k_nbr : int, optional
            The number of nearest neighbors to consider when assigning values to the mesh vertices. Default is 20.
        max_threshold : float or None, optional
            The maximum value to include in the mask. Any values greater than this threshold will be replaced with the threshold value. Default is None.
        min_threshold : float or None, optional
            The minimum value to include in the mask. Any values less than this threshold will be replaced with the threshold value. Default is None.
        """
        mesh = copy.deepcopy(self.mesh)

        # assign each vertex with a value based on its nearest point in points  
        tree = KDTree(points)
        _, idxs = tree.query(mesh.points, k=k_nbr)
        value_mask = np.take(values, idxs)
        value_mask = np.mean(value_mask, axis=1)

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
        scene.add_mesh(mesh)
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
        points = self.get_vertices()
        scene.add_points(points + location, **kwargs)


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
        Before calling this method in Jupyter Notebook environment, the :code:`static` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('static')
        """
        scene = pv.Plotter()

        width = self.get_width()

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

        width = obj1.get_width()

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
        Before calling this method in Jupyter Notebook environment, the :code:`static` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('static')
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

        width = self.get_width()

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
        center = points_get_center(self.get_vertices)

        self.mesh_o3d.rotate(rot, center)
        
        if mesh4d.output_msg:
            print("reorientated 1 3d object")


# utils for data & object transform

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


# utils for object cropping and other operations

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
            percent = (n + 1) / (end - start + 1)
            utils.progress_bar(percent, back_str=" loading: {}".format(filedir))

    return o3_ls