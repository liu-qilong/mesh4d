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
import matplotlib.colors as mcolors

import mesh4d
import mesh4d.config.param
from mesh4d import kps, field, utils
from mesh4d.analyse import measure

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
        mesh: Union[None, pv.core.pointset.PolyData] = None,
        texture: Union[None, pv.core.objects.Texture] = None,
        **kwargs,
    ):
        self.mesh = mesh
        self.texture = texture

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
            # dec_mesh = self.mesh.decimate_pro(dec_ratio)
            dec_mesh = self.mesh.decimate(dec_ratio)
            return np.array(dec_mesh.points)
        
        elif sample_num == vertices_num:
            return self.get_vertices()
        
        else:
            try:
                sub_time = math.ceil(np.log2(sample_num / vertices_num))
                sub_mesh = self.mesh.subdivide(sub_time, 'loop')

                dec_ratio = 1 - sample_num / len(sub_mesh.points)
                dec_mesh = self.mesh.decimate(dec_ratio)
                return np.array(dec_mesh.points)
            
            except:
                print("fail to provide denser sampling points. original vertices will be provided")
                return self.get_vertices()
        
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
        left = measure.points_get_max_bound(self.get_vertices())[0]
        right = measure.points_get_min_bound(self.get_vertices())[0]
        return left - right

    def show(self, elements: str = 'mp') -> pv.Plotter:
        """Show the loaded mesh and the sampled point cloud.

        Parameters
        ---
        elements
            tbf

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the :code:`static` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('static')
        """
        scene = pv.Plotter()

        width = self.get_width()

        if 'm' in elements:
            self.add_mesh_to_scene(scene)

        if 'p' in elements:
            self.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-6*width)

        scene.camera_position = 'xy'
        scene.show(interactive_update=True)

        return scene

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

    def show(self, kps_names: Union[None, list, tuple] = None, elements: str = 'mpk') -> pv.Plotter:
        """Show the loaded mesh, the sampled point cloud, and the key points attached to it.

        Parameters
        ---
        kps_names
            a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        elements
            tbf

        Attention
        ---
        Before calling this method in Jupyter Notebook environment, the :code:`static` is needed to be selected: ::

            import pyvista as pv
            pv.set_jupyter_backend('static')
        """
        scene = pv.Plotter()

        width = self.get_width()

        if 'm' in elements:
            self.add_mesh_to_scene(scene)

            if 'k' in elements:
                self.add_kps_to_scene(scene, kps_names, radius=0.02*width)

        if 'p' in elements:
            self.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-6*width)

            if 'k' in elements:
                self.add_kps_to_scene(scene, kps_names, radius=0.02*width, location=[1.5*width, 0, 0])

        scene.camera_position = 'xy'
        scene.show(interactive_update=True)

        return scene

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

    def show_deform(self, kps_names: Union[None, list, tuple] = None, mode: str = 'nonrigid', cmap: str = "cool") -> pv.Plotter:
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
        scene.show(interactive_update=True)

        return scene

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
        center = measure.points_get_center(self.get_vertices)

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


def load_mesh_series(
        folder: str,
        start: int = 0,
        end: int = 1,
        stride: int = 1,
    ) -> tuple:
    """ Load a series of obj files from a folder.
    
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

    Return
    ---
    Iterable[pv.core.pointset.PolyData]
        A list of :mod:`pyvista` mesh.
    Iterable[pv.core.objects.Texture]
        A list of :mod:`pyvista` texture.
    """
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files if '.obj' in f]
    files.sort()

    mesh_ls = []
    texture_ls = []

    for n in range(start, end + 1, stride):
        filedir = files[n]
        mesh_ls.append(pv.read(filedir))
        texture_ls.append(pv.read_texture(filedir.replace('.obj', '.jpg')))
        
        if mesh4d.output_msg:
            percent = (n + 1) / (end - start + 1)
            utils.progress_bar(percent, back_str=" loading: {}".format(filedir))

    return mesh_ls, texture_ls


def init_obj_series(
        mesh_ls: Iterable[pv.core.pointset.PolyData],
        texture_ls: Union[Iterable[pv.core.objects.Texture], None] = None,
        obj_type: Type[Obj3d] = Obj3d,
        **kwargs,
    ) -> Iterable[Type[Obj3d]]:
    """ Load a series of mesh files from a folder and initialise them as 3D objects.
    
    Parameters
    ---
    mesh_ls
        a list of :mod:`pyvista` mesh.
    texture_ls
        a list of :mod:`pyvista` texture or :code:`None`.
    obj_type
        The 3D object class. Any class derived from :class:`Obj3d` is accepted.
    **kwargs
        Configuration parameters for initialisation of the 3D object can be passed in via :code:`**kwargs`.

    Return
    ---
    Iterable[Type[Obj3d]]
        a list of 3D object.
    """
    o3_ls = []

    for idx in range(len(mesh_ls)):
        mesh = mesh_ls[idx]

        if texture_ls is not None:
            texture = texture_ls[idx]
        else:
            texture = None
            
        o3_ls.append(
            obj_type(mesh=mesh, texture=texture, **kwargs)
            )

    return o3_ls