"""The 4D object contains a series of 3D objects (:mod:`mesh4d.obj3d`). In :mod:`mesh4d.obj4d`, 4D object classes with different features and capabilities are developed, serving for different analysis needs and scenarios. At current stage, there are 3 types of 4D object:

- Static 4D object :class:`Obj4d`

  It contains a list of 3D objects.

- Static 4D object :class:`Obj4d_Kps` with key points

  It's derived from :class:`Obj4d` and attach key points (:class:`mesh4d.kps.Kps`) to each of the 3D object via Vicon motion capture data (:class:`mesh4d.kps.MarkerSet`).

- Dynamic/Deformable 4D object :class:`Obj4d_Deform`

  It's derived from :class:`Obj4d_Kps` and attach the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) and non-rigid deformation (:class:`mesh4d.field.Trans_Nonrigid`) to each of the 3D object by registration.
"""
from __future__ import annotations
from typing import Type, Union, Iterable

import os
import numpy as np
import pyvista as pv

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d
from mesh4d import kps, field, utils

class Obj4d(object):
    """Static 4D object. Contains a list of 3D objects.

    Parameters
    ---
    start_time
        the start time of the coordinates data.
    fps
        the number of frames per second (fps).

    Note
    ---
    `Class Attributes`

    self.start_time
        the start time of the coordinates data.
    self.fps
        the number of frames per second (fps).
    self.obj_ls
        a :class:`list` of 3D objects.

    Example
    ---
    Use :func:`load_obj_series` to load a list of 3D objects and then add them to the 4D object: ::

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
    def __init__(self, start_time: float = 0.0, fps: int = 120):
        self.obj_ls = []
        self.start_time = start_time
        self.fps = fps

    def add_obj(self, *objs: Iterable(Type[obj3d.Obj3d]), **kwargs):
        """ Add object(s).
        
        Parameters
        ---
        *objs
            unspecified number of 3D objects.

            .. seealso::

                About the :code:`*` symbol and its effect, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_

        Example
        ---
        Let's say we have two 3D objects :code:`o3_a`, :code:`o3_b` and 4D object :code:`o4`. 3D objects can be passed into the :meth:`add_obj` method one by one: ::

            o4.add_obj(o3_a, o3_b)

        3D objects can be passed as a list: ::

            o3_ls = [o3_a, o3_b]
            o4.add_obj(*o3_ls)
        """
        for obj in objs:
            self.obj_ls.append(obj)

    def export_gif(self, output_folder: str = "output/", filename: str = "obj4d"):
        """Illustrate the 4D object.
        
        Parameters
        ---
        output_folder
            the output folder of the generated :code:`.gif` file.
        filename
            the output filename of the generated :code:`.gif` file.
        """
        scene = pv.Plotter()
        scene.open_gif(os.path.join(output_folder, filename))
        
        plot_num = len(self.obj_ls)

        for idx in range(0, plot_num):
            obj = self.obj_ls[idx]
            scene.clear()

            width = obj.get_width()

            obj.add_mesh_to_scene(scene)
            obj.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-3*width)
            
            scene.camera_position = 'xy'
            scene.write_frame()

            if mesh4d.output_msg:
                percent = (idx + 1) / plot_num
                utils.progress_bar(percent, back_str=" exported the {}-th frame".format(idx))

        scene.close()


class Obj4d_Kps(Obj4d):
    """Static 4D object :class:`Obj4d_Kps` with key points. Derived from :class:`Obj4d` and attach key points (:class:`mesh4d.kps.Kps`) to each of the 3D object via Vicon motion capture data (:class:`mesh4d.kps.MarkerSet`).

    Parameters
    ---
    **kwargs
        configuration parameters of the base classes (:class:`Obj3d`) can be passed in via :code:`**kwargs`.

    Example
    ---
    Load Vicon motion capture data (:class:`mesh4d.kps.MarkerSet`) when initialising the 4D object. And use :func:`load_obj_series` to load a list of 3D objects and add them to the 4D object: ::

        from mesh4d import obj3d, obj4d, kps

        o3_ls = obj3d.load_obj_series(
            folder='data/6kmh_softbra_8markers_1/',
            start=0,
            end=1,
            sample_num=1000,
            obj_type=obj3d.Obj3d_Kps
        )

        vicon = kps.MarkerSet()
        vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
        vicon.interp_field()

        o4 = obj4d.Obj4d_Kps(
            fps=120,
        )

        o4.add_obj(*o3_ls)
    """
    def load_markerset(self, name: str, markerset: Union[kps.MarkerSet, None] = None):
        """Slice the :class:`~mesh4d.kps.MarkerSet` object into :class:`~mesh4d.kps.kps` objects and attached them to the corresponding frames.

        Parameters
        ---
        name
            the name of the :class:`~mesh4d.kps.Kps` objects. Noted that this name will also been used as its keyword in :class:`~mesh4d.obj3d.Obj3d_Kps` object's :attr:`kps_group` attribute.
        markerset
            Vicon motion capture data (:class:`mesh4d.kps.MarkerSet`).
        """
        for idx in range(len(self.obj_ls)):
            obj = self.obj_ls[idx]
            obj.load_kps_from_markerset(name, markerset, self.start_time + idx / self.fps)

    def assemble_markerset(self, name: str) -> kps.MarkerSet:
        """Assemble key points object in different frames into a marker set object (:class:`mesh4d.kps.MarkerSet`).
        
        Parameters
        ---
        name
            the name of the key points object, i.e. its keyword in the 3D object's :attr:`kps_group` dictionary.
        """
        markerset = kps.MarkerSet()
        markerset.fps = self.fps

        for obj in self.obj_ls:
            points_dict = obj.kps_group[name].points

            for point_name in points_dict.keys():
                if point_name not in markerset.markers.keys():
                    markerset.markers[point_name] = kps.Marker(name=point_name, fps=self.fps)
                
                markerset.markers[point_name].append_data(coord=points_dict[point_name], convert=False)

        return markerset

    def export_gif(self, output_folder: str = "output/", filename: str = "obj4d", kps_names: Union[None, list, tuple] = None):
        """Illustrate the 4D object.
        
        Parameters
        ---
        output_folder
            the output folder of the generated :code:`.gif` file.
        filename
            the output filename of the generated :code:`.gif` file.
        kps_names
            a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        """
        scene = pv.Plotter()
        scene.open_gif(os.path.join(output_folder, filename + '.gif'))

        plot_num = len(self.obj_ls)

        for idx in range(0, plot_num):
            obj = self.obj_ls[idx]
            scene.clear()
            
            width = obj.get_width()

            obj.add_mesh_to_scene(scene)
            obj.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-3*width)
            obj.add_kps_to_scene(scene, kps_names, radius=0.02*width)
            obj.add_kps_to_scene(scene, kps_names, radius=0.02*width, location=[1.5*width, 0, 0])
            
            scene.camera_position = 'xy'
            scene.write_frame()

            if mesh4d.output_msg:
                percent = (idx + 1) / plot_num
                utils.progress_bar(percent, back_str=" exported the {}-th frame".format(idx))

        scene.close()


class Obj4d_Deform(Obj4d_Kps):
    """Dynamic/Deformable 4D object :class:`Obj4d_Deform`. Derived from :class:`Obj4d_Kps` and attach the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) and non-rigid deformation (:class:`mesh4d.field.Trans_Nonrigid`) to each of the 3D object by registration.

    Parameters
    ---
    enable_rigid
        whether enable rigid transformation registration ot not.
    enable_nonrigid
        whether enable non-rigid transformation registration ot not.
    **kwargs
        configuration parameters of the base classes (:class:`Obj3d` and :class:`Obj3d_Kps`) can be passed in via :code:`**kwargs`.

    Note
    ---
    `Class Attributes`

    self.enable_rigid
        whether enable rigid transformation registration ot not. Default as :code:`False`.
    self.enable_nonrigid
        whether enable non-rigid transformation registration ot not. Default as :code:`False`.

    Example
    ---
    Load Vicon motion capture data (:class:`mesh4d.kps.MarkerSet`) when initialising the 4D object. Use :func:`load_obj_series` to load a list of 3D objects. And then add them to the 4D object. In the procedure of adding, the program will implement the activated transformation estimation automatically: ::

        from mesh4d import obj3d, kps, obj4d

        o3_ls = obj3d.load_obj_series(
            folder='data/6kmh_softbra_8markers_1/',
            start=0,
            end=1,
            sample_num=1000,
            obj_type=obj3d.Obj3d_Deform
        )

        vicon = kps.MarkerSet()
        vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
        vicon.interp_field()

        o4 = obj4d.Obj4d_Deform(
            fps=120,
            enable_rigid=True,
            enable_nonrigid=True,
        )

        o4.add_obj(*o3_ls)
    """
    def __init__(self, enable_rigid: bool = False, enable_nonrigid: bool = False,  **kwargs):
        Obj4d_Kps.__init__(self, **kwargs)
        self.enable_rigid = enable_rigid
        self.enable_nonrigid = enable_nonrigid

    def regist(self, **kwargs):
        """Implement registration among 3D objects in :attr:`self.obj_ls`.

        Parameters
        ---
        **kwargs
            configuration parameters for the registration and the configuration parameters of the base classes (:class:`Obj3d` and :class:`Obj3d_Kps`)'s :meth:`add_obj` method can be passed in via :code:`**kwargs`.
        """
        reg_num = len(self.obj_ls)

        for idx in range(reg_num):
            if idx == 0:
                self.process_first_obj()
                continue

            if self.enable_rigid:
                self.process_rigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the previous one

            if self.enable_nonrigid:
                self.process_nonrigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the later one

            if mesh4d.output_msg:
                percent = (idx + 1) / reg_num
                utils.progress_bar(percent, back_str=" registered the {}-th frame".format(idx))

    def process_first_obj(self):
        """Process the first added 3D object.
        
        Attention
        ---
        Called by :meth:`add_obj`."""
        pass

    def process_rigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        """Estimate the rigid transformation of the added 3D object. The lastly added 3D object is used as source object and the newly added 3D object as the target object.

        Parameters
        ---
        idx_source
            the index of the source 3D object in :code:`self.obj_ls`.
        idx_target
            the index of the target 3D object in :code:`self.obj_ls`.

        Attention
        ---
        The estimated transformation is load to the source object, via its :meth:`mesh4d.obj3d.Obj3d_Deform.set_trans_rigid` method.
        
        Attention
        ---
        Called by :meth:`add_obj`."""
        trans = field.Trans_Rigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        """Estimate the non-rigid transformation of the added 3D object. The lastly added 3D object is used as source object and the newly added 3D object as the target object.

        Parameters
        ---
        idx_source
            the index of the source 3D object in :code:`self.obj_ls`.
        idx_target
            the index of the target 3D object in :code:`self.obj_ls`.

        Attention
        ---
        The estimated transformation is load to the source object, via its :meth:`mesh4d.obj3d.Obj3d_Deform.set_trans_nonrigid` method.
        
        Attention
        ---
        Called by :meth:`add_obj`.
        """
        trans = field.Trans_Nonrigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)

    def export_deform_gif(self, output_folder: str = "output/", filename: str = "obj4d_deform", kps_names: Union[None, list, tuple] = None, mode: str = 'nonrigid', is_color_body: bool = True, cmap: str = "cool"):
        """Illustrate the 4D object with estimated displacement field.

        - The mesh will be coloured with the distance of deformation. The mapping between distance and color is controlled by :attr:`cmap` argument. Noted that in default setting, light bule indicates small deformation and purple indicates large deformation.
        - The sampled points will be attached with displacement vectors to illustrate the displacement field.
        - The deformed key points will be shown attached to the mesh and point cloud.
        
        Parameters
        ---
        output_folder
            the output folder of the generated :code:`.gif` file.
        filename
            the output filename of the generated :code:`.gif` file.
        kps_names
            a list of names of the :class:`~mesh4d.kps.Kps` objects to be shown. Noted that a :class:`~mesh4d.kps.Kps` object's name is its keyword in :attr:`self.kps_group`.
        mode
            
            - :code:`nonrigid`: the non-rigid transformation will be used to deform the object.
            - :code:`rigid`: the rigid transformation will be used to deform the object.

        is_color_body
            color the mesh with deformation distance or not.

        cmap
            the color map name. 
            
            .. seealso::
                For full list of supported color map, please refer to `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
        """
        scene = pv.Plotter()
        scene.open_gif(os.path.join(output_folder, filename + '.gif'))

        plot_num = len(self.obj_ls) - 1

        for idx in range(0, plot_num):
            obj = self.obj_ls[idx]
            scene.clear()

            if mode == 'nonrigid' and obj.trans_nonrigid is not None:
                trans = obj.trans_nonrigid
            elif mode == 'rigid' and obj.trans_rigid is not None:
                trans = obj.trans_rigid
            else:
                if mesh4d.output_msg:
                    print("fail to provide deformed object")

                scene.close()
                return

            deform_obj = obj.get_deform_obj3d(mode=mode)
            dist = np.linalg.norm(obj.mesh.points - deform_obj.mesh.points, axis = 1)
            width = deform_obj.get_width()

            if is_color_body:
                deform_obj.mesh["distances"] = dist
                deform_obj.add_mesh_to_scene(scene, cmap=cmap)

            else:
                deform_obj.add_mesh_to_scene(scene)

            if mode == 'nonrigid' and obj.trans_nonrigid is not None:
                trans.add_to_scene(scene, location=[1.5*width, 0, 0], cmap=cmap)

            elif mode == 'rigid' and obj.trans_rigid is not None:
                trans.add_to_scene(scene, location=[1.5*width, 0, 0], cmap=cmap, original_length=width)
            
            deform_obj.add_kps_to_scene(scene, kps_names, radius=0.02*width)
            deform_obj.add_kps_to_scene(scene, kps_names, radius=0.02*width, location=[1.5*width, 0, 0])
            
            scene.camera_position = 'xy'
            scene.write_frame()

            if mesh4d.output_msg:
                percent = (idx + 1) / plot_num
                utils.progress_bar(percent, back_str=" exported the {}-th frame".format(idx))

        scene.close()

    def offset_rotate(self):
        """Offset the rotation according to the estimated rigid transformation.

        Tip
        ---
        This method usually serves for reorientate all 3D objects to a referencing direction, since that the rigid transformation (:class:`mesh4d.field.Trans_Rigid`) is estimated one follow one in the 3D objects list.
        
        Example
        ---
        Let's say we have an properly loaded 4D object :code:`o4`, we'd like to view it before and after reorientated: ::

            import copy

            o4_offset = copy.deepcopy(o4)
            o4_offset.offset_rotate()
            
            o4.show()
            o4_offset.show()
        """
        for obj in self.obj_ls[1:]:  # the first 3d object doesn't need reorientation
            obj.offset_rotate()

        if mesh4d.output_msg:
            print("4d object reorientated")

    def vkps_track(self, kps: Type[kps.Kps], frame_id: int = 0, name: str = 'vkps', k_nbr: int = 1):
        """Virtual key points tracking.

        - Firstly, attach a set of key points (:class:`~mesh4d.kps.Kps`) to a frame of 3D object.
        - Then the algorithm will estimated its location in the proceeding and following frames, according to the revealed deformation between frames.

        Note
        ---
        Since the virtual tracking is based on the revealed deformation between frames, for using virtual key points tracking feature, the :attr:`enable_nonrigid` attributes in the initialisation of the 4D object must be set as :code:`True`. ::

            o4 = obj4d.Obj4d_Deform(
                fps=120,
                enable_nonrigid=True,
            )

        Parameters
        ---
        kps
            key points object (:class:`~mesh4d.kps.Kps`).
        frame_id
            the frame number to which the key points object attach.
        name
            name of the virtual key points as its keyword when attached to a 3D object.
        k_nbr
            see :meth:`mesh4d.filed.Trans_Nonrigid.shift_points`.
        """
        self.obj_ls[frame_id].attach_kps(name, kps)

        # track forward
        for idx in range(frame_id + 1, len(self.obj_ls)):
            previous_obj = self.obj_ls[idx - 1]
            previous_kps = previous_obj.kps_group[name]
            current_kps = previous_obj.trans_nonrigid.shift_kps(previous_kps, k_nbr=k_nbr)

            current_obj = self.obj_ls[idx]
            current_obj.attach_kps(name, current_kps)

        # track backward
        for idx in range(frame_id - 1, -1, -1):
            later_obj = self.obj_ls[idx + 1]
            later_kps = later_obj.kps_group[name]
            later_trans_invert = later_obj.trans_nonrigid.invert()
            current_kps = later_trans_invert.shift_kps(later_kps)

            current_obj = self.obj_ls[idx]
            current_obj.attach_kps(name, current_kps)
