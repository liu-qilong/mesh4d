"""The 4D object contains a series of 3D objects (:mod:`UltraMotionCapture.obj3d`). In :mod:`UltraMotionCapture.obj4d`, 4D object classes with different features and capabilities are developed, serving for different analysis needs and scenarios. At current stage, there are 3 types of 4D object:

- Static 4D object :class:`Obj4d`

  It contains a list of 3D objects.

- Static 4D object :class:`Obj4d_Kps` with key points

  It's derived from :class:`Obj4d` and attach key points (:class:`UltraMotionCapture.kps.Kps`) to each of the 3D object via Vicon motion capture data (:class:`UltraMotionCapture.kps.MarkerSet`).

- Dynamic/Deformable 4D object :class:`Obj4d_Deform`

  It's derived from :class:`Obj4d_Kps` and attach the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) and non-rigid deformation (:class:`UltraMotionCapture.field.Trans_Nonrigid`) to each of the 3D object by registration.
"""
from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np

import UltraMotionCapture
import UltraMotionCapture.config.param
from UltraMotionCapture import obj3d
from UltraMotionCapture import kps
from UltraMotionCapture import field

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


class Obj4d_Kps(Obj4d):
    """Static 4D object :class:`Obj4d_Kps` with key points. Derived from :class:`Obj4d` and attach key points (:class:`UltraMotionCapture.kps.Kps`) to each of the 3D object via Vicon motion capture data (:class:`UltraMotionCapture.kps.MarkerSet`).

    Parameters
    ---
    **kwargs
        configuration parameters of the base classes (:class:`Obj3d`) can be passed in via :code:`**kwargs`.

    Example
    ---
    Load Vicon motion capture data (:class:`UltraMotionCapture.kps.MarkerSet`) when initialising the 4D object. And use :func:`load_obj_series` to load a list of 3D objects and add them to the 4D object: ::

        from UltraMotionCapture import obj3d, obj4d, kps

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
            markerset=vicon,
            fps=120,
        )

        o4.add_obj(*o3_ls)
    """
    def add_obj(self, *objs: Iterable[Type[obj3d.Obj3d_Kps]], **kwargs):
        """ Add object(s) and attach key points (:class:`UltraMotionCapture.kps.Kps`) to each of the 3D object via Vicon motion capture data (:attr:`markerset`).
        
        Parameters
        ---
        *objs
            unspecified number of 3D objects.

            .. warning::
            
                The 3D objects' class must derived from :class:`UltraMotionCapture.obj3d.Obj3d_Kps`.

            .. seealso::

                About the :code:`*` symbol and its effect, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_
        
        **kwargs
            configuration parameters of the base classes (:class:`Obj3d`)'s :meth:`add_obj` method can be passed in via :code:`**kwargs`.

        Example
        ---
        Let's say we have two 3D objects :code:`o3_a`, :code:`o3_b` and 4D object :code:`o4`. 3D objects can be passed into the :meth:`add_obj` method one by one: ::

            o4.add_obj(o3_a, o3_b)

        3D objects can be passed as a list: ::

            o3_ls = [o3_a, o3_b]
            o4.add_obj(*o3_ls)
        """
        Obj4d.add_obj(self, *objs, **kwargs)

    def load_markerset(self, name: str, markerset: Union[kps.MarkerSet, None] = None):
        """Slice the :class:`~UltraMotionCapture.kps.MarkerSet` object into :class:`~UltraMotionCapture.kps.kps` objects and attached them to the corresponding frames.

        Parameters
        ---
        name
            the name of the :class:`~UltraMotionCapture.kps.Kps` objects. Noted that this name will also been used as its keyword in :class:`~UltraMotionCapture.obj3d.Obj3d_Kps` object's :attr:`kps_group` attribute.
        markerset
            Vicon motion capture data (:class:`UltraMotionCapture.kps.MarkerSet`).
        """
        for idx in range(len(self.obj_ls)):
            obj = self.obj_ls[idx]
            obj.load_kps(name, markerset, self.start_time + idx / self.fps)


class Obj4d_Deform(Obj4d_Kps):
    """Dynamic/Deformable 4D object :class:`Obj4d_Deform`. Derived from :class:`Obj4d_Kps` and attach the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) and non-rigid deformation (:class:`UltraMotionCapture.field.Trans_Nonrigid`) to each of the 3D object by registration.

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
    Load Vicon motion capture data (:class:`UltraMotionCapture.kps.MarkerSet`) when initialising the 4D object. Use :func:`load_obj_series` to load a list of 3D objects. And then add them to the 4D object. In the procedure of adding, the program will implement the activated transformation estimation automatically: ::

        from UltraMotionCapture import obj3d, kps, obj4d

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
            markerset=vicon,
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

    def add_obj(self, *objs: Iterable[Type[obj3d.Obj3d_Deform]], **kwargs):
        """ Add object(s) and attach key points (:class:`UltraMotionCapture.kps.Kps`) to each of the 3D object via Vicon motion capture data (:attr:`markerset`). And then implement the activated transformation estimation.
        
        Danger
        ---
        This method shall only be called for one time.

        Parameters
        ---
        *objs
            unspecified number of 3D objects.

            .. warning::
            
                The 3D objects' class must derived from :class:`UltraMotionCapture.obj3d.Obj3d_Deform`.

            .. seealso::

                About the :code:`*` symbol and its effect, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_
        
        **kwargs
            configuration parameters for the registration and the configuration parameters of the base classes (:class:`Obj3d` and :class:`Obj3d_Kps`)'s :meth:`add_obj` method can be passed in via :code:`**kwargs`.

            .. seealso::

                Technically, the configuration parameters for the registration are passed to :meth:`UltraMotionCapture.field.Trans_Rigid.regist` for rigid transformation and :meth:`UltraMotionCapture.field.Trans_Nonrigid.regist`, and they then call :mod:`probreg`'s registration method.
                
                For accepted parameters, please refer to `probreg.cpd.registration_cpd <https://probreg.readthedocs.io/en/latest/probreg.html?highlight=registration_cpd#probreg.cpd.registration_cpd>`_.

        Example
        ---
        Let's say we have two 3D objects :code:`o3_a`, :code:`o3_b` and 4D object :code:`o4`. 3D objects can be passed into the :meth:`add_obj` method one by one: ::

            o4.add_obj(o3_a, o3_b)

        3D objects can be passed as a list: ::

            o3_ls = [o3_a, o3_b]
            o4.add_obj(*o3_ls)
        """
        reg_start_index = len(self.obj_ls)
        Obj4d_Kps.add_obj(self, *objs, **kwargs)
        reg_end_index = len(self.obj_ls) - 1
        
        for idx in range(reg_start_index, reg_end_index + 1):
            if idx == 0:
                self.process_first_obj()
                continue

            if self.enable_rigid:
                self.process_rigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the previous one

            if self.enable_nonrigid:
                self.process_nonrigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the later one

    def process_first_obj(self):
        """Process the first added 3D object.
        
        Attention
        ---
        Called by :meth:`add_obj`."""
        pass

    def process_rigid_dynamic(self, idx_source, idx_target, **kwargs):
        """Estimate the rigid transformation of the added 3D object. The lastly added 3D object is used as source object and the newly added 3D object as the target object.

        Parameters
        ---
        idx_source
            the index of the source 3D object in :code:`self.obj_ls`.
        idx_target
            the index of the target 3D object in :code:`self.obj_ls`.

        Attention
        ---
        The estimated transformation is load to the source object, via its :meth:`UltraMotionCapture.obj3d.Obj3d_Deform.set_trans_rigid` method.
        
        Attention
        ---
        Called by :meth:`add_obj`."""
        trans = field.Trans_Rigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def process_nonrigid_dynamic(self, idx_source, idx_target, **kwargs):
        """Estimate the non-rigid transformation of the added 3D object. The lastly added 3D object is used as source object and the newly added 3D object as the target object.

        Parameters
        ---
        idx_source
            the index of the source 3D object in :code:`self.obj_ls`.
        idx_target
            the index of the target 3D object in :code:`self.obj_ls`.

        Attention
        ---
        The estimated transformation is load to the source object, via its :meth:`UltraMotionCapture.obj3d.Obj3d_Deform.set_trans_nonrigid` method.
        
        Attention
        ---
        Called by :meth:`add_obj`."""
        trans = field.Trans_Nonrigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)

    def offset_rotate(self):
        """Offset the rotation according to the estimated rigid transformation.

        Tip
        ---
        This method usually serves for reorientate all 3D objects to a referencing direction, since that the rigid transformation (:class:`UltraMotionCapture.field.Trans_Rigid`) is estimated one follow one in the 3D objects list.
        
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

        if UltraMotionCapture.output_msg:
            print("4d object reorientated")

    def vkps_track(self, kps: Type[kps.Kps], start_frame: int = 0, end_frame: int = -1):
        """tbf"""
        pass