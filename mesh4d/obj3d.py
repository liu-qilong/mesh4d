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
    def __init__(
        self,
        mesh: Union[None, pv.core.pointset.PolyData] = None,
        texture: Union[None, pv.core.objects.Texture] = None,
        **kwargs,
    ):
        self.mesh = mesh
        self.texture = texture

    def get_vertices(self) -> np.array:
        return np.array(self.mesh.points)
    
    def get_sample_points(self, sample_num: int) -> np.array:
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
        full_kps = kps.Kps()
        points = self.get_sample_points(sample_num)

        for idx in range(len(points)):
            full_kps.add_point("point {}".format(idx), points[idx])

        return full_kps
        
    def get_width(self) -> float:
        left = measure.points_get_max_bound(self.get_vertices())[0]
        right = measure.points_get_min_bound(self.get_vertices())[0]
        return left - right

    def show(self, off_screen: bool = False, elements: str = 'mp', is_export: bool = False, export_folder: str = '', export_name: str = 'screeenshot') -> pv.Plotter:
        scene = pv.Plotter(off_screen=off_screen)

        width = self.get_width()

        if 'm' in elements:
            self.add_mesh_to_scene(scene)

        if 'p' in elements:
            self.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-6*width)

        scene.camera_position = 'xy'
        scene.show(interactive_update=True)

        if is_export:
            export_path = os.path.join(export_folder, f'{export_name}.png')
            scene.update()
            scene.screenshot(export_path)
            
            if mesh4d.output_msg:
                print("export image: {}".format(export_path))

        return scene

    def add_mesh_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), show_edges: bool =True, **kwargs) -> pv.Plotter:
        scene.add_mesh(self.mesh.translate(location, inplace=False), show_edges=show_edges, **kwargs)

    def add_pcd_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), **kwargs) -> pv.Plotter:
        points = self.get_vertices()
        scene.add_points(points + location, **kwargs)


class Obj3d_Kps(Obj3d):
    def __init__(self, **kwargs):
        Obj3d.__init__(self, **kwargs)
        self.kps_group = {}

    def attach_kps(self, name: str, kps: Type[kps.Kps]):
        self.kps_group[name] = kps
    
    def load_kps_from_markerset(self, name: str, markerset: Type[kps.MarkerSet], time: float = 0.0):
        kps = markerset.get_time_coord(time)
        self.attach_kps(name, kps)

    def show(self, off_screen: bool = False, kps_names: Union[None, list, tuple] = None, elements: str = 'mpk', is_export: bool = False, export_folder: str = '', export_name: str = 'screeenshot') -> pv.Plotter:
        scene = pv.Plotter(off_screen=off_screen)

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

        if is_export:
            export_path = os.path.join(export_folder, f'{export_name}.png')
            scene.update()
            scene.screenshot(export_path)
            
            if mesh4d.output_msg:
                print("export image: {}".format(export_path))

        return scene

    def add_kps_to_scene(self, scene: pv.Plotter, kps_names: Union[None, tuple, list] = None, location: np.array = np.array((0, 0, 0)), color: Union[None, str] = None, **kwargs) -> pv.Plotter:
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
    def __init__(self, **kwargs):
        Obj3d_Kps.__init__(self, **kwargs)
        self.trans_rigid = None
        self.trans_nonrigid = None

    def set_trans_rigid(self, trans_rigid: field.Trans_Rigid):
        self.trans_rigid = trans_rigid

    def set_trans_nonrigid(self, trans_nonrigid: field.Trans_Nonrigid):
        self.trans_nonrigid = trans_nonrigid

    def get_deform_obj3d(self, mode: str = "nonrigid"):
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
    pcd_copy = copy.deepcopy(pcd)
    return np.asarray(pcd_copy.points)


def np2pcd(points: np.array) -> o3d.cpu.pybind.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def load_mesh_series(
        folder: str,
        start: int = 0,
        end: int = 1,
        stride: int = 1,
        data_type: str = '.obj',
        load_texture: bool = True
    ) -> tuple:
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files if data_type in f]
    files.sort()

    mesh_ls = []
    texture_ls = []

    for n in range(start, end + 1, stride):
        filedir = files[n]
        mesh_ls.append(pv.read(filedir))

        if load_texture:
            texture_ls.append(pv.read_texture(filedir.replace(data_type, '.jpg')))
        
        if mesh4d.output_msg:
            percent = (n - start + 1) / (end - start + 1)
            utils.progress_bar(percent, back_str=" loading: {}".format(filedir))

    return mesh_ls, texture_ls


def init_obj_series(
        mesh_ls: Iterable[pv.core.pointset.PolyData],
        texture_ls: Union[Iterable[pv.core.objects.Texture], None] = None,
        obj_type: Type[Obj3d] = Obj3d,
        **kwargs,
    ) -> Iterable[Type[Obj3d]]:
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