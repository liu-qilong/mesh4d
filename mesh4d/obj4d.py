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
    def __init__(self, start_time: float = 0.0, fps: int = 120):
        self.obj_ls = []
        self.start_time = start_time
        self.fps = fps

    def add_obj(self, *objs: Iterable(Type[obj3d.Obj3d]), **kwargs):
        for obj in objs:
            self.obj_ls.append(obj)

    def show(self, off_screen: bool = False, elements: str = 'mp', stack_dist: float = 1000, zoom_rate: float = 3.5, window_size: list = [2000, 800], skip: int = 1, m_props: dict = {}, p_props: dict = {}, is_export: bool = False, export_folder: str = '', export_name: str = 'screeenshot') -> pv.Plotter:
        scene = pv.Plotter(off_screen=off_screen)
        plot_num = len(self.obj_ls)

        for idx in range(0, plot_num, skip):
            obj = self.obj_ls[idx]

            width = obj.get_width()

            if 'm' in elements:
                obj.add_mesh_to_scene(scene, location=[0, 0, idx * stack_dist / skip], **m_props)
            
            if 'p' in elements:
                obj.add_pcd_to_scene(scene, location=[1.5*width, 0, idx * stack_dist / skip], point_size=1e-5*width, **p_props)
            
        scene.camera_position = 'zy'
        scene.camera.azimuth = 45
        scene.camera.zoom(zoom_rate)
        scene.window_size = window_size
        scene.enable_parallel_projection()

        scene.show(interactive_update=True)

        if is_export:
            export_path = os.path.join(export_folder, f'{export_name}.png')
            scene.update()
            scene.screenshot(export_path)
            
            if mesh4d.output_msg:
                print("export image: {}".format(export_path))

        return scene
        
    def animate(self, export_folder: str = "output/", filename: str = "obj4d", file_type: str = 'mp4', fps: int = 12, mp4_quality: int = 5, elements: str = 'mp', m_props: dict = {}, k_props: dict = {}, p_props: dict = {}):
        scene = pv.Plotter(off_screen=True)

        if file_type == 'gif':
            scene.open_gif(os.path.join(export_folder, filename) + '.gif', framerate=fps)
        elif file_type == 'mp4':
            scene.open_moive(os.path.join(export_folder, filename) + '.mp4', framerate=fps, quality=mp4_quality)
        else:
            print('invalid file type')
            return
        
        plot_num = len(self.obj_ls)

        for idx in range(0, plot_num):
            obj = self.obj_ls[idx]
            scene.clear()

            width = obj.get_width()

            if 'm' in elements:
                obj.add_mesh_to_scene(scene, **m_props)
            
            if 'p' in elements:
                obj.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-3*width, **p_props)
            
            scene.camera_position = 'xy'
            scene.write_frame()

            if mesh4d.output_msg:
                percent = (idx + 1) / plot_num
                utils.progress_bar(percent, back_str=" exported the {}-th frame".format(idx))

        scene.close()


class Obj4d_Kps(Obj4d):
    def load_markerset(self, name: str, markerset: Union[kps.MarkerSet, None] = None):
        for idx in range(len(self.obj_ls)):
            obj = self.obj_ls[idx]
            obj.load_kps_from_markerset(name, markerset, self.start_time + idx / self.fps)

    def assemble_markerset(self, name: str, start_id: int = 0) -> kps.MarkerSet:
        markerset = kps.MarkerSet()
        markerset.fps = self.fps

        for obj in self.obj_ls[start_id:]:
            points_dict = obj.kps_group[name].points

            for point_name in points_dict.keys():
                if point_name not in markerset.markers.keys():
                    markerset.markers[point_name] = kps.Marker(name=point_name, fps=self.fps)
                
                markerset.markers[point_name].append_data(coord=points_dict[point_name])

        return markerset
    
    def show(self, off_screen: bool = False, kps_names: Union[None, list, tuple] = None, elements: str = 'mpk', stack_dist: float = 1000, zoom_rate: float = 3.5, window_size: list = [2000, 800], skip: int = 1, m_props: dict = {}, k_props: dict = {}, p_props: dict = {}, is_export: bool = False, export_folder: str = '', export_name: str = 'screeenshot') -> pv.Plotter:
        scene = pv.Plotter(off_screen=off_screen)
        plot_num = len(self.obj_ls)

        for idx in range(0, plot_num, skip):
            obj = self.obj_ls[idx]
            width = obj.get_width()

            if 'm' in elements:
                obj.add_mesh_to_scene(scene, location=[0, 0, idx * stack_dist / skip], **m_props)

                if 'k' in elements:
                    obj.add_kps_to_scene(scene, kps_names, location=[0, 0, idx * stack_dist / skip], radius=0.02*width, **k_props)
            
            if 'p' in elements:
                obj.add_pcd_to_scene(scene, location=[1.5*width, 0, idx * stack_dist / skip], point_size=1e-5*width, **p_props)

                if 'k' in elements:
                    obj.add_kps_to_scene(scene, kps_names, location=[1.5*width, 0, idx * stack_dist / skip], radius=0.02*width, **k_props)

            if ('m' not in elements) and ('p' not in elements):
                if 'k' in elements:
                    obj.add_kps_to_scene(scene, kps_names, location=[0, 0, idx * stack_dist / skip], radius=0.02*width, **k_props)
            
        scene.camera_position = 'zy'
        scene.camera.azimuth = 45
        scene.camera.zoom(zoom_rate)
        scene.window_size = window_size
        scene.enable_parallel_projection()
        scene.show(interactive_update=True)

        if is_export:
            export_path = os.path.join(export_folder, f'{export_name}.png')
            scene.update()
            scene.screenshot(export_path)
            
            if mesh4d.output_msg:
                print("export image: {}".format(export_path))

        return scene

    def animate(self, export_folder: str = "output/", filename: str = "obj4d", file_type: str = 'mp4', fps: int = 12, mp4_quality: int = 5, kps_names: Union[None, list, tuple] = None, elements: str = 'mpk', m_props: dict = {}, k_props: dict = {}, p_props: dict = {}, k_radius_factor: float = 0.02):
        scene = pv.Plotter(off_screen=True)
        
        if file_type == 'gif':
            scene.open_gif(os.path.join(export_folder, filename) + '.gif', fps=fps)
        elif file_type == 'mp4':
            scene.open_movie(os.path.join(export_folder, filename) + '.mp4', framerate=fps, quality=mp4_quality)
        else:
            print('invalid file type')
            return

        plot_num = len(self.obj_ls)

        for idx in range(0, plot_num):
            obj = self.obj_ls[idx]
            scene.clear()
            
            width = obj.get_width()

            if 'm' in elements:
                obj.add_mesh_to_scene(scene, **m_props)

                if 'k' in elements:
                    obj.add_kps_to_scene(scene, kps_names, radius=k_radius_factor * width, **k_props)

            if 'p' in elements:
                obj.add_pcd_to_scene(scene, location=[1.5*width, 0, 0], point_size=1e-3*width, **p_props)

                if 'k' in elements:
                    obj.add_kps_to_scene(scene, kps_names, radius=k_radius_factor * width, location=[1.5*width, 0, 0], **k_props)

            if ('m' not in elements) and ('p' not in elements):
                if 'k' in elements:
                    obj.add_kps_to_scene(scene, kps_names, radius=k_radius_factor * width, **k_props)
            
            scene.camera_position = 'xy'
            scene.write_frame()

            if mesh4d.output_msg:
                percent = (idx + 1) / plot_num
                utils.progress_bar(percent, back_str=" exported the {}-th frame".format(idx))

        scene.close()


class Obj4d_Deform(Obj4d_Kps):
    def __init__(self, enable_rigid: bool = False, enable_nonrigid: bool = False,  **kwargs):
        Obj4d_Kps.__init__(self, **kwargs)
        self.enable_rigid = enable_rigid
        self.enable_nonrigid = enable_nonrigid

    def regist(self, **kwargs):
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
        pass

    def process_rigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        trans = field.Trans_Rigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        trans = field.Trans_Nonrigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)

    def offset_rotate(self):
        for obj in self.obj_ls[1:]:  # the first 3d object doesn't need reorientation
            obj.offset_rotate()

        if mesh4d.output_msg:
            print("4d object reorientated")

    def vkps_track(self, kps: Type[kps.Kps], start_id: int = 0, name: str = 'vkps'):
        self.obj_ls[start_id].attach_kps(name, kps)

        track_num = 0
        total_num = len(self.obj_ls)

        # track forward
        for idx in range(start_id + 1, len(self.obj_ls)):
            previous_obj = self.obj_ls[idx - 1]
            previous_kps = previous_obj.kps_group[name]
            current_kps = previous_obj.trans_nonrigid.shift_kps(previous_kps)

            current_obj = self.obj_ls[idx]
            current_obj.attach_kps(name, current_kps)

            if mesh4d.output_msg:
                track_num = track_num + 1
                percent = (track_num + 1) / total_num
                utils.progress_bar(percent, back_str=" complete virtual landmark tracking at the {}-th frame".format(idx))

        # track backward
        """
        for idx in range(start_id - 1, -1, -1):
            later_obj = self.obj_ls[idx + 1]
            later_kps = later_obj.kps_group[name]
            later_trans_invert = later_obj.trans_nonrigid.invert()
            current_kps = later_trans_invert.shift_kps(later_kps)

            current_obj = self.obj_ls[idx]
            current_obj.attach_kps(name, current_kps)

            if mesh4d.output_msg:
                track_num = track_num + 1
                percent = track_num / total_num
                utils.progress_bar(percent, back_str=" complete virtual landmark tracking at the {}-th frame".format(idx))
        """
