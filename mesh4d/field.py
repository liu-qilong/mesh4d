from __future__ import annotations
from typing import Type, Union, Iterable

import os
import copy
import numpy as np
import pyvista as pv
import open3d as o3d
from probreg import cpd
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d, kps
from mesh4d.analyse import measure

class Trans(object):
    def __init__(self, source_obj: Union[None, Type[obj3d.Obj3d]] = None, target_obj: Union[None, Type[obj3d.Obj3d]] = None, **kwargs):
        self.source = source_obj
        self.target = target_obj

    def shift_points(self, points: np.array, **kwargs) -> np.array:
        return points

    def shift_kps(self, kps: Type[kps.Kps], **kwargs) -> Type[kps.Kps]:
        deform_kps = type(kps)()
        
        for name, coord in kps.points.items():
            point = np.array((coord, ))
            coord_deform = self.shift_points(point, **kwargs)
            deform_kps.add_point(name, coord_deform[0])
        
        return deform_kps

    def shift_mesh(self, mesh: pv.core.pointset.PolyData, **kwargs) -> pv.core.pointset.PolyData:
        mesh_deform = copy.deepcopy(mesh)
        mesh_deform.points = self.shift_points(mesh_deform.points, **kwargs)
        return mesh_deform

    def show(self, off_screen: bool = False, is_export: bool = False, export_folder: str = '', export_name: str = 'screeenshot') -> pv.Plotter:
        scene = pv.Plotter(off_screen=off_screen)
        self.add_to_scene(scene)
        scene.camera_position = 'xy'
        scene.show(interactive_update=True)

        if is_export:
            export_path = os.path.join(export_folder, f'{export_name}.png')
            scene.update()
            scene.screenshot(export_path)
            
            if mesh4d.output_msg:
                print("export image: {}".format(export_path))

        return scene

    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), **kwargs) -> pv.Plotter:
        pass


class Trans_Rigid(Trans):
    def regist(self, sample_num: int = 1000, **kwargs):
        source_pcd = obj3d.np2pcd(self.source.mesh.get_sample_points(sample_num=sample_num))
        target_pcd = obj3d.np2pcd(self.target.mesh.get_sample_points(sample_num=sample_num))

        tf_param, _, _ = cpd.registration_cpd(
            source_pcd, target_pcd, 'rigid', **kwargs
        )
        self.parse(tf_param)
        self.fix()

    def parse(self, tf_param: Type[cpd.CoherentPointDrift]):
        self.rot = tf_param.rot
        self.scale = tf_param.scale
        self.t = tf_param.t

    def fix(self):
        if np.abs(self.scale - 1) > 0.05 and mesh4d.output_msg:
            print("warnning: large rigid scale {}".format(self.scale))

    def shift_points(self, points: np.array) -> np.array:
        
        # return self.scale * np.matmul(points, self.rot.T) + np.expand_dims(self.t, axis=0)
        return (self.scale * np.matmul(self.rot, points.T) + np.expand_dims(self.t, axis=1)).T

    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), original_length: Union[None, float] = None, **kwargs) -> pv.Plotter:
        vectors = np.array([
            [0, 0, 0],  # origin
            [1, 0, 0],  # x axis
            [0, 1, 0],  # y axis
            [0, 0, 1],  # z axis
        ])
        vectors_deform = self.shift_points(vectors)

        if original_length is None:
            original_length = np.linalg.norm(vectors_deform[0] - vectors[0])/2

        def add_axes(scene, vectors, shaft_radius=0.02, tip_radius=0.05, opacity=1):
            param = {
                'start': vectors[0],
                'scale': np.linalg.norm(vectors[1] - vectors[0]) * original_length,
                'shaft_radius': shaft_radius,
                'tip_radius': tip_radius,   
            }
            arrow_x = pv.Arrow(direction=vectors[1] - vectors[0], **param)
            arrow_y = pv.Arrow(direction=vectors[2] - vectors[0], **param)
            arrow_z = pv.Arrow(direction=vectors[3] - vectors[0], **param)

            scene.add_mesh(arrow_x.translate(location, inplace=True), color='gold', opacity=opacity)
            scene.add_mesh(arrow_y.translate(location, inplace=True), color='teal', opacity=opacity)
            scene.add_mesh(arrow_z.translate(location, inplace=True), color='darkolivegreen', opacity=opacity)

        add_axes(scene, vectors, opacity=0.3)
        add_axes(scene, vectors_deform)

    def invert(self):
        trans_invert = type(self)(source_obj=self.target, target_obj=self.source)
        trans_invert.scale = 1/self.scale
        trans_invert.rot = self.rot.T
        trans_invert.t = -1/self.scale * np.matmul(self.t, self.rot)
        # the last operation seems to be different from the formula, this is because the default vector in linear algebra is column vector, while in numpy is line vector

        return trans_invert


class Trans_Nonrigid(Trans):
    def regist(self, field_nbr: int = 100):
        self.source_points = self.source.get_vertices()
        self.deform_points = measure.nearest_points_from_plane(self.target.mesh, self.source_points)
        self.field = RBFInterpolator(self.source_points, self.deform_points, neighbors=field_nbr)

    def shift_points(self, points: np.array) -> np.array:
        return self.field(points)

    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), **kwargs) -> pv.Plotter:
        pdata = pv.vector_poly_data(self.source_points, self.deform_points - self.source_points)
        glyph = pdata.glyph()
        scene.add_mesh(glyph.translate(location, inplace=False), **kwargs)

    def invert(self):
        trans_invert = type(self)(source_obj=self.target, target_obj=self.source)
        trans_invert.field_nbr = self.field_nbr
        trans_invert.source_points = self.deform_points
        trans_invert.deform_points = self.source_points
        trans_invert.field = RBFInterpolator(trans_invert.source_points, trans_invert.deform_points, neighbors=trans_invert.field_nbr)

        return trans_invert


def transform_rst2sm(R: np.array, s: float, t: np.array) -> tuple[float, np.array]:
    M = np.diag(np.full(4, 1, dtype='float64'))
    M[0:3, 0:3] = R
    M[0:3, 3] = t/s
    return float(s), M


def transform_sm2rst(s: float, M: np.array) -> tuple[np.array, float, np.array]:
    R = M[0:3, 0:3]
    t = M[0:3, 3]*s
    return R, s, t