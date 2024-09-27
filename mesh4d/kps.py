from __future__ import annotations
from typing import Type, Union, Iterable

import os
import copy
import random
import numpy as np
import pandas as pd
import pyvista as pv
from scipy import interpolate
import matplotlib.colors as mcolors

import mesh4d
import mesh4d.config.param
from mesh4d import field, utils
from mesh4d.analyse import visual

class Kps(object):
    def __init__(self):
        self.points = {}

    def add_point(self, name: str, coord: np.array):
        self.points[name] = coord

    def get_points_coord(self, names: Union[None, list, tuple] = None) -> np.array:
        if names is None:
            points = [np.expand_dims(coord, axis=0) for coord in self.points.values()]
        else:
            points = [np.expand_dims(self.points[name], axis=0) for name in names]

        return np.concatenate(points)

    @staticmethod
    def diff(kps1: Type[Kps], kps2: Type[Kps]) -> dict:
        names = kps1.points.keys()
        points1 = kps1.get_points_coord(names)
        points2 = kps2.get_points_coord(names)

        disp = points1 - points2
        dist = np.linalg.norm(disp, axis=1)
        dist_mean = np.mean(dist)
        dist_std = np.std(dist)

        diff_dict = {
            'disp': disp,
            'dist': dist,
            'dist_mean': dist_mean,
            'dist_std': dist_std,
            'diff_str': "{:.3} ± {:.3} (mm)".format(dist_mean, dist_std)
        }
        return diff_dict

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

    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), radius: float = 1, **kwargs) -> pv.Plotter:
        pvpcd_kps = visual.np2pvpcd(self.get_points_coord(), radius=radius)
        scene.add_mesh(pvpcd_kps.translate(location, inplace=False), **kwargs)


class Marker(object):
    def __init__(self, name: str, start_time: float = 0, fps: int = 100):
        self.name = name
        self.start_time = start_time
        self.fps = fps

        # x, y, z data are stored in 3xN numpy array
        self.coord = None  # coordinates
        self.speed = None  # speed
        self.accel = None  # acceleration

        self.x_field = None
        self.y_field = None
        self.z_field = None

    def append_data(self, coord: np.array, speed: float = 0, accel: float = 0):
        # adjust array layout
        coord = np.expand_dims(coord, axis=0).T
        speed = np.expand_dims(speed, axis=0)
        accel = np.expand_dims(accel, axis=0)

        # if self.coord, self.speed, and self.accel haven't been initialised, initialise them
        # otherwise, append the newly arrived data to its end
        if self.coord is None:
            self.coord = coord
        else:
            self.coord = np.concatenate((self.coord, coord), axis=1)

        if self.speed is None:
            self.speed = speed
        else:
            self.speed = np.concatenate((self.speed, speed), axis=0)
        
        if self.accel is None:
            self.accel = accel
        else:
            self.accel = np.concatenate((self.accel, accel), axis=0)

    def fill_data(self, data_input: np.array):
        if self.coord is None:
            self.coord = data_input
        elif self.speed is None:
            self.speed = data_input
        elif self.accel is None:
            self.accel = data_input

    def get_frame_num(self) -> int:
        return self.coord.shape[1]
    
    def get_duration(self) -> float:
        return (self.get_frame_num() - 1) / self.fps

    def interp_field(self, kind: str = 'quadratic'):
        if self.coord is None:
            if mesh4d.output_msg:
                print("coordinates information not found")

            return

        frame_range = range(len(self.coord[0]))

        self.x_field = interpolate.interp1d(frame_range, self.coord[0], kind=kind)
        self.y_field = interpolate.interp1d(frame_range, self.coord[1], kind=kind)
        self.z_field = interpolate.interp1d(frame_range, self.coord[2], kind=kind)

    def get_frame_coord(self, frame_id: int) -> np.array:
        return self.coord[:, frame_id]

    def get_time_coord(self, time: float) -> np.array:
        if self.x_field is None:
            if mesh4d.output_msg:
                print("coordinates field need to be interped first")
            
            return

        frame_id = (time - self.start_time) * self.fps
        coord_interp = np.array(
            [self.x_field(frame_id),
             self.y_field(frame_id),
             self.z_field(frame_id)]
        )

        return coord_interp

    def reslice(self, fps: int = 120, start_time: Union[None, float] = None, end_time: Union[None, float] = None):
        if fps is None:
            fps = self.fps

        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.start_time + self.get_duration()

        marker = Marker(
            name=self.name,
            start_time=start_time,
            fps=fps,
            )
        
        for time in np.arange(start_time, end_time + 1/fps, 1/fps):
            marker.append_data(coord=self.get_time_coord(time))

        return marker

    @staticmethod
    def diff(marker1: Marker, marker2: Marker) -> dict:
        disp = []

        for frame in range(marker1.get_frame_num()):
            time = marker1.start_time + frame / marker1.fps
            coord1 = marker1.get_frame_coord(frame)
            coord2 = marker2.get_time_coord(time)
            disp.append(coord1 - coord2)
            
        dist = np.linalg.norm(disp, axis=1)
        dist_mean = np.mean(dist)
        dist_std = np.std(dist)

        diff_dict = {
            'disp': disp,
            'dist': dist,
            'dist_mean': dist_mean,
            'dist_std': dist_std,
            'diff_str': "{:.3} ± {:.3} (mm)".format(dist_mean, dist_std)
        }

        return diff_dict

    @staticmethod
    def concatenate(marker1: Type[Marker], marker2: Type[Marker]) -> Marker:
        marker = Marker(
            name=marker1.name, 
            start_time=marker1.start_time, 
            fps=marker1.fps,
            )
        
        if marker2.get_frame_num() == 1:
            marker2_reslice = marker2
        else:
            marker2_reslice = marker2.reslice(marker1.fps)

        
        marker.coord = np.concatenate((marker1.coord, marker2_reslice.coord), axis=1)
        marker.speed = np.concatenate((marker1.speed, marker2_reslice.speed), axis=0)
        marker.accel = np.concatenate((marker1.accel, marker2_reslice.accel), axis=0)

        marker.frame_num = marker.coord.shape[1]
        
        return marker
    
    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), trace_fps: float = 100, trace_width: float = 2, trace_op: float = 0.5, radius: float = 1, color: str = 'gold', **kwargs) -> pv.Plotter:
        points = self.coord.transpose()
        dots = visual.np2pvpcd(points, radius=radius)

        points_trace = [self.get_time_coord(t) for t in np.arange(
            self.start_time, 
            self.start_time + (len(points) - 1)/self.fps,
            1/trace_fps
            )]
        points_trace.append(points[-1])
        
        lines = pv.lines_from_points(points_trace)
        scene.add_mesh(dots.translate(location, inplace=False), color=color, **kwargs)
        scene.add_mesh(lines, color=color, line_width=trace_width, opacity=trace_op)

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


class MarkerSet(object):
    def __init__(self):
        self.markers = {}

    def load_from_vicon(self, filedir: str, trans_cab: Union[None, field.Trans_Rigid] = None):
        # trigger calibration parameters loading
        Marker('None')

        def parse(df, df_head):
            self.fps = df_head.values.tolist()[0][0]  # parse the fps
            col_names = df.columns.values.tolist()

            for col_id in range(len(col_names)):
                col_name = col_names[col_id]
                point_name = col_name.split('.')[0]

                # skip columns that contain NaN
                # (checking start from row 4, because for speed and acceleration the first few rows are empty)
                # or that follows the 'X' columns
                if df.loc[4:, col_name].isnull().values.any():
                    if mesh4d.output_msg:
                        percent = (col_id + 1) / len(col_names)
                        utils.progress_bar(percent, back_str=" parsing the {}-th column".format(col_id))

                    continue

                if 'Unnamed' in col_name:
                    if mesh4d.output_msg:
                        percent = (col_id + 1) / len(col_names)
                        utils.progress_bar(percent, back_str=" parsing the {}-th column".format(col_id))
                        
                    continue
                
                else:
                    # the first occurrence of a point
                    if point_name not in self.markers.keys():
                        self.markers[point_name] = Marker(name=point_name, fps=self.fps)

                    # fill the following 3 columns' X, Y, Z values into the point's object
                    try:
                        data_input = df.loc[2:, col_name:col_names[col_id+2]].to_numpy(dtype=float).transpose()

                        if trans_cab is not None:
                            data_input = trans_cab.shift_points(data_input.T).T

                        self.markers[point_name].fill_data(data_input)

                    except:
                        if mesh4d.output_msg:
                            print("error happended when loading kps file: column {}".format(col_name))

                    if mesh4d.output_msg:
                        percent = (col_id + 1) / len(col_names)
                        utils.progress_bar(percent, back_str=" parsing the {}-th column".format(col_id))

        df = pd.read_csv(filedir, skiprows=2)  # skip the first two rows
        df_head = pd.read_csv(filedir, nrows=1)  # only read the first two rows
        parse(df, df_head)
        
        if mesh4d.output_msg:
            print("loaded 1 vicon file: {}".format(filedir))

    def load_from_array(self, array: np.array, index: Union[None, list, tuple] = None, start_time: float = 0.0, fps: int = 120, trans_cab = None):
        self.fps = fps
        self.start_time = start_time
        point_num = array.shape[1]

        for idx in range(point_num):
            points = array[:, idx, :]

            if trans_cab is not None:
                points = trans_cab.shift_points(points)

            if index is None:
                self.markers[idx] = Marker(name=idx, start_time=self.start_time, fps=self.fps)
                self.markers[idx].fill_data(points.T)

            elif len(index) == point_num:
                self.markers[index[idx]] = Marker(name=index[idx], fps=self.fps)
                self.markers[index[idx]].fill_data(points.T)

            else:
                raise ValueError('length of index and point number must be the same')

    def interp_field(self, **kwargs):
        for point in self.markers.values():
            point.interp_field(**kwargs)

    def get_frame_coord(self, frame_id: int, kps_class: Type[Kps] = Kps) -> Type[Kps]:
        kps = kps_class()

        for name, marker in self.markers.items():
            coord = marker.get_frame_coord(frame_id)
            kps.add_point(name, coord)

        return kps
    
    def get_time_coord(self, time: float) -> np.array:
        kps = Kps()

        for name, marker in self.markers.items():
            coord = marker.get_time_coord(time)
            kps.add_point(name, coord)

        return kps
    
    def to_array(self) -> tuple:
        index = []
        array_ls = []

        for name, marker in self.markers.items():
            index.append(name)
            array_ls.append(marker.coord.T)

        # (marker, frame, axis) -> (frame, marker, axis)
        array = np.swapaxes(np.array(array_ls), 0, 1)

        return array, index
    
    def extract(self, marker_names: Iterable[str]) -> MarkerSet:
        markerset = MarkerSet()

        for marker_name in marker_names:
            marker_extract = copy.deepcopy(self.markers[marker_name])
            markerset.markers[marker_name] = marker_extract

        return markerset
    
    def split(self, marker_names: Iterable[str]) -> tuple:
        other_marker_names = []

        for name in self.markers.keys():
            if name not in marker_names:
                other_marker_names.append(name)

        return self.extract(marker_names), self.extract(other_marker_names)
    
    @staticmethod
    def diff(markerset1: MarkerSet, markerset2: MarkerSet) -> dict:
        # estimate the difference of each key point
        diff_dict = {}
        dist_ls = []

        for name in markerset1.markers.keys():
            diff = Marker.diff(markerset1.markers[name], markerset2.markers[name])
            diff_dict[name] = diff
            dist_ls.append(diff['dist'])

            if mesh4d.output_msg:
                print("estimated error of marker {}: {}".format(name, diff['diff_str']))

        # estimate the overall difference
        dist_array = np.array(dist_ls).reshape((-1,))
        dist_mean = np.mean(dist_array)
        dist_std = np.std(dist_array)

        # combine the estimation result and print the overall difference
        overall_diff_dict = {
            'diff_dict': diff_dict,
            'dist_mean': dist_mean,
            'dist_std': dist_std,
            'diff_str': "diff = {:.3} ± {:.3} (mm)".format(dist_mean, dist_std),
        }

        if mesh4d.output_msg:
            print("whole duration error: {}".format(overall_diff_dict['diff_str']))

        return overall_diff_dict

    def reslice(self, fps: int, start_time: Union[None, float] = None, end_time: Union[None, float] = None):
        markerset = MarkerSet()

        for name in self.markers.keys():
            markerset.markers[name] = self.markers[name].reslice(fps, start_time, end_time)

        return markerset

    @staticmethod
    def concatenate(markerset1: Type[Marker], markerset2: Type[Marker]) -> Marker:
        markerset = MarkerSet()

        for name in markerset1.markers.keys():
            markerset.markers[name] = Marker.concatenate(
                markerset1.markers[name],
                markerset2.markers[name],
            )

        return markerset
    
    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), trace_fps: float = 100, trace_width: float = 5, trace_op: float = 0.5, radius: float = 1, color: Union[str, None] = None, **kwargs) -> pv.Plotter:
        if color is None:
            # prepare random color select
            random_color = True
            seed = 26
            color_ls = list(mcolors.CSS4_COLORS.keys())
        else:
            random_color = False

        for marker in self.markers.values():
            # random color select
            if random_color:
                random.seed(seed)
                color = random.choice(color_ls)
                seed = seed + 1
            marker.add_to_scene(scene=scene, location=location, trace_fps=trace_fps, trace_width=trace_width, trace_op=trace_op, radius=radius, color=color, **kwargs)

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