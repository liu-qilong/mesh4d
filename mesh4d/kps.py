"""The :mod:`mesh4d.kps` module stands for *key points*. In :mod:`mesh4d` package, key points are essential elements to facilitate the processing of 4D images.

There are two different perspectives to arrange key points data: *time-wise* and *point-wise*. Reflecting these two ways of arrangement:

- The :class:`Kps` contain all key points' data at a specific moment;
- While the :class:`Marker` contains a specific key point's data within a time period. To aggregate all key points' data, :class:`MarkerSet` is provided.
"""
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
    """A collection of the key points that can be attached to a 3D object, i.e. a frame of the 4D object.

    Note
    ---
    `Class Attributes`

    self.points
        :math:`N` key points in 3D space stored in a dictionary.

    Example
    ---
    After initialisation, the :class:`Kps` object is empty. Load key points from Vicon motion capture data stored in a :class:`MarkerSet` object with :meth:`load_from_markerset_frame` or :meth:`load_from_markerset_time`. ::

        from mesh4d import kps
        
        vicon = kps.MarkerSet()
        vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
        vicon.interp_field()

        points = kps.Kps()
        points.load_from_markerset_time(vicon, time = 1.01)
    """
    def __init__(self):
        self.points = {}

    def add_point(self, name: str, coord: np.array):
        """Add a point to :attr:`self.points` with its name and its coordinates represented in a (3, ) :class:`numpy.array`.

        Parameters
        ---
        name
            the name of the added point.
        point
            (3, ) :class:`numpy.array`.
        """
        self.points[name] = coord

    def get_points_coord(self, names: Union[None, list, tuple] = None) -> np.array:
        """Concatenating all coordinates into a (N, 3) :class:`numpy.array` and return.
        
        Parameters
        ---
        names
            the names of the points to be retrieved. If doesn't input this parameter, all points will be returned.

        Returns
        ---
        :class:`numpy.array`
            coordinates stored in a (N, 3) :class:`numpy.array` in the order of the :attr:`names`.
        """
        if names is None:
            points = [np.expand_dims(coord, axis=0) for coord in self.points.values()]
        else:
            points = [np.expand_dims(self.points[name], axis=0) for name in names]

        return np.concatenate(points)

    @staticmethod
    def diff(kps1: Type[Kps], kps2: Type[Kps]) -> dict:
        """Compute the difference of one key points object with another.

        Warning
        ---
        These two key points objects must contain the same number of points with the same names.

        Parameters
        ---
        kps1
            a key points object.
        kps2
            another key points object.

        Returns
        ---
        :class:`dict`
            A dictionary that contains the comparison result:

            - :code:`'disp'`: the displacement vectors from the predicted key points to the ground truth key points stored in a (N, 3) :class:`numpy.array`.
            - :code:`'dist'`: the distances from the first key points to the second key points stored in a (N, ) :class:`numpy.array`.
            - :code:`'dist_mean'`: the mean distances from the first key points to the second key points.
            - :code:`'dist_std'`: the standard deviation of distances.
            - :code:`'diff_str'`: a string in form of :code:`'dist_mean ± dist_std (mm)`.
        """
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

    def show(self) -> pv.Plotter:
        """Illustrate the key points object.
        """
        scene = pv.Plotter()
        self.add_to_scene(scene)

        scene.camera_position = 'xy'
        scene.show(interactive_update=True)
        
        return scene

    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), radius: float = 1, **kwargs) -> pv.Plotter:
        """Add the visualisation of current object to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        location
            the displace location represented in a (3, ) :class:`numpy.array`.
        radius
            radius of the key points

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
        pvpcd_kps = visual.np2pvpcd(self.get_points_coord(), radius=radius)
        scene.add_mesh(pvpcd_kps.translate(location, inplace=False), **kwargs)


class Marker(object):
    """Storing single key point's coordinates data within a time period. Usually loaded from the Vicon motion capture data. In this case, a key point is also referred as a marker.

    Parameters
    ---
    name
        the name of the marker.
    start_time
        the start time of the coordinates data.
    fps
        the number of frames per second (fps).

        Attention
        ---
        Noted that the original unit of Vicon raw data is millimetre (mm).

    Note
    ---
    self.name
        The name of the marker.
    self.start_time
        The start time of the coordinates data.
    self.fps
        The number of frames per second (fps).
    self.coord
        (3, N) :class:`numpy.array` storing the :code:`N` frames of coordinates data.
    self.speed
        (N, ) :class:`numpy.array` storing :code:`N` frames of speed data.
    self.accel
        (N, ) :class:`numpy.array` storing :code:`N` frames of acceleration data.
    self.x_field
        An :class:`scipy.interpolate.interp1d` object that storing the interpolated function of the :math:`x` coordinates of all frames. Used for estimated the :math:`x` coordinate of any intermediate time between frames.
    self.y_field
        An :class:`scipy.interpolate.interp1d` object that storing the interpolated function of the :math:`y` coordinates of all frames. Used for estimated the :math:`y` coordinate of any intermediate time between frames.
    self.z_field
        An :class:`scipy.interpolate.interp1d` object that storing the interpolated function of the :math:`z` coordinates of all frames. Used for estimated the :math:`z` coordinate of any intermediate time between frames.

    Tip
    ---
    When loading Vicon motion capture data, the whole process is arranged by a :class:`MakerSet` object, which creates :class:`Marker` objects for each key point and loads data into it accordingly. Therefore, usually the end user doesn't need to touch the :class:`Marker` class.

    Tip
    ---
    In other parts of the package, points coordinates storing in :class:`numpy.array` are usually in the shape of (N, 3), while here we adopt (3, N), for the convenience of data interpolation :meth:`interp_field`.
    """

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
        """Append a frame of coordinates, speed, and acceleration data. after transforming to 3dMD coordinates.

        Parameters
        ---
        coord
            (3, ) :class:`numpy.array` storing the coordinates data.
        speed
            the speed storing in a :class:`float`. Default as :code:`0`.
        accel
            the acceleration storing in a :class:`float`. Default as :code:`0`.
        """
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
        """Fill coordinates, speed, and acceleration data of all frames after transforming to 3dMD coordinates. Noted that the first calling fills the coordinates data, the second calling fills the speed data, and the third calling fills the acceleration data, respectively.

        Parameters
        ---
        data_input

            - (3, N) :class:`numpy.array` when loading coordinates data.
            - Or (N, ) :class:`numpy.array` for loading speed data or acceleration data.

        Attention
        ---
        Other than appending data frame by frame, as :meth:`append_data` does, it's more convenient to load the data at one go when data loading data from a parsed Vicon motion capture data (:meth:`MarkerSet.load_from_vicon`). This method is designed for this purpose. Usually the end user don't need to call this method manually.
        """
        if self.coord is None:
            self.coord = data_input
        elif self.speed is None:
            self.speed = data_input
        elif self.accel is None:
            self.accel = data_input

    def get_frame_num(self) -> int:
        """Get the number of frames.
        """
        return self.coord.shape[1]
    
    def get_duration(self) -> float:
        """Get the whole time duration.
        """
        return (self.get_frame_num() - 1) / self.fps

    def interp_field(self, kind: str = 'quadratic'):
        """Interpolating the :math:`x, y, z` coordinates data to estimate its continues change. After that, the coordinates at the intermediate time between frames is accessible.

        Warnings
        ---
        Before interpolation, the coordinates data, i.e. :attr:`self.coord`, must be properly loaded.

        Parameters
        ---
        kind
            kind of interpolation.

            .. seealso::
                `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_
        """
        if self.coord is None:
            if mesh4d.output_msg:
                print("coordinates information not found")

            return

        frame_range = range(len(self.coord[0]))

        self.x_field = interpolate.interp1d(frame_range, self.coord[0], kind=kind)
        self.y_field = interpolate.interp1d(frame_range, self.coord[1], kind=kind)
        self.z_field = interpolate.interp1d(frame_range, self.coord[2], kind=kind)

    def get_frame_coord(self, frame_id: int) -> np.array:
        """Get coordinates data according to frame id.
        
        Parameters
        ---
        frame_id
            index of the frame to get coordinates data.
        
        Return
        ---
        :class:`numpy.array`
            The structure of the returned array is :code:`array[0-2 as x-z][frame_id]`
        """
        return self.coord[:, frame_id]

    def get_time_coord(self, time: float) -> np.array:
        """Get coordinates data according to time stamp.

        Parameters
        ---
        time
            time stamp to get coordinates data.

        Return
        ---
        :class:`numpy.array`
            The structure of the returned array is :code:`array[0-2 as x-z][time]`
        
        Warnings
        ---
        The interpolation must be properly done before accessing coordinates data according to time stamp, which means the :meth:`interp_field` must be called first.
        """
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

    def reslice(self, fps_new: int = 120):
        """Return the marker object re-slicing to another frame rate. Noted that the original object won't be altered.
        
        Attention
        ---
        The new marker set object haven't undergo :meth:`interp_field`.

        Parameters
        ---
        fps_new
            the new frame rate (frames per second)
        """
        marker = Marker(
            name=self.name,
            start_time=self.start_time,
            fps=fps_new,
            )
        
        frame_num_new = int(self.get_duration() * fps_new) + 1

        for idx in range(frame_num_new):
            time = self.start_time + idx / fps_new
            marker.append_data(coord=self.get_time_coord(time))

        return marker

    @staticmethod
    def diff(marker1: Marker, marker2: Marker) -> dict:
        """Compute the difference of one marker object with another.

        Warning
        ---
        For a frame from the first marker, its difference with the second marker is computed referencing with the corresponding time slice of the second marker. Therefore, the second marker must contain the time period of the first object.

        Parameters
        ---
        marker1
            a marker object.
        marker2
            another marker object.

        Returns
        ---
        :class:`dict`
            A dictionary that contains the comparison result:

            - :code:`'disp'`: the displacement vectors from the predicted key points to the ground truth key points stored in a (N, 3) :class:`numpy.array`.
            - :code:`'dist'`: the distances from the first marker to the second marker stored in a (N, ) :class:`numpy.array`.
            - :code:`'dist_mean'`: the mean distances from the first marker to the second marker.
            - :code:`'dist_std'`: the standard deviation of distances.
            - :code:`'diff_str'`: a string in form of :code:`'dist_mean ± dist_std (mm)`.
        """
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
        """Concatenate two marker object.

        Attention
        ---
        If the second marker object has more than 1 frame of data, it will be re-slicing to the same frame rate as the first object and then be concatenated. Otherwise, it will be directly concatenated.

        Parameters
        ---
        marker1
            the first marker object.
        marker2
            the second marker object.
        """
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
        """Add the visualisation of current object to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        location
            the displace location represented in a (3, ) :class:`numpy.array`.
        trace_fps
            when drawing the trajectory, interpolate the marker points to :code:`trace_fps` to draw a smooth and continues trajectory.
        trace_width
            line width of the trajectory.
        trace_op
            opacity of the trajectory.
        radius
            radius of the key points.
        color
            color of the points and trajectory.
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

    def show(self) -> pv.Plotter:
        """Illustrate the key points object.
        """
        scene = pv.Plotter()
        self.add_to_scene(scene)

        scene.camera_position = 'xy'
        scene.show(interactive_update=True)

        return scene


class MarkerSet(object):
    """A collection of :class:`Marker` s. At current stage, it's usually loaded from the Vicon motion capture data.

    Parameters
    ---
    filedir
        directory of the :code:`.csv` key points coordinates data exported from the motion capture Vicon system.

    Note
    ---
    `Class Attributes`

    self.fps
        the number of frames per second (fps).
    self.markers
        a :class:`Dictonary` of :class:`Marker` s, with the corresponding marker names as their keywords.

    Example
    ---
    The Vicon motion capture data shall be exported as a :code:`.csv` file. After initialising the :class:`MarkerSet` data, we can load it providing the :code:`.csv` file's directory: ::

        from mesh4d import kps

        vicon = kps.MarkerSet()
        vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
        vicon.interp_field()

    Usually we implement the interpolation after loading the data, as shown in the last line of code. Then we can access the coordinates, speed, and acceleration data of any marker at any specific time: ::

        print(vicon.get_frame_coord(10))
        print(vicon.get_time_coord(1.0012)

    We can also access the specific marker with the marker name: ::

        print(vicon.points.keys())
        print(vicon.points['Bra_Miss Sun:CLAV'].get_frame_coord(10))
        print(vicon.points['Bra_Miss Sun:CLAV'].get_time_coord(1.0012))

    We can also plot and save the motion track as a :code:`.gif` file for illustration: ::

        vicon.plot_track(step=3, end_frame=100)
    
    """
    def __init__(self):
        self.markers = {}

    def load_from_vicon(self, filedir: str, trans_cab: Union[None, field.Trans_Rigid] = None):
        """Load and parse data from :code:`.csv` file exported from the Vicon motion capture system.

        Parameters
        ---
        filedir
            the directory of the :code:`.csv` file.

            .. attention::
                Noted that the original unit of 3dMD raw data is millimetre (mm).
        
        trans_cab
            tbf
        """
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
        """tbf
        array layout (frame, marker, axis)
        """
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
        """After loading Vicon motion capture data, the :class:`MarkerSet` object only carries the key points' coordinates in discrete frames. To access the coordinates at any specific time, it's necessary to call :meth:`interp_field`.

        Parameters
        ---
        **kwargs
            arguments to be passed to :meth:`Marker.interp_field`.
        """
        for point in self.markers.values():
            point.interp_field(**kwargs)

    def get_frame_coord(self, frame_id: int, kps_class: Type[Kps] = Kps) -> Type[Kps]:
        """Get coordinates data according to frame id and packed as a :class:`Kps` object.
        
        Parameters
        ---
        frame_id
            index of the frame to get coordinates data.
        kps_class
            the class of key points object to be initialised and returned.

        Return
        ---
        :class:`Kps` or its derived class
            The extracted coordinates data packed as a :class:`Kps` (or its derived class) object.
        """
        kps = kps_class()

        for name, marker in self.markers.items():
            coord = marker.get_frame_coord(frame_id)
            kps.add_point(name, coord)

        return kps
    
    def get_time_coord(self, time: float) -> np.array:
        """Get coordinates data according to time stamp.

        Parameters
        ---
        time
            time stamp to get coordinates data.

        Return
        ---
        :class:`numpy.array`
            The structure of the returned array is :code:`array[marker_id][0-2 as x-z][time]`
        
        Warnings
        ---
        The interpolation must be properly done before accessing coordinates data according to time stamp, which means the :meth:`interp_field` must be called first.

        WARNING
        ---
        The returned value will be transferred to :class:`Kps` in future development.
        """
        kps = Kps()

        for name, marker in self.markers.items():
            coord = marker.get_time_coord(time)
            kps.add_point(name, coord)

        return kps
    
    def to_array(self) -> tuple:
        """tbf"""
        index = []
        array_ls = []

        for name, marker in self.markers.items():
            index.append(name)
            array_ls.append(marker.coord.T)

        # (marker, frame, axis) -> (frame, marker, axis)
        array = np.swapaxes(np.array(array_ls), 0, 1)

        return array, index
    
    def extract(self, marker_names: Iterable[str]) -> MarkerSet:
        """Return the assembled marker set with extracted markers. Noted that the original marker set won't be altered.
        
        Parameters
        ---
        marker_names
            a list of marker names to be extracted
        """
        markerset = MarkerSet()

        for marker_name in marker_names:
            marker_extract = copy.deepcopy(self.markers[marker_name])
            markerset.markers[marker_name] = marker_extract

        return markerset
    
    def split(self, marker_names: Iterable[str]) -> tuple:
        """Return the markerset splitted into two part. Noted that the original marker set won't be altered.
        
        Parameters
        ---
        marker_names
            a list of marker names to be extracted to the first part.

        Retrun
        ---
        MarkerSet, MarkerSet
            return a tuple of two :class:`MarkerSet`. The first one contains the markers from the :attr:`marker_names`. The second one contains the remaining markers.
        """
        other_marker_names = []

        for name in self.markers.keys():
            if name not in marker_names:
                other_marker_names.append(name)

        return self.extract(marker_names), self.extract(other_marker_names)
    
    @staticmethod
    def diff(markerset1: MarkerSet, markerset2: MarkerSet) -> dict:
        """Compute the difference of one marker set object with another.

        Warning
        ---
        The same as :meth:`Marker.diff`, the second marker set must contain the time period of the first marker set.

        Parameters
        ---
        marker1
            a marker set object.
        marker2
            another marker set object.

        Returns
        ---
        :class:`dict`
            A dictionary that contains the comparison result:

            - :code:`diff_dict`: a dictionary that stores the original comparison results between each pair of key points.
            - :code:`'dist_mean'`: the mean distances from the first marker set to the second marker set.
            - :code:`'dist_std'`: the standard deviation of distances.
            - :code:`'diff_str'`: a string in form of :code:`'dist_mean ± dist_std (mm)`.
        """
        # estimate the difference of each key point
        diff_dict = {}
        dist_ls = []

        for name in markerset1.markers.keys():
            diff = Marker.diff(markerset1.markers[name], markerset2.markers[name])
            diff_dict[name] = diff
            dist_ls.append(diff['dist'])

            if mesh4d.output_msg:
                print("estimated error of frame {}: {}".format(name, diff['diff_str']))

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

    def reslice(self, fps_new: int):
        """Return the marker set object re-slicing to another frame rate. Noted that the original object won't be altered.

        Attention
        ---
        The new marker set object haven't undergo :meth:`interp_field`.
        
        Parameters
        ---
        fps_new
            the new frame rate (frames per second)"""
        markerset = MarkerSet()

        for name in self.markers.keys():
            markerset.markers[name] = self.markers[name].reslice(fps_new)

        return markerset

    @staticmethod
    def concatenate(markerset1: Type[Marker], markerset2: Type[Marker]) -> Marker:
        """Concatenate two marker set object.

        Attention
        ---
        If the second marker object has more than 1 frame of data, it will be re-slicing to the same frame rate as the first object and then be concatenated. Otherwise, it will be directly concatenated.

        Parameters
        ---
        markerset1
            the first marker set object.
        markerset2
            the second marker set object.
        """
        markerset = MarkerSet()

        for name in markerset1.markers.keys():
            markerset.markers[name] = Marker.concatenate(
                markerset1.markers[name],
                markerset2.markers[name],
            )

        return markerset
    
    def add_to_scene(self, scene: pv.Plotter, location: np.array = np.array((0, 0, 0)), trace_fps: float = 100, trace_width: float = 5, trace_op: float = 0.5, radius: float = 1, color: Union[str, None] = None, **kwargs) -> pv.Plotter:
        """Add the visualisation of current object to a :class:`pyvista.Plotter` scene.
        
        Parameters
        ---
        scene
            :class:`pyvista.Plotter` scene to add the visualisation.
        location
            the displace location represented in a (3, ) :class:`numpy.array`.
        trace_fps
            when drawing the trajectory, interpolate the marker points to :code:`trace_fps` to draw a smooth and continues trajectory.
        trace_width
            line width of the trajectory.
        trace_op
            opacity of the trajectory.
        radius
            radius of the key points.
        color
            color of the points and trajectory.
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

    def show(self) -> pv.Plotter:
        """Illustrate the key points object.
        """
        scene = pv.Plotter()
        self.add_to_scene(scene)

        scene.camera_position = 'xy'
        scene.show(interactive_update=True)

        return scene