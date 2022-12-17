"""The :mod:`UltraMotionCapture.kps` module stands for *key points*. In :mod:`UltraMotionCapture` package, key points are essential elements to facilitate the processing of 4D images.

There are two different perspectives to arrange key points data: *time-wise* and *point-wise*. Reflecting these two ways of arrangement:

- The :class:`Kps` and `Kps_Deform` contain all key points' data at a specific moment;
- While the :class:`Marker` contains a specific key point's data within a time period. To aggregate all key points' data, :class:`MarkerSet` is provided.
"""

from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import interpolate

from UltraMotionCapture import field
from UltraMotionCapture import obj3d
from UltraMotionCapture import utils

class Kps(object):
    """A collection of the key points that can be attached to a 3D object, i.e. a frame of the 4D object.

    Note
    ---
    `Class Attributes`

    self.kps_source_points
        :math:`N` key points in 3D space stored in a (N, 3) :class:`numpy.array`.

    Example
    ---
    After initialisation, the :class:`Kps` object is empty. There are two ways to load key points into it:

    Manually selecting key points with :meth:`select_kps_points`. ::

        from UltraMotionCapture import kps

        points = kps.Kps()
        points.select_kps_points()  # this will trigger a point selection window

    Load key points from Vicon motion capture data stored in a :class:`MarkerSet` object with :meth:`load_from_markerset_frame` or :meth:`load_from_markerset_time`. ::

        from UltraMotionCapture import kps
        
        vicon = kps.MarkerSet()
        vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
        vicon.interp_field()

        points = kps.Kps()
        points.load_from_markerset_frame(vicon)
    """
    def __init__(self):
        self.kps_source_points = None  # key points are stored in Nx3 numpy array

    def select_kps_points(self, source: o3d.geometry.PointCloud):
        """ Interactive manual points selection.

        Parameters
        ---
        source
            an :class:`open3d.geometry.PointCloud` object for points selection.

        Warnings
        ---
        At current stage, the interactive manual points selection is realised with :mod:`open3d` package. It will be transferred to :mod:`pyvista` package in future development.
        """
        print("please select key points")

        def pick_points(pcd):
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            return vis.get_picked_points()

        pick = pick_points(source)
        points = obj3d.pcd2np(source)
        self.set_kps_source_points(
            points[pick, :]
        )
        print("selected key points:\n{}".format(self.kps_source_points))

    def load_from_markerset_frame(self, markerset: MarkerSet, frame_id: int = 0):
        """Load key points to the :class:`Kps` object providing the :class:`MarkerSet` and frame index.

        Parameters
        ---
        markerset
            a :class:`MarkerSet` object carrying Vicon motion capture data, which contains various frames.
        frame_id
            the frame index of the Vicon motion capture data to be loaded.
        """
        points = markerset.get_frame_coord(frame_id)
        points_cal = points
        self.set_kps_source_points(points_cal)

    def load_from_markerset_time(self, markerset: MarkerSet, time: float = 0.0):
        """Load key points to the :class:`Kps` object providing the :class:`MarkerSet` and time stamp.

        Parameters
        ---
        markerset
            a :class:`MarkerSet` object carrying Vicon motion capture data, which contains various frames.

            Warnings
            ---
            Before passing into :meth:`load_from_markerset_time`, call :meth:`MarkerSet.interp_field` first so that coordinates data at any specific time is accessible.
        time
            the time stamp of Vicon motion data to be loaded.
        """
        points = markerset.get_time_coord(time)
        points_cal = points
        self.set_kps_source_points(points_cal)

    def set_kps_source_points(self, points: np.array):
        """Other than manually selecting points or loading points from Vicon motion capture data, the :attr:`kps_source_points` can also be directly overridden with a (N, 3) :class:`numpy.array`, representing :math:`N` key points in 3D space.

        Parameters
        ---
        points
            (N, 3) :class:`numpy.array`.
        """
        self.kps_source_points = points

    def get_kps_source_points(self) -> np.array:
        """ Get the key points coordinates.
        """
        if self.kps_source_points is None:
            print("source key points haven't been set")
        else:
            return self.kps_source_points


class Kps_Deform(Kps):
    """Adding deformation feature to the :class:`Kps` class.

    Note
    ---
    `Class Attributes`

    self.trans
        An :class:`UltraMotionCapture.field.Trans_Nonrigid` object that stores the deformation information.
    self.kps_deform_points
        (N, 3) :class:`numpy.array`.
    """
    def __init__(self):
        Kps.__init__(self)
        self.trans = None
        self.kps_deform_points = None

    def set_trans(self, trans: field.Trans_Nonrigid):
        """ Setting the transformation of the deformable key points object.

        Parameters
        ---
        trans
            an :meth:`UltraMotionCapture.field.Trans_Nonrigid` object that represents the transformation.
        """
        self.trans = trans
        self.kps_deform_points = self.trans.shift_points(self.kps_source_points)

    def get_kps_deform_points(self) -> np.array:
        """ Get the key points coordinates after transformation.
        """
        return self.kps_deform_points


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

    Note
    ---
    `Class Attributes`

    self.name
        The name of the marker.
    self.start_time
        The start time of the coordinates data.
    self.fps
        The number of frames per second (fps).
    self.coord
        :math:`3 \\times N` :class:`numpy.array` storing the coordinates data, with :math:`x, y, z` as rows and frame ids as the columns.
    self.speed
        :math:`3 \\times N` :class:`numpy.array` storing the speed data, with :math:`x, y, z` as rows and frame ids as the columns.
    self.accel
        :math:`3 \\times N` :class:`numpy.array` storing the acceleration data, with :math:`x, y, z` as rows and frame ids as the columns.
    self.frame_num
        The number of total frames.
    self.x_field
        An :class:`scipy.interpolate.interp1d` object that storing the interpolated function of the :math:`x` coordinates of all frames. Used for estimated the :math:`x` coordinate of any intermediate time between frames.
    self.y_field
        An :class:`scipy.interpolate.interp1d` object that storing the interpolated function of the :math:`y` coordinates of all frames. Used for estimated the :math:`y` coordinate of any intermediate time between frames.
    self.z_field
        An :class:`scipy.interpolate.interp1d` object that storing the interpolated function of the :math:`z` coordinates of all frames. Used for estimated the :math:`z` coordinate of any intermediate time between frames.

    Tip
    ---
    When loading Vicon motion capture data, the whole process is arranged by a :class:`MakerSet` object, which creates :class:`Marker` objects for each key point and loads data into it accordingly. Therefore, usually the end user doesn't need to touch the :class:`Marker` class.
    """
    def __init__(self, name: str, start_time: float = 0.0, fps: int = 100):
        self.name = name
        self.start_time = start_time
        self.fps = fps

        # x, y,z data are stored in 3xN numpy array
        self.coord = None  # coordinates
        self.speed = None  # speed
        self.accel = None  # acceleration

        self.frame_num = None

        self.x_field = None
        self.y_field = None
        self.z_field = None

    def fill_data(self, data_input):
        """Filling coordinates, speed, and acceleration data, one by one, into the :class:`Marker` object.

        Parameters
        ---
        data_input
            :math:`3 \\times N` :class:`numpy.array`.

        Attention
        ---
        Called by the :class:`MarkerSet` object when parsing the Vicon motion capture data (:meth:`MarkerSet.load_from_vicon`). Usually the end user don't need to call this method manually.
        """
        if self.coord is None:
            self.coord = data_input
            self.frame_num = data_input.shape[1]
        elif self.speed is None:
            self.speed = data_input
        elif self.accel is None:
            self.accel = data_input

    def interp_field(self):
        """Interpolating the :math:`x, y, z` coordinates data to estimate its continues change. After that, the coordinates at the intermediate time between frames is accessible.

        Warnings
        ---
        Before interpolation, the coordinates data, i.e. :attr:`self.coord`, must be properly loaded.
        """
        if self.coord is None:
            print("coordinates information not found")
            return

        frame_range = range(len(self.coord[0]))

        self.x_field = interpolate.interp1d(frame_range, self.coord[0], 'slinear')
        self.y_field = interpolate.interp1d(frame_range, self.coord[1], 'slinear')
        self.z_field = interpolate.interp1d(frame_range, self.coord[2], 'slinear')

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
            print("coordinates field need to be interped first")
            return

        frame_id = (time - self.start_time) * self.fps
        coord_interp = np.array(
            [self.x_field(frame_id),
             self.y_field(frame_id),
             self.z_field(frame_id)]
        )
        return coord_interp

    def plot_track(
            self,
            line_start_frame: int = 0,
            line_end_frame: Union[int, None] = None,
            dot_start_frame: int = 0,
            dot_end_frame: Union[int, None] = None,
            line_alpha: float = 0.5,
            line_width: float = 1.0,
            dot_s: float = 10,
            dot_alpha: float = 0.5,
            dpi: int = 300,
            is_show: bool = True,
            is_save: bool = False,
    ):
        """Plotting the marker motion track.

        Parameters
        ---
        line_start_frame
            start frame of line plotting.
        line_end_frame
            end frame of line plotting, default as :code:`None`, which means plot till the end.
        dot_start_frame
            start frame of dot plotting.
        dot_end_frame
            end frame of dot plotting, default as :code:`None`, which means plot till the end.
        is_show
            weather show the generated graph or not.
        is_save
            weather save the generated graph or not.
        Others
            parameters passed to :meth:`plot_add_line` and :meth:`plot_add_dot` to controlling the appearance.
        """
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(projection='3d')

        self.plot_add_line(
            ax,
            line_start_frame,
            line_end_frame,
            line_alpha,
            line_width)

        self.plot_add_dot(
            ax,
            dot_start_frame,
            dot_end_frame,
            dot_s,
            dot_alpha,
        )

        font = {'family': 'Times New Roman'}
        plt.title('Marker Point Trajectory\n' + self.name, fontdict=font)
        ax.set_xlabel('X-axis', fontdict=font)
        ax.set_ylabel('Y-axis', fontdict=font)
        ax.set_zlabel('Z-axis', fontdict=font)
        ax.tick_params(labelsize=7)

        if is_show:
            plt.show()

        if is_save:
            plt.savefig('output/save-line'
                        + str(line_start_frame) + ' ' + str(line_end_frame)
                        + '-dot' + str(dot_start_frame) + ' ' + str(dot_end_frame))

    def plot_add_line(
            self,
            ax: plt.subplot,
            line_start_frame: int = 0,
            line_end_frame: Union[int, None] = None,
            line_alpha: float = 0.5,
            line_width: float = 1,
            **kwargs
    ):
        """Adding motion track lines to the :class:`matplotlib.pyplot.subplot` object created in :meth:`plot_track`.

        Tip
        ---
        About the appearance controlling parameters, please refer to `Pyplot tutorial - matplotlib <https://matplotlib.org/stable/tutorials/introductory/pyplot.html#pyplot-tutorial>`_.

        Additional appearance controlling parameters can be passed into :code:`**kwargs`, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_.
        """
        ax.plot3D(
            self.coord[0, line_start_frame:line_end_frame],
            self.coord[1, line_start_frame:line_end_frame],
            self.coord[2, line_start_frame:line_end_frame],
            'gray', alpha=line_alpha, linewidth=line_width, **kwargs
        )

    def plot_add_dot(
            self,
            ax: plt.subplot,
            dot_start_frame: int = 0,
            dot_end_frame: Union[int, None] = None,
            dot_s: int = 10,
            dot_alpha: float = 0.5,
            **kwargs
    ):
        """Adding motion dots in different frames to the :class:`matplotlib.pyplot.subplot` object created in :meth:`plot_track`.

        Tip
        ---
        About the appearance controlling parameters, please refer to `Pyplot tutorial - matplotlib <https://matplotlib.org/stable/tutorials/introductory/pyplot.html#pyplot-tutorial>`_.

        Additional appearance controlling parameters can be passed into :code:`**kwargs`, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_.
        """
        ax.scatter3D(
            self.coord[0, dot_start_frame:dot_end_frame],
            self.coord[1, dot_start_frame:dot_end_frame],
            self.coord[2, dot_start_frame:dot_end_frame],
            s=dot_s, alpha=dot_alpha, **kwargs
        )


class MarkerSet(object):
    """A collection of :class:`Marker` s. At current stage, it's usually loaded from the Vicon motion capture data.

    Note
    ---
    `Class Attributes`

    self.fps
        The number of frames per second (fps).
    self.points
        A :class:`Dictonary` of :class:`Marker` s, with the corresponding marker names as their keywords.

    Example
    ---
    The Vicon motion capture data shall be exported as a :code:`.csv` file. After initialising the :class:`MarkerSet` data, we can load it providing the :code:`.csv` file's directory: ::

        from UltraMotionCapture import kps

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
        pass

    def load_from_vicon(self, filedir: str):
        """Load and parse data from :code:`.csv` file exported from the Vicon motion capture system.

        Parameters
        ---
        filedir
            the directory of the :code:`.csv` file.
        """
        def parse(df, df_head):
            self.fps = df_head.values.tolist()[0][0]  # parse the fps
            self.points = {}
            col_names = df.columns.values.tolist()

            for col_id in range(len(col_names)):
                col_name = col_names[col_id]
                point_name = col_name.split('.')[0]

                # skip columns that contain NaN
                # (checking start from row 4, because for speed and acceleration the first few rows are empty)
                # or that follows the 'X' columns
                if df.loc[4:, col_name].isnull().values.any():
                    continue

                if 'Unnamed' in col_name:
                    continue

                # the first occurrence of a point
                if point_name not in self.points.keys():
                    self.points[point_name] = Marker(point_name, fps=self.fps)

                # fill the following 3 columns' X, Y, Z values into the point's object
                data_input = df.loc[2:, col_name:col_names[col_id+2]].to_numpy(dtype=float).transpose()
                self.points[point_name].fill_data(data_input)

        df = pd.read_csv(filedir, skiprows=2)  # skip the first two rows
        df_head = pd.read_csv(filedir, nrows=1)  # only read the first two rows
        parse(df, df_head)

        print("loaded 1 vicon file: {}".format(filedir))

    def interp_field(self):
        """After loading Vicon motion capture data, the :class:`MarkerSet` object only carries the key points' coordinates in discrete frames. To access the coordinates at any specific time, it's necessary to call :meth:`interp_field`.
        """
        for point in self.points.values():
            point.interp_field()

    def get_frame_coord(self, frame_id: int) -> np.array:
        """Get coordinates data according to frame id.
        
        Parameters
        ---
        frame_id
            index of the frame to get coordinates data.

        Return
        ---
        :class:`numpy.array`
            The structure of the returned array is :code:`array[marker_id][0-2 as x-z][frame_id]`

        WARNING
        ---
        The returned value will be transferred to :class:`Kps` in future development.
        """
        points = []
        for point in self.points.values():
            points.append(
                point.get_frame_coord(frame_id)
            )
        return np.array(points)

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
        points = []
        for point in self.points.values():
            points.append(
                point.get_time_coord(time)
            )
        return np.array(points)

    def plot_track(
            self,
            start_frame: int = 0,
            end_frame: Union[int, None] = None,
            step: int = 1,
            remove: bool = True,
            *args,
            **kwargs
    ):
        """Plotting the marker motion track.

        Parameters
        ---
        start_frame
            start frame of plotting.
        end_frame
            end frame of plotting, default as :code:`None`, which means plot till the end.
        step
            Plot 1 frame for every :code:`step` frame. The purpose is reducing graph generating time.
        remove
            after generating the :code:`.gif` file, remove the frames' images or not.

        Tip
        ---
        Additional appearance controlling parameters can be passed into :code:`**kwargs`, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_ and `Pyplot tutorial - matplotlib <https://matplotlib.org/stable/tutorials/introductory/pyplot.html#pyplot-tutorial>`_
        """
        if end_frame is None:
            first_point = list(self.points.values())[0]
            end_frame = first_point.frame_num

        for frame_id in range(start_frame, end_frame, step):
            self.plot_frame(frame_id, is_show=False, is_save=True, *args, **kwargs)

        utils.images_to_gif('output/', remove=remove)

    def plot_frame(
            self,
            frame_id: int,
            dpi: int = 300,
            is_add_line: bool = True,
            is_show: bool = True,
            is_save: bool = False,
            **kwargs
    ):
        """
        Plot a specific frame.

        Parameters
        ---
        frame_id
            index of the frame to be plotted.
        dpi
            the dots per inch (dpi) of the generated graph, controlling the graph quality.
        is_add_line
            weather add track links or not
        is_show
            weather show the generated graph or not.
        is_save
            weather save the generated graph or not.
        """
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(projection='3d')

        for point in self.points.values():
            point.plot_add_dot(ax, frame_id, frame_id+1, **kwargs)
            if is_add_line:
                point.plot_add_line(ax)

        font = {'family': 'Times New Roman'}
        plt.title('Marker Point Trajectory', fontdict=font)
        ax.set_xlabel('X-axis', fontdict=font)
        ax.set_ylabel('Y-axis', fontdict=font)
        ax.set_zlabel('Z-axis', fontdict=font)
        ax.tick_params(labelsize=7)

        plt.savefig('output/gif-{:0>4d}'.format(frame_id))
        if is_show:
            plt.show()

        if is_save:
            filedir = 'output/gif-' + str(frame_id)
            plt.savefig(filedir)
            print('saved ' + filedir)