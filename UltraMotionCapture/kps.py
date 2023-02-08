"""The :mod:`UltraMotionCapture.kps` module stands for *key points*. In :mod:`UltraMotionCapture` package, key points are essential elements to facilitate the processing of 4D images.

There are two different perspectives to arrange key points data: *time-wise* and *point-wise*. Reflecting these two ways of arrangement:

- The :class:`Kps` and `Kps_Deform` contain all key points' data at a specific moment;
- While the :class:`Marker` contains a specific key point's data within a time period. To aggregate all key points' data, :class:`MarkerSet` is provided.
"""

from __future__ import annotations
from typing import Type, Union, Iterable

import os
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import interpolate

import UltraMotionCapture
import UltraMotionCapture.config.param
from UltraMotionCapture import field, utils

class Kps(object):
    """A collection of the key points that can be attached to a 3D object, i.e. a frame of the 4D object.

    Note
    ---
    `Class Attributes`

    self.points
        :math:`N` key points in 3D space stored in a dictionary.
    self.scale_rate
        the scaling rate of the Vicon key points.

    Example
    ---
    After initialisation, the :class:`Kps` object is empty. Load key points from Vicon motion capture data stored in a :class:`MarkerSet` object with :meth:`load_from_markerset_frame` or :meth:`load_from_markerset_time`. ::

        from UltraMotionCapture import kps
        
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
            - :code:`'dist'`: the distances from the predicted key points to the ground truth key points stored in a (N, ) :class:`numpy.array`.
            - :code:`'dist_mean'`: the mean distances from the predicted key points to the ground truth key points.
            - :code:`'dist_std'`: the standard deviation of distances from the predicted key points to the ground truth key points.
            - :code:`'diff_str'`: a string in form of :code:`'dist_mean ± dist_std (mm)`.
        """
        names = kps1.points.keys()
        points1 = kps1.get_points_coord(names) / kps1.scale_rate
        points2 = kps2.get_points_coord(names) / kps2.scale_rate

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


class Kps_Deform(Kps):
    """Adding deformation feature to the :class:`Kps` class.

    Note
    ---
    `Class Attributes`

    self.trans
        an :class:`UltraMotionCapture.field.Trans_Nonrigid` object that stores the deformation information.
    self.deform_kps
        the deformed key points object.
    """
    def __init__(self):
        Kps.__init__(self)
        self.trans = None

    def set_trans(self, trans: field.Trans_Nonrigid, kps_class: Type[Kps_Deform]) -> Type[Kps_Deform]:
        """ Setting the transformation of the deformable key points object.

        Parameters
        ---
        trans
            an :meth:`UltraMotionCapture.field.Trans_Nonrigid` object that represents the transformation.
        kps_class
            the key point object class to loaded the deformed key points.

        Returns
        ---
        the deformed key points object.
        """
        self.trans = trans
        self.deform_kps = kps_class()
        
        for name, coord in self.points.items():
            coord_deform = self.trans.shift_points((coord, ))
            self.deform_kps.add_point(name, coord_deform)
        
        return self.deform_kps

    def get_deform_points_coord(self) -> np.array:
        """ Get the key points coordinates after transformation.
        """
        return self.deform_kps.get_points_coord()


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
    scale_rate
        the scaling rate of the Vicon key points.

        Attention
        ---
        Noted that the original unit of Vicon raw data is millimetre (mm). The default :attr:`scale_rate` transforms it to metre (m).

        Warning
        ---
        This value must be the same as :class:`UltraMotionCapture.obj3d.Obj3d`'s :attr:`scale_rate`.

    Note
    ---
    `Class Attributes`

    cls.cab_s
        rotation matrix :math:`\\boldsymbol R \in \mathbb{R}^{3 \\times 3}` stored in (3, 3) :class:`numpy.array` used for converting Vicon coordinates to 3dMD coordinates.
    cls.cab_r
        scaling rate :math:`s \in \mathbb{R}` stored in a :class:`float` variable used for converting Vicon coordinates to 3dMD coordinates.
    cls.cab_t
        translation vector :math:`\\boldsymbol t \in \mathbb{R}^{3}` stored in (3, ) :class:`numpy.array` used for converting Vicon coordinates to 3dMD coordinates.

    self.name
        The name of the marker.
    self.start_time
        The start time of the coordinates data.
    self.fps
        The number of frames per second (fps).
    self.scale_rate
        the scaling rate of the Vicon key points.
    self.coord
        (3, N) :class:`numpy.array` storing the coordinates data, with :math:`x, y, z` as rows and frame ids as the columns.
    self.speed
        (3, N) :class:`numpy.array` storing the speed data, with :math:`x, y, z` as rows and frame ids as the columns.
    self.accel
        (3, N) :class:`numpy.array` storing the acceleration data, with :math:`x, y, z` as rows and frame ids as the columns.
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

    Tip
    ---
    In other parts of the package, points coordinates storing in :class:`numpy.array` are usually in the shape of (N, 3), while here we adopt (3, N), for the convenience of data interpolation :meth:`interp_field`.
    """
    cab_s = None
    cab_r = None
    cab_t = None

    def __init__(self, name: str, start_time: float = 0.0, fps: int = 100, scale_rate: float = 1e-3):
        if self.cab_s is None:
            self.load_cab_rst()
            if UltraMotionCapture.output_msg:
                print('calibration parameters loaded')

        self.name = name
        self.start_time = start_time
        self.fps = fps
        self.scale_rate = scale_rate

        # x, y, z data are stored in 3xN numpy array
        self.coord = None  # coordinates
        self.speed = None  # speed
        self.accel = None  # acceleration

        self.frame_num = None

        self.x_field = None
        self.y_field = None
        self.z_field = None

    @classmethod
    def load_cab_rst(cls):
        """Load the calibration parameters from Vicon to 3dMD coordination system.
        """
        mod_path = os.path.dirname(UltraMotionCapture.__file__)
        cls.cab_r = np.load(os.path.join(mod_path, 'config/calibrate/r.npy'))
        cls.cab_s = np.load(os.path.join(mod_path, 'config/calibrate/s.npy'))
        cls.cab_t = np.load(os.path.join(mod_path, 'config/calibrate/t.npy'))

    def fill_data(self, data_input: np.array):
        """Filling coordinates, speed, and acceleration data after converting to 3dMD coordinates, one by one, into the :class:`Marker` object.

        Parameters
        ---
        data_input
            (3, N) :class:`numpy.array`.

        Attention
        ---
        Called by the :class:`MarkerSet` object when parsing the Vicon motion capture data (:meth:`MarkerSet.load_from_vicon`). Usually the end user don't need to call this method manually.
        """
        data_input = self.scale_rate * (
            self.cab_s * np.matmul(self.cab_r, data_input) + np.expand_dims(self.cab_t, axis=1)
            )

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
            if UltraMotionCapture.output_msg:
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
            if UltraMotionCapture.output_msg:
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

    Parameters
    ---
    filedir
        directory of the :code:`.csv` key points coordinates data exported from the motion capture Vicon system.
    scale_rate
        the scaling rate of the Vicon key points.

        Attention
        ---
        Noted that the original unit of Vicon raw data is millimetre (mm). The default :attr:`scale_rate` transforms it to metre (m).

        Warning
        ---
        This value must be the same as :class:`UltraMotionCapture.obj3d.Obj3d`'s :attr:`scale_rate`.


    Note
    ---
    `Class Attributes`

    self.fps
        the number of frames per second (fps).
    self.scale_rate
        the scaling rate of the Vicon key points.
    self.markers
        a :class:`Dictonary` of :class:`Marker` s, with the corresponding marker names as their keywords.

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
    def load_from_vicon(self, filedir: str, scale_rate: float = 1e-3):
        """Load and parse data from :code:`.csv` file exported from the Vicon motion capture system.

        Parameters
        ---
        filedir
            the directory of the :code:`.csv` file.
        scale_rate
            the scaling rate of the 3D object.

            .. attention::
                Noted that the original unit of 3dMD raw data is millimetre (mm). The default :attr:`scale_rate` transforms it to metre (m).

            .. seealso::
                Reason for providing :code:`scale_rate` parameter is explained in :class:`Obj3d_Deform`.
        """
        self.scale_rate = scale_rate

        def parse(df, df_head):
            self.fps = df_head.values.tolist()[0][0]  # parse the fps
            self.markers = {}
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
                if point_name not in self.markers.keys():
                    self.markers[point_name] = Marker(
                        name=point_name, 
                        fps=self.fps, 
                        scale_rate=self.scale_rate)

                # fill the following 3 columns' X, Y, Z values into the point's object
                data_input = df.loc[2:, col_name:col_names[col_id+2]].to_numpy(dtype=float).transpose()
                self.markers[point_name].fill_data(data_input)

        df = pd.read_csv(filedir, skiprows=2)  # skip the first two rows
        df_head = pd.read_csv(filedir, nrows=1)  # only read the first two rows
        parse(df, df_head)
        
        if UltraMotionCapture.output_msg:
            print("loaded 1 vicon file: {}".format(filedir))

    def interp_field(self):
        """After loading Vicon motion capture data, the :class:`MarkerSet` object only carries the key points' coordinates in discrete frames. To access the coordinates at any specific time, it's necessary to call :meth:`interp_field`.
        """
        for point in self.markers.values():
            point.interp_field()

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
        kps.scale_rate = self.scale_rate

        for name, marker in self.markers.items():
            coord = marker.get_frame_coord(frame_id)
            kps.add_point(name, coord)

        return kps

    def get_time_coord(self, time: float, kps_class: Type[Kps] = Kps) -> np.array:
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
        kps = kps_class()
        kps.scale_rate = self.scale_rate

        for name, marker in self.markers.items():
            coord = marker.get_time_coord(time)
            kps.add_point(name, coord)

        return kps

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
            first_point = list(self.markers.values())[0]
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

        for point in self.markers.values():
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

            if UltraMotionCapture.output_msg:
                print('saved ' + filedir)