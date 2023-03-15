from __future__ import annotations
from typing import Type, Union, Iterable

import os
import sys
import pickle
import imageio
import numpy as np
import pyvista as pv

import mesh4d.config.param
from mesh4d import kps

def images_to_gif(folder: Union[str, None] = None, remove: bool = False):
    """Convert images in a folder into a gif.
    
    Parameters
    ---
    folder
        the directory of the folder storing the images.
    remove
        after generating the :code:`.gif` image, whether remove the original static images or not.

    Example
    ---
    ::

        import mesh4d as umc
        umc.utils.images_to_gif(folder="output/", remove=True)
    """
    files = os.listdir(folder)
    files.sort()
    images = []

    for file in files:
        if ('png' in file or 'jpg' in file) and ('gif-' in file):
            images.append(imageio.imread(folder + file))
            if remove:
                os.remove(folder + file)

    if len(images) == 0:
        print("No images in folder")
    else:
        imageio.mimsave(folder + 'output.gif', images)


def progress_bar(percent: float, bar_len: int = 20, front_str: str = '', back_str: str = ''):
    """Print & refresh the progress bar in terminal.

    Parameters
    ---
    percent
        percentage from 0 to 1.
    bar_len
        length of the progress bar
    front_str
        string proceeding the progress bar
    back_str
        string following the progress bar

    Warning
    ---
    Avoid including "new line" in :attr:`front_str` or :code:`back_str`.
    """
    sys.stdout.write("\r")
    sys.stdout.write("{}[{:<{}}] {:.1%}{}".format(front_str, "=" * int(bar_len * percent), bar_len, percent, back_str))
    sys.stdout.flush()
    # avoiding '%' appears when progress completed
    if percent == 1:
        print()


def save_pkl_object(obj, export_folder: str = 'output/', export_name: str = 'pickle'):
    """Save an object to an :code:`.pkl` file.
    
    .. seealso::
        `Saving an Object (Data persistence) - StackOverflow <https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence>`_
    
    Parameters
    ---
    obj
        the object for storing locally.
    export_folder
        The folder to save the labelled landmarks to.
    export_name
        The filename to save the labelled landmarks.

    Example
    ---
    ::

         save_pkl_object(vicon, 'data/', 'vicon')
    """
    filepath = os.path.join(export_folder, "{}.pkl".format(export_name))
    
    with open(filepath, 'wb') as outp:  # overwrites any existing file
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filepath: str):
    """Load an object stored in :code:`.pkl` file.

    .. seealso::
        `Saving an Object (Data persistence) - StackOverflow <https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence>`_

    Parameters
    ---
    filepath
        the path of the :code:`.pkl` file.


    Example
    ---
    ::

         vicon = load_pkl_object('data/vkps/vicon.pkl')
    """
    with open(filepath, 'rb') as inp:
        return pickle.load(inp)