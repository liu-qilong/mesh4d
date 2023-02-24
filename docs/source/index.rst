.. Ultra Motion Capture documentation master file, created by
   sphinx-quickstart on Fri Nov 18 10:57:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to :code:`mesh4d`'s documentation!
================================================

.. attention::

   This project is under active development.

.. figure:: figures/overall_structure.png
   
   Overall structure of the core modules.

This package (`repository link <https://github.com/TOB-KNPOB/Ultra-Motion-Capture>`_) is developed for the data processing of the 3dMD 4D scanning system. Comparing with traditional motion capture system, such as Vicon:

- Vicon motion capture system can provided robust & accurate key points tracking based on physical marker points attached to human body. But it suffer from the lack of continuous surface deformation information.

- 3dMD 4D scanning system can record continuous surface deformation information. But it doesn't provide key point tracking feature and it's challenging to track the key points via Computer Vision approach, even with the state of the art methods in academia [#]_.

To facilitate human factor research, we deem it an important task to construct a hybrid system that can integrate the advantages and potentials of both systems. The motivation and the core value of this project can be described as: *adding continuous spatial dynamic information to Vicon* or *adding discrete key points information to 3dMD*, leading to advancing platform for human factor research in the domain of dynamic human activity.

.. [#] Min, Z., Liu, J., Liu, L., & Meng, M. Q.-H. (2021). Generalized coherent point drift with multi-variate gaussian distribution and watson distribution. IEEE Robotics and Automation Letters, 6(4), 6749–6756. https://doi.org/10.1109/lra.2021.3093011

.. tip::
   Before jump into the :doc:`api`, please read the :doc:`dev notes` and :doc:`design principles` to get an overall understanding of the technical settings and program structure.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 100

   dev notes
   design principles
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
