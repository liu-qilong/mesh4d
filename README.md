# `mesh4d`

## Introduction

This package is developed for the data processing of the 3dMD 4D scanning system. Compared with traditional motion capture systems, such as Vicon:

- Vicon motion capture system can provide robust & accurate key points tracking based on physical marker points attached to the human body. But it suffers from the lack of continuous surface deformation information.

- 3dMD 4D scanning system can record continuous surface deformation information. But it doesn't provide key point tracking functionality and it's challenging to track the key points via the Computer Vision approach, even with the state-of-the-art methods in academia[^Min_Z_2021].

[^Min_Z_2021]: Min, Z., Liu, J., Liu, L., & Meng, M. Q.-H. (2021). Generalized coherent point drift with multi-variate gaussian distribution and Watson distribution. IEEE Robotics and Automation Letters, 6(4), 6749–6756. https://doi.org/10.1109/lra.2021.3093011

To facilitate dynamic shape analysis research, we deem it an important task to construct a hybrid system that can integrate the advantages and potentials of both systems. The motivation and the core value of this project can be described as *adding continuous spatial dynamic information to Vicon* or *adding discrete key points information to 3dMD*, leading to an advancing platform for human factor research in the domain of dynamic human activity.

## Setup

```
git clone https://github.com/liu-qilong/mesh4d.git
cd mesh4d
conda create -n mesh4d python=3.10
conda activate mesh4d
pip install -r requirements.txt
python -m pip install --editable .
```

## Overall structure

![overall structure](https://github.com/liu-qilong/mesh4d/blob/main/gallery/overall-structure.png?raw=true)

_P.S. The solid arrow pointing from `class A` to `class B` indicates that `class B` is derived from `class A`, while the dotted arrow indicates that a `class A` object contains a `class B` object as an attribute._