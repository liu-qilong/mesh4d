# Design Principles

As discussed in {ref}`Project Code`, there are 5 `.py` files directly under the `UltraMotionCapture/` folder. These are 4 **core modules** ({mod}`UltraMotionCapture.obj4d`, {mod}`UltraMotionCapture.obj3d`, {mod}`UltraMotionCapture.kps`, {mod}`UltraMotionCapture.field`) and 1 **auxiliary modules** ({mod}`UltraMotionCapture.utils`) of this project.

They serve as the skeleton to support any downstream analysis task. Therefore, it's necessary to understand their inner relationship so that you would know how to utilise it, tweak it, and advance it to fit your customised need.

## What makes an analysable 4D scene?

As suggested by the name, 4D scanning is an imaging system that can record 3D + T data, i.e. 3 dimensions of space and 1 dimension of time. To be more specific, the 3dMD 4D scanning system records 3D image series in very high time- and space- resolution.

Therefore, it can provide very rich information on the dynamic movement and deformation of the human body during active activities. However, there is a crucial lack of inner-relationship information between different frames of 3D images:

```{attention}
For example, with the 5th and 6th frames of 3D images, we know that the former one transforms into the next one. However, for any specific point in the 5th frame, we don't know which point in the 6th frame it transfers to.

*Such lack of information blocks the way of any sophisticated and thematic analysis of the 4D data*, such as tracing the movement of the nipple points and tracking the variation of the upper arm area during some kind of sports activity.
```

Actually, the whole {mod}`UltraMotionCapture` project is motivated and centred around this bottleneck problem. We aims at revealing the so-called inner-relationship information between different frames. **An analysable 4D scene must consists such information**. At the most meticulous level, the inner-relationship information can be represented as a *displacement field*. That is, specifically, in which direction and to what distance a point in a 3D frame is moving.

## Construction of the analysable 4D scene

The development of the core modules and 1 auxiliary modules ({mod}`UltraMotionCapture.utils`) is centred around constructing an {ref}`analysable 4D scene <What makes an analysable 4D scene?>`. It follows such a pattern:

- The 4D object defined in {mod}`UltraMotionCapture.obj4d` contains of a series of 3D objects.
- The 3D object defined in {mod}`UltraMotionCapture.obj3d` contains the loaded mesh, point cloud, key point coordinates (provided by Vicon), and the estimated displacement field.
- Tools for handling key point coordinates and estimating displacement field are provided in {mod}`UltraMotionCapture.kps` and {mod}`UltraMotionCapture.field`, respectively.

At this stage, the structure is still quite clear, right? It's even more simple in actual usage:

```{code}
import UltraMotionCapture as umc

# load Vicon motion capture data
vicon = umc.kps.MarkerSet()
vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
vicon.interp_field()

# load 3dMD scanning data
o3_ls = umc.obj3d.load_obj_series(
 folder='data/6kmh_softbra_8markers_1/',
 start=0,
 end=1,
 sample_num=1000,
 obj_type=umc.obj3d.Obj3d_Deform
)

# initialise 4D scene object
o4 = umc.obj4d.Obj4d_Deform(
 markerset=vicon,
 fps=120,
 enable_rigid=True,
 enable_nonrigid=True,
)

# load 3D data to 3d scene
# the displacement field estimation will be implemented automatically
o4.add_obj(*o3_ls)
```

In this code example, a 4D scene object for further analysis is prepared in 4 steps.

```{tip}
For specific meaning and usage of these functions and their inputs, please refer to the {doc}`api`.
```

## Object-orientated development

Now we have an analysable 4D object, but inside it, there aren't many functions for analysis. What's the deal?

It's related to our *design idea*: object-orientated development. If you're not familiar with this idea, let me explain some bit of it to you. Object-orientated development, aka object-orientated programming, is a kind of programming paradigm that abstracts the programming problem into objects. In the real world, problems are always related to *objects*. For example:

```{admonition} Example
The traffic light scheduling problem is consist of 3 types of objects, aka 3 classes: `car`, `human`, and `road`. Each of the objects contains a series of actions (represented as a function in programming), variables (such as speed and size, represented as attributes in programming), and interactions with other objects.
```

In our project, the object relationship is:
- 4D object contains a series of 3D objects.
- 3D object contains mesh object, point cloud object, key points object, and displacement field object.

It makes it much easier to manage complex elements with numerous parameters that need to be taken care of since it groups elements into independent objects in a logical way. However, that's only half of the advantage of object-orientated programming. **Inheritance** is even more powerful. Let's go back to the traffic light scheduling example:

```{admonition} Example
There are various kinds of `car`, such as `bus`, `truck`, and `van`. They are all `car`s so they should share some common attributes and functions with `car` objects, while they all have some special attributes and functions.

In object-orientated programming, a `bus` class can be defined as derived from the `car` object, inheriting attributes and functions of `car` class, and adding/revising its supplement attributes and functions. And so do the `truck` and `van` classes.
```

With inheritance, the development of classes can lay out in an incremental fashion. If you jump into {mod}`UltraMotionCapture.obj3d` and {mod}`UltraMotionCapture.obj4d`, you will find that they all contain 3 classes, 1 without suffix and 2 with suffixe `_Kps` or `_Deform`:

- The classes without suffix ({class}`~UltraMotionCapture.obj3d.Obj3d` and {class}`~UltraMotionCapture.obj4d.Obj4d`) are the basic classes for 3D and 4D objects. Only basic features like loading from 3dMD scanning data and sampling the point cloud are realised.
- The classes with suffix `_Kps` ({class}`~UltraMotionCapture.obj3d.Obj3d_Kps` and {class}`~UltraMotionCapture.obj4d.Obj4d_Kps`) are derived from {class}`~UltraMotionCapture.obj3d.Obj3d` and {class}`~UltraMotionCapture.obj4d.Obj4d`, respectively. The major development is attaching key points ({class}`~UltraMotionCapture.kps.Kps`) to it.
- The classes with suffix `_Deform` ({class}`~UltraMotionCapture.obj3d.Obj3d_Deform` and {class}`~UltraMotionCapture.obj4d.Obj4d_Deform`) are derived from {class}`~UltraMotionCapture.obj3d.Obj3d_Kps` and {class}`~UltraMotionCapture.obj4d.Obj4d_Kps`, respectively. The major development is attaching the displacement field ({class}`~UltraMotionCapture.field.Trans_Rigid` and {class}`~UltraMotionCapture.field.Trans_Nonrigid`) to it.

```{tip}
At the page of each module, such as {mod}`UltraMotionCapture.obj4d`, an inheritance relationship graph is shown under the table of classes.
```

Now let's go back to the original question: **the 4D object class provided by the core modules doesn't have many functions for analysis. Such functions will be realised in the derived classes.** Specifically, when we'd like to extend the ability of any of the classes that we discussed upwards, we derive a new class and insert/revise attributes or functions in it to fit the need.

In this way, with one set of unified core modules, this package can be fine-tuned for any future analysis demands. **With all the extended classes serving for advanced analysis, we form a future-proof, evolvable ecosystem for human factor research**.

```{danger}
The extended classes should be placed in {mod}`UltraMotionCapture.analysis` sub-package. Since the core modules are providing a skeleton for all downstream classes, it shall not be modified unless there are very strong reasons to do so, otherwise unpredictable issues may emerge.
```

```{attention}
Considering the intensive use of object-orientated development, developers involved in this project are expected to be proficient in Python object-orientated programming. 
```

## Overall structure of the core modules

The overall structure of the core modules is illustrated below. Noted that the solid arrow pointing from `class A` to `class B` indicates that `class B` is derived from `class A`, while the dotted arrow indicates that a `class A` object contains a `class B` object as an attribute:

```{figure} figures/overall_structure.png
Overall structure of the core modules.
```