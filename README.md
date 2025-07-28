# robot_grasp
<img width="686" height="504" alt="image" src="https://github.com/user-attachments/assets/d8ce50b7-367d-44da-91b5-61412b7b8504" />

This is a repo which includes the grasping manipulation for **cylinder** and **cuboid**. We use position control in cartesian space with null space control in joint space to grasp the object.

<img width="2908" height="809" alt="图片1" src="https://github.com/user-attachments/assets/f792571b-d4fc-4525-9597-8ef6c804c376" />

## Running
To run the demo, please make sure in your environment there's mujoco-python library.
For cylinder grasping, please go to `/cylinder' and run
```
python3 grasp_cylinder.py
```
For box grasping, please go to `/box' and run
```
python3 grasp_box.py
```
