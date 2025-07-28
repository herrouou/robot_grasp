# robot_grasp
<img width="686" height="504" alt="image" src="https://github.com/user-attachments/assets/d8ce50b7-367d-44da-91b5-61412b7b8504" />

This is a repo which includes the grasping manipulation for **cylinder** and **cuboid**. We use position control in cartesian space with null space control in joint space to grasp the object.

<img width="2908" height="809" alt="图片1" src="https://github.com/user-attachments/assets/f792571b-d4fc-4525-9597-8ef6c804c376" />

## Running with python env
To run the demo, please make sure in your environment there's mujoco-python library.
For cylinder grasping, please go to `/cylinder' and run
```
python3 grasp_cylinder.py
```
For box grasping, please go to `/box' and run
```
python3 grasp_box.py
```
## Running with ROS env
Although the result on ROS was not good, we provided the ROS workspace for grasping control.
Make sure that you have ubuntu 20 and ROS noetic on local machine.
Please install ROS Noetic on ubuntu 20.04. And rum:

```
sudo apt install libglfw3-dev
sudo apt install ros-noetic-combined-robot-hw
sudo apt install ros-noetic-boost-sml
```
Then install the mujoco 3.3.3 as following:
```
mkdir /home/username/.mujoco
```
Then download the tar.gz file:
https://github.com/google-deepmind/mujoco/releases/download/3.3.3/mujoco-3.3.3-linux-x86_64.tar.gz
Run:
```
tar -xvf mujoco-3.3.3-linux-x86_64.tar.gz -C ~/.mujoco/
```
Add these into your `.bashrc` file
```
export MUJOCO_DIR=~/.mujoco/mujoco-3.3.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_DIR/lib
export LIBRARY_PATH=$LIBRARY_PATH:$MUJOCO_DIR/lib
```

Then go to the catkin_ws and do
```
catkin build
source devel/setup.bash 
```
Run the launch file:
```
roslaunch allegro_hand_grasp_plugin grasp_demo.launch
```
