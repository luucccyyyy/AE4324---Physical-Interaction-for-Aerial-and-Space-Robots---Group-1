# AE4324---Physical-Interaction-for-Aerial-and-Space-Robots---Group-1
Group 1: Lucy Zhang (5549450), Vincent Maan, Guusje Schellekens (5491681)

This repository contains the code written for the AE4324 assignment at TU Delft. It is designed to work together with the Edubot robot and its ROS2 driver, available here: https://github.com/BioMorphic-Intelligence-Lab/edubot

**Requirements**
- ROS2 (Jazzy)
- Python 3.10+
- Edubot ROS2 Workspace (link above)

Install the following Python dependencies in the terminal: 
pip3 install numpy scipy matplotlib pynput pandas sympy --break-system-packages

**Set up**
1. Clone our repository into ROS2 workspace
   cd ~/ae4324_ws/src
   git clone https://github.com/luucccyyyy/AE4324---Physical-Interaction-for-Aerial-and-Space-Robots---Group-1 ae4324_robot
2. Build package
   cd ~/ae4324_ws
   colcon build --packages-select ae4324_robot
   source install/setup.bash
3. Start the Edubot simulation or real robot in a separate terminal:

   for simulation:
   
   source ~/edubot/ros_ws/install/setup.bash
   ros2 launch lerobot sim_position.launch.py

   for real robot:
   
   source ~/edubot/ros_ws/install/setup.bash
   ros2 launch lerobot hw_position.launch.py

**Reproduce our results**

## Task 1.2 (Workspace visualisation)

In terminal:

python3 workspace_visualizer_python.py

RViz visualisation (simulation needs to run as well)

ros2 run ae4324_robot workspace_visualizer.py

## Task 2.1 (IK feasibility)

In terminal: 

Test all 5 poses

python3 inverse_kinematics.py

Send a specific pose to the robot:

ros2 run ae4324_robot point_publisher.py

Example of how to enter a pose: 0.2,0.1,0.4,0,0,-1.57

## Task 2.2 (Multiple IK solutions)

In terminal:

ros2 run ae4324_robot angle_publisher.py

Example of how to enter a joint angle: 1.303,-0.182,0.446,1.307,-0.267

## Task 2.3 (Flame trajectory)

In terminal: 

ros2 run ae4324_robot trajectory_publisher.py

Task 3.3 (Constant velocity control)

In terminal: 

python3 velocity_test.py

## Task 4 (Pick and place)

In terminal:

ros2 run ae4324_robot pick_and_place.py

## Task 5 (Cube stacking)

In terminal:

ros2 run ae4324_robot cube_stacking.py


**File Overview**

| File | Description | Task |
|------|-------------|------|
| `robot_arm.py` | Core library: FK, IK, Jacobian | All |
| `workspace_visualizer_python.py` | Python workspace plots (matplotlib) | 1.2 |
| `workspace_visualizer.py` | RViz workspace visualisation node | 1.2 |
| `inverse_kinematics.py` | IK feasibility testing for all 5 poses | 2.1 |
| `point_publisher.py` | Send a Cartesian pose to the robot | 2.1 |
| `angle_publisher.py` | Send joint angles directly to the robot | 2.2 |
| `trajectory_publisher.py` | Execute flame trajectory on the robot | 2.3 |
| `velocity_test.py` | Constant velocity Cartesian control | 3.3 |
| `pick_and_place.py` | Pick and place a fragile object | 4 |
| `cube_stacking.py` | Autonomously stack 3 cubes | 5 |
| `trace_video.py` | Post-process video to track end-effector | 2.3 |
| `tu_flame.csv` | Flame outline pixel coordinates | 2.3 |
| `tu_flame_cartesian_trajectory.csv` | Pre-computed flame Cartesian trajectory | 2.3 |
