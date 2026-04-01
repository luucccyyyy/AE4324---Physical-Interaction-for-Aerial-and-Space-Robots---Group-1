import rclpy
import math
import numpy as np
from rclpy.node import Node
from .robot_arm import Edubot
import os
import pandas as pd
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pynput import keyboard
import time

class FollowTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        #Load in edubot
        self.robot = Edubot()
        self.joint_home = np.array([
            0,
            np.deg2rad(105),
            np.deg2rad(-70),
            np.deg2rad(-60),
            np.deg2rad(90),
            0.5  # gripper
        ])

        self.current_joints = self.joint_home.copy() #Start at home position

        #load trajectory parameters
        self._beginning = self.get_clock().now()

        #For the gripper control
        self.gripper_active = False
        self.keys_pressed = {}
        def on_press(key):
            try:
                self.keys_pressed[key.char] = True
            except AttributeError:
                if key == keyboard.Key.up:
                    self.keys_pressed['up'] = True
                elif key == keyboard.Key.down:
                    self.keys_pressed['down'] = True
                elif key == keyboard.Key.left:
                    self.keys_pressed['left'] = True
                elif key == keyboard.Key.right:
                    self.keys_pressed['right'] = True

        def on_release(key):
            try:
                self.keys_pressed[key.char] = False
            except AttributeError:
                if key == keyboard.Key.up:
                    self.keys_pressed['up'] = False
                elif key == keyboard.Key.down:
                    self.keys_pressed['down'] = False
                elif key == keyboard.Key.left:
                    self.keys_pressed['left'] = False
                elif key == keyboard.Key.right:
                    self.keys_pressed['right'] = False

        #Start keyboard listener
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        #Create publisher and start timer that runs through trajectory
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.marker_pub = self.create_publisher(Marker, 'traj_markers', 10)

        #Start at Home Position
        self.go_home()

        #Going from pre-gasp/hovering position to grasp
        grape_position = np.array([0.05, 0.1, 0.07])
        pre_grasp = grape_position + np.array([0, 0, 0.05])
        grape_joint_state, _ = self.robot.inverse_kinematics_newton_raphson(grape_position)
        pre_grasp_joint_state, _ = self.robot.inverse_kinematics_newton_raphson(pre_grasp)
        choice = input("Type 0 for move_j or 1 for move_l: ")
        move_l = (choice == "1")
        if move_l:
            self.move_l(pre_grasp, 5, 50)
            self.move_l(grape_position, 7, 100)
        else:
            self.move_j(pre_grasp_joint_state, 5, 50)
            self.move_j(grape_joint_state, 7, 100)

        #Close gripper
        self.gripper_control()

        #Lift back up to pre-grasp height before moving horizontally next
        if move_l:
            self.move_l(pre_grasp, 7, 100)  
        else:
            self.move_j(pre_grasp_joint_state, 7, 100)  

        #Move to above dropping position
        dropping_position = np.array([-0.05, 0.1, 0.07])
        pre_drop = dropping_position + np.array([0, 0, 0.05])
        dropping_joint_state, _ = self.robot.inverse_kinematics_newton_raphson(dropping_position)
        pre_drop_joint_state, _ = self.robot.inverse_kinematics_newton_raphson(pre_drop)

        if move_l:
            self.move_l(pre_drop, 5, 50)
            self.move_l(dropping_position, 7, 100)
        else:
            self.move_j(pre_drop_joint_state, 5, 50) 
            self.move_j(dropping_joint_state, 7, 100)

        #Open gripper to release
        self.gripper_control()

        #Go back up
        if move_l:
            self.move_l(pre_drop, 7, 100)
        else:
            self.move_j(pre_drop_joint_state, 7, 100)

        #Go home as midpoint
        self.go_home(self.current_joints)

        #Move to above previous dropping position again to pick back up
        if move_l:
            self.move_l(pre_drop, 5, 50)
            self.move_l(dropping_position, 7, 100)
        else:
            self.move_j(pre_drop_joint_state, 5, 50)
            self.move_j(dropping_joint_state, 7, 100)

        #Close gripper to pick up
        self.gripper_control()

        #Lift up
        if move_l:
            self.move_l(pre_drop, 7, 100)
        else:
            self.move_j(pre_drop_joint_state, 7, 100)


        #Return to above original position
        if move_l:
            self.move_l(pre_grasp, 5, 50)
            self.move_l(grape_position, 7, 100)
        else:
            self.move_j(pre_grasp_joint_state, 5, 50)
            self.move_j(grape_joint_state, 7, 100)

        #Open gripper to place back
        self.gripper_control()

        #Lift and go home
        if move_l:
            self.move_l(pre_grasp, 7, 100)
        else:
            self.move_j(pre_grasp_joint_state, 7, 100)
        self.go_home(self.current_joints)

    def go_home(self, starting_joints=None):
        #takes the robot to the home position
        if starting_joints is not None:
            self.move_j(self.joint_home, 5, 50)
        else:
            #If no starting point provided, just publish the home position
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            point = JointTrajectoryPoint()
            point.positions = self.joint_home
            msg.points = [point]
            self._publisher.publish(msg)
        return

    def gripper_control(self):
        print("Now actively controlling gripper. Press a and d to control gripper and q to move on")
        self.gripper_active = True
        change_gripper_by = 0.05 #rad
        period = 0.05 #s

        #Small delay to ensure listener is active
        time.sleep(0.1)

        while self.gripper_active:
            gripper_change = 0

            #Control gripper with a/d keys
            if self.keys_pressed.get('d', False):
                gripper_change += change_gripper_by #rad
            if self.keys_pressed.get('a', False):
                gripper_change -= change_gripper_by #rad

            #make sure that the gripper cannot be commanded to overactuate
            if self.current_joints[5] + gripper_change > np.pi / 2 or \
                    self.current_joints[5] + gripper_change < -np.pi / 2:
                gripper_change = 0

            self.current_joints[5] += gripper_change
            print(f"Current gripper value = {np.round(self.current_joints[5], 4)}", end="\r")

            #Check for the quit message to continue
            if self.keys_pressed.get('q', False):
                self.gripper_active = False

            #now publish the current joint with updated gripper as the new position
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            point = JointTrajectoryPoint()

            point.positions = self.current_joints.copy()
            msg.points = [point]
            self._publisher.publish(msg)

            time.sleep(period)

        print("Gripper control has been quit. Moving on...")
        return

    def move_j(self, final_joints, traj_time=5, num_points=50):
        #Takes in final joints for q0 to q3, not the gripper
        wait_time = traj_time / num_points
        final_joints = final_joints.copy()
        if len(final_joints) == 5:
            final_joints = np.append(final_joints, self.current_joints[5])
        final_joints[4] = np.deg2rad(90)

        interpolated_trajectory = np.array([
            np.linspace(self.current_joints[i], final_joints[i], num_points)
            for i in range(6)
        ]).T

        for joint_state in interpolated_trajectory:
            #Publish the point in the current trajectory
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            joint_names = [
                "Shoulder_Rotation",
                "Shoulder_Pitch",
                "Elbow",
                "Wrist_Pitch",
                "Wrist_Roll",
                "Gripper"
            ]

            joint_signs = [1, 1, 1, 1, 1, 1]
            point = JointTrajectoryPoint()
            q_signed = [joint_state[i] * joint_signs[i] for i in range(6)]
            point.positions = q_signed
            msg.points = [point]
            self._publisher.publish(msg)

            #Wait for the joint so it doesn't send them all at once
            time.sleep(wait_time)
        print("Move in J completed. Moving on...")
        self.current_joints = final_joints
        return

    def move_l(self, final_pos, traj_time, num_points, final_gripper=None):
        wait_time = traj_time/num_points
        current_pos = self.robot.forward_kinematics(self.current_joints)
        interpolated_trajectory = np.array([np.linspace(current_pos[i], final_pos[i], num_points) for i in range(len(final_pos))]).T
        joint_array = np.zeros((np.size(interpolated_trajectory, 0), 6))

        guess = self.current_joints[:5].copy()
        for idx, point in enumerate(interpolated_trajectory):
            q_i, err = self.robot.inverse_kinematics_newton_raphson(point, initial_guess=guess)
            joint_array[idx, :5] = q_i
            joint_array[idx, 4] = np.deg2rad(90)
            joint_array[idx, 5] = self.current_joints[5]
            guess = q_i  #update guess for next point

        for joint_state in joint_array:
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            joint_names = [
                "Shoulder_Rotation",
                "Shoulder_Pitch",
                "Elbow",
                "Wrist_Pitch",
                "Wrist_Roll",
                "Gripper"
            ]

            joint_signs = [1, 1, 1, 1, 1, 1]
            point = JointTrajectoryPoint()
            q_signed = [joint_state[i] * joint_signs[i] for i in range(6)]
            point.positions = q_signed
            msg.points = [point]
            self._publisher.publish(msg)

            #Wait for the joint so it doesn't send them all at once
            time.sleep(wait_time)
        print("Move in L completed. Moving on...")
        self.current_joints = joint_array[-1,:]
        return

def main(args=None):
    rclpy.init(args=args)

    example_traj = FollowTraj()

    rclpy.spin(example_traj)

    example_traj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()