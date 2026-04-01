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

class StackCubes(Node):

    def __init__(self):
        super().__init__('stack_cubes')
        self.robot = Edubot()
        self.joint_home = np.array([
            0,
            np.deg2rad(105),
            np.deg2rad(-70),
            np.deg2rad(-60),
            np.deg2rad(90),
            0.5  # gripper open
        ])
        self.current_joints = self.joint_home.copy()
        self._beginning = self.get_clock().now()

        CUBE_HEIGHT = 0.02  # 2 cm
        GRIPPER_CLOSED = 0.35  # rad
        GRIPPER_OPEN = 0.5

        cube_positions = [
            np.array([0.10, 0.10, 0.01]),   # cube 1
            np.array([0.10, 0.10, 0.01]),   # cube 2
            np.array([0.10, 0.10, 0.01]),   # cube 3
        ]

        stack_x, stack_y = -0.10, 0.10
        stack_positions = [
            np.array([stack_x, stack_y, 0.01]),                 
            np.array([stack_x, stack_y, 0.01 + CUBE_HEIGHT]),    
            np.array([stack_x, stack_y, 0.01 + 2 * CUBE_HEIGHT]),
        ]

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

        def on_release(key):
            try:
                self.keys_pressed[key.char] = False
            except AttributeError:
                if key == keyboard.Key.up:
                    self.keys_pressed['up'] = False
                elif key == keyboard.Key.down:
                    self.keys_pressed['down'] = False

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.marker_pub = self.create_publisher(Marker, 'traj_markers', 10)

        self.go_home()

        for i in range(3):
            print(f"\n=== Picking cube {i+1} ===")
            self.pick(cube_positions[i])

            print(f"\n=== Placing cube {i+1} at stack level {i+1} ===")
            self.place(stack_positions[i])

            print(f"\n=== Cube {i+1} stacked. Returning home. ===")
            self.go_home(self.current_joints)

        print("\n=== All 3 cubes stacked! ===")

    def pick(self, position):
        pre_grasp = position + np.array([0, 0, 0.06])  #6cm above: safe transit height
        grasp_approach = position + np.array([0, 0, 0.02])
        #Open gripper before approaching
        self.set_gripper_open()

        #Approach from above
        self.move_j_to_pos(pre_grasp, traj_time=3, num_points=30)

        #Descend straight down to object

        self.move_l(grasp_approach, traj_time=2, num_points=20)
        self.move_l(position, traj_time=2, num_points=20)
        #Close gripper to grasp
        print("Close gripper (a=close, d=open, q=done): ")
        self.gripper_control()

        #Lift straight back up
        self.move_l(pre_grasp, traj_time=3, num_points=30)

    def place(self, position):
        """Move to above stack position, descend, open gripper, lift back up."""
        pre_place = position + np.array([0, 0, 0.06])  # 6cm above target

        #Transit to above stack
        self.move_j_to_pos(pre_place, traj_time=3, num_points=30)

        #Descend carefully onto the stack
        self.move_l(position, traj_time=4, num_points=40)

        #Open gripper to release
        print("Open gripper (d=open, a=close, q=done): ")
        self.gripper_control()

        #Lift straight back up to avoid knocking stack
        self.move_l(pre_place, traj_time=3, num_points=30)

    def move_j_to_pos(self, target_pos, traj_time=3, num_points=30):
        """Helper: compute IK for a Cartesian position and move_j there."""
        joint_state, err = self.robot.inverse_kinematics_newton_raphson(
            target_pos, initial_guess=self.current_joints[:5].copy()
        )
        if err > 1e-3:
            print(f"WARNING: IK did not converge, error={err:.4f}")
        self.move_j(joint_state, traj_time, num_points)

    def set_gripper_open(self):
        """Publish a single message with gripper open, no movement."""
        self.current_joints[5] = 0.5
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        point = JointTrajectoryPoint()
        point.positions = self.current_joints.copy().tolist()
        msg.points = [point]
        self._publisher.publish(msg)

    def go_home(self, starting_joints=None):
        if starting_joints is not None:
            self.move_j(self.joint_home, 4, 40)
        else:
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            point = JointTrajectoryPoint()
            point.positions = self.joint_home.tolist()
            msg.points = [point]
            self._publisher.publish(msg)

    def gripper_control(self):
        print("a=close  d=open  q=continue")
        self.gripper_active = True
        change_gripper_by = 0.05
        period = 0.05
        time.sleep(0.1)

        while self.gripper_active:
            gripper_change = 0
            if self.keys_pressed.get('d', False):
                gripper_change += change_gripper_by
            if self.keys_pressed.get('a', False):
                gripper_change -= change_gripper_by

            new_val = self.current_joints[5] + gripper_change
            if -np.pi/2 < new_val < np.pi/2:
                self.current_joints[5] = new_val

            print(f"Gripper = {np.round(self.current_joints[5], 3)}", end="\r")

            if self.keys_pressed.get('q', False):
                self.gripper_active = False

            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            point = JointTrajectoryPoint()
            point.positions = self.current_joints.copy().tolist()
            msg.points = [point]
            self._publisher.publish(msg)
            time.sleep(period)

        print("\nGripper set. Moving on...")

    def move_j(self, final_joints, traj_time=4, num_points=40):
        wait_time = traj_time / num_points
        final_joints = final_joints.copy()
        if len(final_joints) == 5:
            final_joints = np.append(final_joints, self.current_joints[5])

        interpolated_trajectory = np.array([
            np.linspace(self.current_joints[i], final_joints[i], num_points)
            for i in range(6)
        ]).T

        for joint_state in interpolated_trajectory:
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            point = JointTrajectoryPoint()
            point.positions = joint_state.tolist()
            msg.points = [point]
            self._publisher.publish(msg)
            time.sleep(wait_time)

        print("move_j done.")
        self.current_joints = final_joints

    def move_l(self, final_pos, traj_time=3, num_points=30):
        wait_time = traj_time / num_points
        current_pos = self.robot.forward_kinematics(self.current_joints)
        interpolated_trajectory = np.array([
            np.linspace(current_pos[i], final_pos[i], num_points)
            for i in range(3)
        ]).T

        joint_array = np.zeros((num_points, 6))
        guess = self.current_joints[:5].copy()

        for idx, pos in enumerate(interpolated_trajectory):
            q_i, err = self.robot.inverse_kinematics_newton_raphson(pos, initial_guess=guess)
            if err > 1e-3:
                print(f"WARNING: IK did not converge at step {idx}, err={err:.4f}")
            joint_array[idx, :5] = q_i
            joint_array[idx,4] =np.deg2rad(90)
            joint_array[idx, 5] = self.current_joints[5]
            guess = q_i

        for joint_state in joint_array:
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            point = JointTrajectoryPoint()
            point.positions = joint_state.tolist()
            msg.points = [point]
            self._publisher.publish(msg)
            time.sleep(wait_time)

        print("move_l done.")
        self.current_joints = joint_array[-1, :]

    def set_gripper(self, angle, settle_time=0.5):
        """Set gripper to a specific angle and wait for it to reach position."""
        self.current_joints[5] = angle
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        point = JointTrajectoryPoint()
        point.positions = self.current_joints.copy().tolist()
        msg.points = [point]
        self._publisher.publish(msg)
        time.sleep(settle_time)  # wait for gripper to physically close/open

def main(args=None):
    rclpy.init(args=args)
    node = StackCubes()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()