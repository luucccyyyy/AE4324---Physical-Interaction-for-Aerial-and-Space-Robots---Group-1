import rclpy
from .robot_arm import Edubot
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class ExampleTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.publish_point()

    def publish_point(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.joint_names = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll"
        ]
        msg.header.stamp = now.to_msg()

        input_string = input("Put in your desired joint angles here (radians): q1,q2,q3,q4,q5\n")
        q = [float(i) for i in input_string.split(",")]

        point = JointTrajectoryPoint()
        point.positions = q
        point.time_from_start = Duration(seconds=2)
        msg.points = [point]
        self._publisher.publish(msg)
        self.get_logger().info(f"Published joint angles (rad): {q}")


def main(args=None):
    rclpy.init(args=args)
    example_traj = ExampleTraj()
    rclpy.spin(example_traj)
    example_traj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()