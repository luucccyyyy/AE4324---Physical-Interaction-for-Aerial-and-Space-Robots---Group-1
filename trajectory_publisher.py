import rclpy
import numpy as np
import time
from geometry_msgs.msg import Point 
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from visualization_msgs.msg import Marker
from .robot_arm import Edubot
class TrajectoryPlayer(Node):

    def __init__(self):
        super().__init__('trajectory_player')
        self.robot = Edubot()
        #Publisher
        self.publisher_ = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.marker_pub = self.create_publisher(Marker, "traj_markers", 10)

        self.file_path = "/home/guusje-schellekens/edubot_ws/src/edubot/ros_ws/src/python_controllers/python_controllers/joint_trajectory.csv"
        self.marker = Marker()
        self.marker.header.frame_id = "world"  #robot base frame
        self.marker.type = Marker.SPHERE_LIST  #shows all points at once
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.01
        self.marker.scale.y = 0.01
        self.marker.scale.z = 0.01
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.points = []  #empty list to store points
        #Start playing
        self.publish_trajectory()

    def publish_trajectory(self):

        #Load trajectory
        try:
            q_array = np.loadtxt(self.file_path, delimiter=",", skiprows=1)
        except Exception as e:
            self.get_logger().error(f"Failed to load file: {e}")
            return

        self.get_logger().info(f"Loaded trajectory with {len(q_array)} points")

        joint_signs = [1, 1, 1, 1, 1]

        input("Press ENTER to start trajectory...")

        for idx, q in enumerate(q_array):

            if np.any(np.isnan(q)):
                self.get_logger().warn(f"Skipping invalid point {idx}")
                continue

            msg = JointTrajectory()

            msg.joint_names = [
                "Shoulder_Rotation",
                "Shoulder_Pitch",
                "Elbow",
                "Wrist_Pitch",
                "Wrist_Roll"
            ]

            msg.header.stamp = self.get_clock().now().to_msg()

            point = JointTrajectoryPoint()

            q_signed = [q[i] * joint_signs[i] for i in range(5)]

            point.positions = q_signed

            ee_pos = self.robot.forward_kinematics(q_signed)  

            #Update marker
            self.marker.header.stamp = self.get_clock().now().to_msg()
            #Add a point to SPHERE_LIST marker
            p = Point()
            p.x = ee_pos[0]
            p.y = ee_pos[1]
            p.z = ee_pos[2]
            self.marker.points.append(p)

            point.time_from_start = Duration(sec=0)

            msg.points = [point]

            self.publisher_.publish(msg)
            self.marker_pub.publish(self.marker)

            self.get_logger().info(f"[{idx}] {np.round(q_signed, 3)}")

            #Control speed 
            time.sleep(0.05)

        self.get_logger().info("Trajectory complete!")


def main(args=None):
    rclpy.init(args=args)

    node = TrajectoryPlayer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()