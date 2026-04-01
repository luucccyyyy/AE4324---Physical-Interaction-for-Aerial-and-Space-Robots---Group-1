
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from .robot_arm import Edubot

class Visualizer(Node):
    def __init__(self):
        super().__init__("visualizer")
        self.pub = self.create_publisher(MarkerArray, 'workspace_markers', 10)
        
        self.robot = Edubot()

        #hold all marker points
        self.num_samples = 5
        self.num_markers = 50000
        self.marker_array = MarkerArray()

        self.plot_workspace()

    def plot_workspace(self):
        [q0Bounds, q1Bounds, q2Bounds, q3Bounds, q4Bounds] = self.robot.get_bounds()

        marker_array = MarkerArray()

        for idx in range(self.num_markers):

            r1 = np.random.uniform(q0Bounds[0], q0Bounds[1])
            r2 = np.random.uniform(q1Bounds[0], q1Bounds[1])
            r3 = np.random.uniform(q2Bounds[0], q2Bounds[1])
            r4 = np.random.uniform(q3Bounds[0], q3Bounds[1])
            r5 = np.random.uniform(q4Bounds[0], q4Bounds[1])

            angles = [r1, r2, r3, r4, r5]

            point = self.robot.forward_kinematics(angles)

            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.position.z = float(point[2])
            pose.orientation.w = 1.0

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = idx
            marker.type = Marker.SPHERE

            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            marker.color.a = 1.0
            marker.color.r = 0.2
            marker.color.g = 0.6
            marker.color.b = 1.0

            marker.pose = pose

            marker_array.markers.append(marker)

        self.pub.publish(marker_array)


def main():
    rclpy.init()
    node = Visualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()


