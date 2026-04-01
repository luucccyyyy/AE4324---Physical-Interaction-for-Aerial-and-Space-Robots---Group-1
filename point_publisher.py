import rclpy


from .robot_arm import Edubot, euler_to_rot
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class ExampleTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        self.robot = Edubot()
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
        input_string = input("Put in your desired point here: x,y,z,rx,ry,rz\n")
        vals = [float(i) for i in input_string.split(",")]
        pos = np.array(vals[:3])
        rot = vals[3:]
        R = euler_to_rot(*rot)
        q, _ = self.robot.inverse_kinematics_optimization(pos, R)
        
        point = JointTrajectoryPoint()
        point.positions = [q[0], q[1], q[2], q[3], q[4]]
        joint_signs = [1,1,1,1,1]
        q_signed = [q[i] *joint_signs[i] for i in range(5)]
        point.positions = q_signed
        point.time_from_start = Duration(seconds=2)
        msg.points = [point]
        self._publisher.publish(msg)
        self.get_logger().info(f"Published: {q_signed}")



def main(args=None):
    rclpy.init(args=args)

    example_traj = ExampleTraj()

    rclpy.spin(example_traj)

    example_traj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
