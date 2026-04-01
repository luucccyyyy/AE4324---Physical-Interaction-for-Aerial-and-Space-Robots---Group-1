import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from python_controllers.robot_arm import Edubot


class Task33VelocityControl(Node):
    def __init__(self):
        super().__init__('velocity_controller_task33')
        self.robot = Edubot()

        self.joint_signs = [1, 1, 1, 1, 1]
        self.joint_names = [
            "Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"
        ]

        self.current_q = None
        self.start_time = None

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile_sensor_data)

        self.publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # 50Hz
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("Task 3.3 Velocity Controller Started!")

    def joint_state_callback(self, msg):

        try:
            q_temp = np.zeros(5)
            q_dot_temp = np.zeros(5)  # array for velocities
            for i, name in enumerate(self.joint_names):
                idx = msg.name.index(name)
                q_temp[i] = msg.position[idx] * self.joint_signs[i]

                if len(msg.velocity) > idx:
                    q_dot_temp[i] = msg.velocity[idx] * self.joint_signs[i]

            self.current_q = q_temp
            self.current_q_dot = q_dot_temp  #Save it
        except ValueError:
            pass

    def control_loop(self):
        if self.current_q is None:
            return

        #TIMER
        now = self.get_clock().now()

        if self.start_time is None:
            self.start_time = now.nanoseconds / 1e9

        t = (now.nanoseconds / 1e9) - self.start_time

        # 1. Calculate exactly where the gripper is right now
        current_pos = self.robot.forward_kinematics(self.current_q)

        # 2. Lock in starting z-height on the very first loop
        if not hasattr(self, 'target_z'):
            self.target_z = current_pos[2]

        # 3. Calculate error
        z_error = self.target_z - current_pos[2]

        # 4. Proportional gain
        Kp = 2.0
        v_z_correction = Kp * z_error

        #CONSTANT VELOCITY TRAJECTORY
        if t < 15.0:
            # Move +x at 4 cm/s and y -4 cm/s but constant z
            self.v_desired = np.array([0.04, -0.04, v_z_correction])
            phase_name = "MOVING (+X)"
        else:
            # When stopped, only hold the z height
            self.v_desired = np.array([0.0, 0.0, v_z_correction])
            phase_name = "STOPPED"


        # 1. Evaluate Jacobian at current posture
        J_full = self.robot.get_jacobian_numeric(self.current_q)
        J_v = J_full[:3, :]

        # 2. DLS
        lambda_sq = 0.001
        J_v_float = J_v.astype(float)
        J_pinv_safe = J_v_float.T @ np.linalg.inv(J_v_float @ J_v_float.T + lambda_sq * np.eye(3))

        # 3. Calculate joint speeds
        q_dot = J_pinv_safe @ self.v_desired
        q_dot = np.clip(q_dot, -1.0, 1.0)

        # 4. Formating message
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        point = JointTrajectoryPoint()

        for i in range(5):
            point.velocities.append(float(q_dot[i] * self.joint_signs[i]))

        point.velocities.append(0.0)  # Gripper
        msg.points = [point]
        self.publisher.publish(msg)

        self.get_logger().info(f"[{phase_name}] Z-Error: {z_error:.4f} | Speeds: {np.round(q_dot, 3)}",
                               throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = Task33VelocityControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before shutdown — safety!
        stop_msg = JointTrajectory()
        stop_msg.joint_names = node.joint_names + ["Gripper"]
        stop_point = JointTrajectoryPoint()
        stop_point.time_from_start.nanosec = 20_000_000
        stop_point.velocities = [0.0] * 6
        stop_msg.points = [stop_point]
        node.publisher.publish(stop_msg)
        node.get_logger().info("Stop command sent. Shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()