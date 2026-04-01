import numpy as np 
from numpy import sin, cos
from warnings import warn
from scipy.optimize import minimize
from time import time
import sympy as sym
from numpy.linalg import inv 

import subprocess as sub
import pandas as pd 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])

def roty(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0,             1, 0,             0],
        [-np.sin(theta),0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotx(theta):
    return np.array([
        [1, 0,              0,             0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def trans(x, y, z):
    return np.array([
        [1,0,0,x],
        [0,1,0,y],
        [0,0,1,z],
        [0,0,0,1]
    ])

def euler_to_rot(rx, ry, rz):
    return rotx(rx)[:3,:3] @ roty(ry)[:3,:3] @ rotz(rz)[:3,:3]

class Edubot():
    def __init__(self):

        self.q0Bounds = [-2, 2]  # Shoulder rotation
        self.q1Bounds = [-1.57, 1.57]  # Shoulder pitch
        self.q2Bounds = [-1.58, 1.58]  # Elbow
        self.q3Bounds = [-1.57, 1.57]  # Wrist pitch
        self.q4Bounds = [-3.14158, 3.14158]

        ''' self.q0Bounds = [-3.14158, 3.14158]
        self.q1Bounds = [-3.14158, 3.14158]
        self.q2Bounds = [-3.14158, 3.14158]
        self.q3Bounds = [-3.14158, 3.14158]
        self.q4Bounds = [-3.14158, 3.14158]'''

        #Joint bounds
        self.T_BW = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.T_SB = trans(0, -0.0452, 0.0165)
        self.T_SU = trans(0, -0.0306, 0.1025) @ roty(-1.57079)
        self.T_UL = trans(0.11257, -0.028, 0)
        self.T_LW = trans(0.0052, -0.1349, 0) @ rotz(1.57079)
        self.T_WG = trans(-0.0601, 0, 0) @ roty(-1.57079)
        self.T_GC = trans(0.0, 0.0, 0.075)
        self.q = np.zeros(5)
        self._init_jacobian_func()


    #Forward Kinematics
    def forward_kinematics(self, q):

        r1, r2, r3, r4, r5 = q[:5]

        T1 = self.T_SB @ rotz(r1)
        T2 = self.T_SU @ rotz(r2)
        T3 = self.T_UL @ rotz(r3)
        T4 = self.T_LW @ rotz(r4)
        T5 = self.T_WG @ rotz(r5)

        T = self.T_BW @ T1 @ T2 @ T3 @ T4 @ T5 @ self.T_GC

        return T[:3, 3]

    def get_transform(self, q):

        r1, r2, r3, r4, r5 = q

        T1 = self.T_SB @ rotz(r1)
        T2 = self.T_SU @ rotz(r2)
        T3 = self.T_UL @ rotz(r3)
        T4 = self.T_LW @ rotz(r4)
        T5 = self.T_WG @ rotz(r5)

        return self.T_BW @ T1 @ T2 @ T3 @ T4 @ T5 @ self.T_GC

    def get_current_joint_angles(self):
        """Return the current joint angles of the robot as a numpy array."""
        return self.q.copy()

    def set_joint_angles(self, q_new):
        """Update the robot's current joint angles."""
        self.q = np.array(q_new)

    def get_bounds(self):
        """Return joint limits as a list in radians."""
        return [self.q0Bounds, self.q1Bounds, self.q2Bounds, self.q3Bounds, self.q4Bounds]

    def get_symbolic_transform(self, q):
        """Identical logic to get_transform, but using SymPy types for differentiation."""
        r1, r2, r3, r4, r5 = q

        def s_rotz(theta):
            return sym.Matrix([
                    [sym.cos(theta), -sym.sin(theta), 0, 0],
                    [sym.sin(theta), sym.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
            ])

        tsb = sym.Matrix(self.T_SB)
        tsu = sym.Matrix(self.T_SU)
        tul = sym.Matrix(self.T_UL)
        tlw = sym.Matrix(self.T_LW)
        twg = sym.Matrix(self.T_WG)
        tgc = sym.Matrix(self.T_GC)
        tbw = sym.Matrix(self.T_BW)

            # Chain them up
        T1 = tsb @ s_rotz(r1)
        T2 = tsu @ s_rotz(r2)
        T3 = tul @ s_rotz(r3)
        T4 = tlw @ s_rotz(r4)
        T5 = twg @ s_rotz(r5)

        return tbw @ T1 @ T2 @ T3 @ T4 @ T5 @ tgc



    def inverse_kinematics_optimization(self, target_position, target_rotation, initial_guess=None):
        x_target, y_target, z_target = target_position

        #Check for Singlarity
        if x_target == 0 and y_target == 0:
            warn("The robot is in singularity!")
            q0 = 0
        else: 
            #due to the geometry of the robot, q0 is just the arctan. 
            #also to note, since joints rotate about +z but starts in -x, a positive rotation means the robot goes into negative y and negative x
            print("target x: ", x_target, "\ntarget y: ", y_target)
            q0 = np.arctan2(-y_target, -x_target)

        q0 = np.clip(q0, self.q0Bounds[0], self.q0Bounds[1])
        
        #Since there will be multiple solutions, solve numerically for the rest of the angles
        if initial_guess is None:
            initial_guess = np.array([q0, 0, 0, 0,0])
        else:
            initial_guess = np.array(initial_guess)

        bounds = self.get_bounds()
        
        #Objective function.
        def objective(q):
            T = self.get_transform(q)
            pos = T[:3,3]
            R = T[:3,:3]
            pos_error = np.sum((pos - target_position)**2) #MSE
            rot_error = np.sum((R-target_rotation)**2)
            return pos_error + 0.1 * rot_error
        
        #Run the optimization
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            tol=1e-10,
            options={'maxiter': 500}
        )
        
        #Check if optimization was successful
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        #Compute final error
        final_pos = self.forward_kinematics(result.x)
        error = np.linalg.norm(final_pos - target_position)
        print(f"Final position error: {error:.6f}")
        
        return result.x, error
    
    def inverse_kinematics_newton_raphson(self, target_position, initial_guess=None,
                                          tol=1e-4, max_iter=200, print_messages=False):
        bounds = self.get_bounds()
        lam = 1e-4  

        if initial_guess is not None:
            q = np.array(initial_guess, dtype=float)
        else:
            x_t, y_t, _ = target_position
            if x_t == 0 and y_t == 0:
                q0_seed = 0.0
            else:
                q0_seed = np.clip(np.arctan2(-y_t, -x_t),
                                  self.q0Bounds[0], self.q0Bounds[1])
            q = np.array([q0_seed, 0.0, 0.0, 0.0, 0.0])

        error_norm = float('inf')

        for i in range(max_iter):
            pos_current = self.get_transform(q)[:3, 3]
            delta_pos = target_position - pos_current
            error_norm = np.linalg.norm(delta_pos)

            if error_norm < tol:
                if print_messages:
                    print(f"Converged in {i} iterations, error={error_norm:.2e}")
                break

            #Damped least-squares pseudoinverse: J^T (J J^T + lam I)^-1
            J = self.get_jacobian_numeric(q)  # 3 × 5
            JJT = J @ J.T  # 3 × 3
            delta_q = J.T @ np.linalg.solve(JJT + lam * np.eye(3), delta_pos)

            q = q + delta_q

            #Enforce joint limits
            for j, (lo, hi) in enumerate(bounds):
                q[j] = np.clip(q[j], lo, hi)

        else:
            if print_messages:
                print(f"Did not converge after {max_iter} iter, error={error_norm:.2e}")

        final_error = np.linalg.norm(target_position - self.get_transform(q)[:3, 3])
        return q, final_error

    def get_jacobian(self, print_final_transformation=False):
        q0 = sym.Symbol("q0");
        q1 = sym.Symbol("q1");
        q2 = sym.Symbol("q2");
        q3 = sym.Symbol("q3");
        q4 = sym.Symbol("q4")
        pi = sym.pi

        def s_rotz(theta):
            return sym.Matrix([
                [sym.cos(theta), -sym.sin(theta), 0, 0],
                [sym.sin(theta), sym.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        def s_roty(theta):
            return sym.Matrix([[sym.cos(theta), 0, sym.sin(theta), 0],
                               [0, 1, 0, 0],
                               [-sym.sin(theta), 0, sym.cos(theta), 0],
                               [0, 0, 0, 1]])

        T_BW = sym.Matrix([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T_SB = sym.Matrix(trans(0, -0.0452, 0.0165))
        T_SU = sym.Matrix(trans(0, -0.0306, 0.1025)) @ s_roty(-pi / 2)
        T_UL = sym.Matrix(trans(0.11257, -0.028, 0))
        T_LW = sym.Matrix(trans(0.0052, -0.1349, 0)) @ s_rotz(pi / 2)
        T_WG = sym.Matrix(trans(-0.0601, 0, 0)) @ s_roty(-pi / 2)
        T_GC = sym.Matrix(trans(0.0, 0.0, 0.075))

        #Chaining
        T0 = T_BW
        T1 = T0 @ T_SB @ s_rotz(q0)
        T2 = T1 @ T_SU @ s_rotz(q1)
        T3 = T2 @ T_UL @ s_rotz(q2)
        T4 = T3 @ T_LW @ s_rotz(q3)
        T5 = T4 @ T_WG @ s_rotz(q4)
        T_ee = T5 @ T_GC

        x, y, z = T_ee[0, 3], T_ee[1, 3], T_ee[2, 3]

        #Derivatives for Linear Jacobian 
        j_11 = sym.diff(x, q0);
        j_12 = sym.diff(x, q1);
        j_13 = sym.diff(x, q2);
        j_14 = sym.diff(x, q3);
        j_15 = sym.diff(x, q4)
        j_21 = sym.diff(y, q0);
        j_22 = sym.diff(y, q1);
        j_23 = sym.diff(y, q2);
        j_24 = sym.diff(y, q3);
        j_25 = sym.diff(y, q4)
        j_31 = sym.diff(z, q0);
        j_32 = sym.diff(z, q1);
        j_33 = sym.diff(z, q2);
        j_34 = sym.diff(z, q3);
        j_35 = sym.diff(z, q4)

        #Each joint rotates around the Z-axis of the frame BEFORE the rotation is applied
        z0 = T0[:3, 2]  # Axis for q0 
        z1 = (T1 @ sym.Matrix(self.T_SU))[:3, 2]  #Axis for q1 
        z2 = (T2 @ sym.Matrix(self.T_UL))[:3, 2]  #Axis for q2
        z3 = (T3 @ sym.Matrix(self.T_LW))[:3, 2]  #Axis for q3
        z4 = (T4 @ sym.Matrix(self.T_WG))[:3, 2]  #Axis for q4

        #6x5 Matrix (3 Linear rows + 3 Rotational rows)
        jacobian = sym.Matrix([
            [j_11, j_12, j_13, j_14, j_15],  #Linear X
            [j_21, j_22, j_23, j_24, j_25],  #Linear Y
            [j_31, j_32, j_33, j_34, j_35],  #Linear Z
            [z0[0], z1[0], z2[0], z3[0], z4[0]],  #Angular X
            [z0[1], z1[1], z2[1], z3[1], z4[1]],  #Angular Y
            [z0[2], z1[2], z2[2], z3[2], z4[2]]   #Angular Z
        ])
        return jacobian


    def solve_jacobian(self, q_array, unit="degrees"):
        
        #want radian inputs to the jacobian, so convert if necessary
        if unit.lower() == "degrees":
            #degree input to rad
            q_array = q_array * np.pi / 180
        elif unit.lower() == "radians":
            #change nothing
            pass 
        else: 
            warn("Incorrect Argument for Unit in Solve_Jacobian Function!")

        q0 = sym.Symbol("q0"); q1 = sym.Symbol("q1"); q2 = sym.Symbol("q2"); q3 = sym.Symbol("q3")
        q_sol = {q0: q_array[0], 
                 q1: q_array[1], 
                 q2: q_array[2], 
                 q3: q_array[3]}
        
        solved_jacobian = self.J.subs(q_sol)
        j_array = np.asarray(solved_jacobian)
        return j_array.astype(float)

    def _init_jacobian_func(self):
        qs = sym.symbols('q0:5')
        T_ee = self.get_symbolic_transform(qs)
        pos = T_ee[:3, 3]

        #Linear jacobian
        J_linear = pos.jacobian(qs)

        #Convert to a fast lambda function
        self.jacobian_func = sym.lambdify([qs], J_linear, 'numpy')

    def get_jacobian_numeric(self, q_values):
        return np.array(self.jacobian_func(q_values))

    def get_current_joint_angles(self):
        return self.q

    #assignment 2.3
    def trace_cartesian_trajectory(self, filename, height, dist, generateOutline=False, saveOutput=False, previewTrajectory=False):
        if generateOutline: 
            prefix = filename.split(".")[0]
            outputName = prefix + ".csv"
            result = sub.run(["./Guusje/outliner", filename, outputName], capture_output=True, text=True)
            if result.returncode == 0:
                print("Success! Output:", result.stdout)
            else: 
                print("Error running the executable:", result.stderr)
            filename = outputName
        
        outline = pd.read_csv(filename)

        csv_x = outline.x
        csv_y = outline.y

        min_height = 0.1 #m
        cosnt_x = dist
        width = height * (np.max(csv_x) - np.min(csv_x))/(np.max(csv_y) - np.min(csv_y))

        #scale the actual trajectory
        normalized_y = (csv_y - np.min(csv_y)) / np.max(csv_y - np.min(csv_y))
        normalized_x = (csv_x - np.min(csv_x)) / np.max(csv_x - np.min(csv_x)) - 0.5 
        print(np.min(normalized_x), np.max(normalized_x))

        actual_x = np.ones(len(normalized_x)) * cosnt_x
        actual_y = normalized_x * width
        actual_z = normalized_y * height; actual_z = (np.max(actual_z) - actual_z) + min_height #flip the z-axis bcs flame is upside down

        cartesian_trajectory = np.array([actual_x, actual_y, actual_z])

        #Save the output to a csv file
        if saveOutput: 
            traj_dict = {"x" : actual_x, "y" : actual_y, "z" : actual_z}
            traj_df = pd.DataFrame(traj_dict)
            outputName = "/home/guusje-schellekens/edubot_ws/src/edubot/ros_ws" + "_cartesian_trajectory.csv"
            traj_df.to_csv(outputName)
        
        #plot trajectory 
        if previewTrajectory:
            plt.figure()
            ax = plt.axes(projection ='3d')
            ax.scatter(actual_x, actual_y, actual_z)
            ax.set_title('3D Cartesian Trajectory of Robot')
            ax.set_xlabel("x[m]")
            ax.set_ylabel("y[m]")

            plt.figure()
            plt.plot(actual_y, actual_z)
            plt.xlabel("Robot y-axis")
            plt.ylabel("Robot z-axis")
            plt.title("2D Projection of Robot Trajectory onto yz plane")
            plt.show()

        return cartesian_trajectory

    def cartesian_to_joint_trajectory(self, cartesian_array,
                                      tol=1e-4, max_iter=400,
                                      saveOutput=True,
                                      checkForwardKinematics=True):

        cartesian_array[:, 1] += 0.1 #shift y-axis, otherwise outside workspace..
        pts = np.array(cartesian_array)
        if pts.shape[0] == 3 and pts.ndim == 2:
            pts = pts.T  # → (N, 3)

        N = pts.shape[0]
        q_array = np.full((N, 5), np.nan)

        x0, y0, _ = pts[0]
        if x0 == 0 and y0 == 0:
            q0_seed = 0.0
        else:
            q0_seed = np.clip(np.arctan2(-y0, -x0),
                              self.q0Bounds[0], self.q0Bounds[1])
        guess = np.array([q0_seed, 0.0, 0.0, 0.0, 0.0])

        failed = 0
        for idx, point in enumerate(pts):
            q_i, err_i = self.inverse_kinematics_newton_raphson(
                point,
                initial_guess=guess,
                tol=tol,
                max_iter=max_iter
            )

            if err_i > tol * 10:
                print(f"[{idx:4d}] WARN: point {np.round(point, 4)} unreachable, "
                      f"error={err_i:.4f}")
                failed += 1
                #Keep the previous guess
                continue

            q_array[idx] = q_i
            guess = q_i  #use solved angles as next initial guess
        print(q_array[:5])
        print(f"Trajectory done. {N - failed}/{N} points solved "
              f"({failed} failed/skipped).")

        if saveOutput:
            df = pd.DataFrame(q_array, columns=["q0", "q1", "q2", "q3", "q4"])
            df.to_csv("joint_trajectory.csv", index=False)
            print("Saved to joint_trajectory.csv")

        if checkForwardKinematics:
            point_array = np.zeros((N, 3))
            for idx, q_i in enumerate(q_array):
                if np.any(np.isnan(q_i)):
                    continue
                point_array[idx] = self.get_transform(q_i)[:3, 3]

            pd.DataFrame(point_array, columns=["x", "y", "z"]).to_csv(
                "reconstruction.csv", index=False)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2],
                       c="green", s=2)
            ax.set_title("Reconstructed trajectory (FK check)")
            plt.show()

        return q_array


if __name__=="__main__":

    t_start = time()
    robot = Edubot()
    t_end = time()
    print(f"Time to instantiate: {t_end - t_start}")
    
    #Define what we're checking:
    fk = False
    ik = False
    cartesian_trajectory = False
    joint_trajectory = True
    test_reconstruction = True
    get_symbolic_jacobian = False

    if fk: 
        #CHECKING THE FORWARD KINEMATICS TO MAKE SURE THEY MAKE SENSE
        pos = robot.forward_kinematics(np.array([np.pi/4, 0, 0, 0]), "radians")
        print(pos)

    if ik:
        #CHECKING THE INVERSE KINEMATICS FUNCTION TO MAKE SURE IT WORKS
        point1 = np.array([-0.1, 0.1 ,0.03])
        [q_array_1, errors1] = robot.inverse_kinematics_newton_raphson(point1, plotting=False)
        reconstruction1 = robot.forward_kinematics(q_array_1, "radians")
        print(f"WITH NO ELEV CONSTRAINTS:\nq's to achieve {point1} from jacobian algorithm: {q_array_1} with error: {errors1}.\nReconstructed position: {reconstruction1}\n")  

        #Now try same thing with Analytical IK, specifying the angle
        desired_elev = 0
        q_array_1, _ = robot.inverse_kinematics_analytical(point1, desired_elev)
        reconstruction1 = robot.forward_kinematics(q_array_1, "radians")
        print(f"USING ANALYTICAL SOLVER:\nq's to achieve {point1} from jacobian algorithm: {q_array_1} with error: {0}.\nReconstructed position: {reconstruction1}")  
        
    if cartesian_trajectory:
        #CHECKING THE TRACE_CARTESIAN_TRAJECTORY POINT WITH generateOutline ENABLED AND DISABLED
        robot.trace_cartesian_trajectory("/home/guusje-schellekens/Downloads/tu_flame.csv", 0.15, 0.2, generateOutline=False, saveOutput=True, previewTrajectory=True)
  
    if joint_trajectory:
        #Load CSV manually
        cartesian_df = pd.read_csv("/home/guusje-schellekens/Downloads/tu_flame_cartesian_trajectory.csv")

        #Convert to Nx3 numpy array
        cartesian_array = cartesian_df[['x', 'y', 'z']].to_numpy()

        q_array = robot.cartesian_to_joint_trajectory(
            cartesian_array,
            tol=1e-4,
            max_iter=400,
            saveOutput=True,
            checkForwardKinematics=True
        )
    if test_reconstruction: 
        actual_data = pd.read_csv("/home/guusje-schellekens/Downloads/tu_flame_cartesian_trajectory.csv")
        recon_data = pd.read_csv("/home/guusje-schellekens/edubot_ws/src/edubot/ros_ws/src/python_controllers/python_controllers/reconstruction.csv")

        actual_points = np.array([actual_data.x, actual_data.y, actual_data.z]).T
        recon_points = np.array([recon_data.x, recon_data.y, recon_data.z]).T

        point_idx = np.arange(np.size(actual_points, 0))
        diffs = np.zeros(np.size(actual_points, 0))
        for idx, actual_point in enumerate(actual_points):
            recon_point = recon_points[idx, :]
            diff = actual_point - recon_point
            total_diff = np.linalg.norm(diff)
            diffs[idx] = total_diff

        plt.figure()
        plt.plot(point_idx, diffs)

        plt.figure()
        ax = ax = plt.axes(projection ='3d')
        ax.scatter(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2])

        plt.figure()
        ax = ax = plt.axes(projection ='3d')
        ax.scatter(recon_points[:, 0], recon_points[:, 1], recon_points[:, 2])

        plt.show()
            


    if get_symbolic_jacobian:
        print("Calculating Symbolic Jacobian (this may take a minute)...")
        J_symbolic = robot.get_jacobian(True)
        J_simplified = sym.simplify(J_symbolic)

        print("--- Symbolic Jacobian Matrix ---")
        sym.pprint(J_simplified)
        
        print("\n--- Numerical Jacobian at Pose 1 ---")
        
        q0, q1, q2, q3, q4 = sym.symbols('q0 q1 q2 q3 q4')
        pose1 = {q0: 0.2, q1: 0.2, q2: 0.2, q3: 0, q4: 1.57}

        J_num = J_simplified.subs(pose1).evalf()
    
        J_clean = sym.nsimplify(J_simplified, tolerance=1e-5, rational=True)
        sym.pprint(J_clean)

        print("Calculating Symbolic Jacobian...")

        print("Row for row")
        if get_symbolic_jacobian:
            print("Calculating and Simplifying Symbolic Jacobian...")
            J_raw = robot.get_jacobian()

            J_collapsed = sym.trigsimp(J_raw)

            J_final = sym.nsimplify(J_collapsed, tolerance=1e-4, rational=False)

            row_labels = ["Linear vx", "Linear vy", "Linear vz",
                          "Angular wx", "Angular wy", "Angular wz"]

            for i in range(J_final.rows):
                print(f"\n--> {row_labels[i]}:")
                sym.pprint(J_final.row(i))

        print("\n--- Task 3.1.2: Numerical Jacobian Evaluation ---")

        task_1_data = [
            [0.2, 0.2, 0.2, 0.000, 1.570, 0.650],  #Pose I
            [0.2, 0.1, 0.4, 0.000, 0.000, -1.570],  #Pose II
            [0.0, 0.0, 0.4, 0.000, -0.785, 1.570],  #Pose III
            [0.0, 0.0, 0.07, 3.141, 0.000, 0.000],  #Pos IV
            [0.0, 0.0452, 0.45, -0.785, 0.000, 3.141]  #Pose V
        ]

        q_syms = sym.symbols('q0 q1 q2 q3 q4')

        for i, data in enumerate(task_1_data):
            target_pos = np.array(data[:3])
            target_euler = data[3:]

            #Euler angles to a 3x3 Rotation Matrix
            target_rot_matrix = euler_to_rot(target_euler[0], target_euler[1], target_euler[2])

            print(f"\n--- Evaluating Pose {i + 1} ---")
            print(f"Target Pos: {target_pos}")
            print(f"Target Rot (RPY): {target_euler}")

            #IK with Position and Rotation
            q_sol, err = robot.inverse_kinematics_optimization(target_pos, target_rot_matrix)

            if err < 1e-2:  #
                subs_dict = {q_syms[j]: q_sol[j] for j in range(5)}

                #Evaluate Jacobian
                J_num = J_simplified.subs(subs_dict).evalf()

                J_final = sym.nsimplify(J_num, tolerance=1e-4, rational=False)

                print(f"Solved Joint Angles (rad): {np.round(q_sol, 3)}")
                sym.pprint(J_final)
            else:
                print(f"Warning: IK failed to reach Pose {i + 1} within tolerance (Error: {err:.4f})")



