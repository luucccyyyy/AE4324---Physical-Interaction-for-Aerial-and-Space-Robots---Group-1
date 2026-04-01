import numpy as np 
from robot_arm import Edubot
from time import time
from robot_arm import euler_to_rot
#This is the answer to part 2.1 for the AE 4324 Assignment
robot = Edubot()


poses = [
    ([0.2, 0.2, 0.2], [0.000, 1.570, 0.650]),
    ([0.2, 0.1, 0.4], [0.000, 0.000, -1.570]),
    ([0.0, 0.0, 0.4], [0.000, -0.785, 1.570]),
    ([0.0, 0.0, 0.07], [3.141, 0.000, 0.000]),
    ([0.0, 0.0452, 0.45], [-0.785, 0.000, 3.141])
]

initial_guesses = [
    [0, 0, 0, 0, 0],
    [0, 0.5, -0.5, 0, 0],
    [-1, 0.5, 0.5, -0.5, 0],
    [1, -0.5, 0.5, 0.5, 0]
]
#Part 1: Using the optimization algorithm
t_start = time()
for i, (pos, rot) in enumerate(poses):


    R = euler_to_rot(*rot)
    q, error = robot.inverse_kinematics_optimization(pos, R)

    print(f"\nPose {i+1}")
    print("q:", q)
    print("error:", error)

    solutions = []

    for guess in initial_guesses:
        q, error = robot.inverse_kinematics_optimization(pos, R, initial_guess=guess)
        if error < 0.02:  # feasible
            solutions.append(q)

    print(f"\nPose {i + 1}: {len(solutions)} feasible solutions found")
    for idx, q_sol in enumerate(solutions):
        print(f"Solution {idx + 1}: {q_sol}")

t_end = time()

print("Time Elapsed For Optimization: ", t_end - t_start)
