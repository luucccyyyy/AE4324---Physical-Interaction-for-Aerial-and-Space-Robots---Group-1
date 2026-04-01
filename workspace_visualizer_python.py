import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

T_BW = np.array([
    [-1, 0, 0, 0],
    [ 0,-1, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1]
])

T_SB = trans(0, -0.0452, 0.0165) @ np.eye(4)
T_SU = trans(0, -0.0306, 0.1025) @ roty(-1.57079)
T_UL = trans(0.11257, -0.028, 0) @ np.eye(4)
T_LW = trans(0.0052, -0.1349, 0) @ rotz(1.57079)
T_WG = trans(-0.0601, 0, 0) @ roty(-1.57079)
T_GC = trans(0.0, 0.0, 0.075)

points = []
N = 50000

for _ in range(N):
    r1 = np.random.uniform(-2, 2)
    r2 = np.random.uniform(-1.57, 1.57)
    r3 = np.random.uniform(-1.58, 1.58)
    r4 = np.random.uniform(-1.57, 1.57)
    r5 = np.random.uniform(-np.pi, np.pi)

    T1 = T_SB @ rotz(r1)
    T2 = T_SU @ rotz(r2)
    T3 = T_UL @ rotz(r3)
    T4 = T_LW @ rotz(r4)
    T5 = T_WG @ rotz(r5)

    T = T_BW @ T1 @ T2 @ T3 @ T4 @ T5 @ T_GC
    p = T[:3, 3]
    points.append(p)

points = np.array(points)

hull_3d = ConvexHull(points)
hull_xy = ConvexHull(points[:, [0, 1]])
hull_yz = ConvexHull(points[:, [1, 2]])
hull_xz = ConvexHull(points[:, [0, 2]])

def get_closed_hull_boundary(pts2d, hull):
    verts = np.append(hull.vertices, hull.vertices[0])
    return pts2d[verts, 0], pts2d[verts, 1]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#Point cloud
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           s=0.5, c='steelblue', alpha=0.1, label='Sampled points')

faces = [points[simplex] for simplex in hull_3d.simplices]
poly = Poly3DCollection(faces, alpha=0.08,
                        facecolor='royalblue', edgecolor='navy', linewidth=0.2)
ax.add_collection3d(poly)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Workspace: Point Cloud + Convex Hull Outline')
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#top view
axes[0].scatter(points[:, 0], points[:, 1],
                s=0.5, c='steelblue', alpha=0.1, label='Sampled points')
bx, by = get_closed_hull_boundary(points[:, [0, 1]], hull_xy)
axes[0].plot(bx, by, 'navy', linewidth=2, label='Convex hull')
axes[0].fill(bx, by, alpha=0.15, color='royalblue')
axes[0].set_xlabel('X [m]')
axes[0].set_ylabel('Y [m]')
axes[0].set_title('Top View (XY Projection)')
axes[0].axis('equal')
axes[0].grid(True)
axes[0].legend(markerscale=10)

#side view
axes[1].scatter(points[:, 1], points[:, 2],
                s=0.5, c='steelblue', alpha=0.1, label='Sampled points')
bx, by = get_closed_hull_boundary(points[:, [1, 2]], hull_yz)
axes[1].plot(bx, by, 'navy', linewidth=2, label='Convex hull')
axes[1].fill(bx, by, alpha=0.15, color='royalblue')
axes[1].set_xlabel('Y [m]')
axes[1].set_ylabel('Z [m]')
axes[1].set_title('Side View (YZ Projection)')
axes[1].axis('equal')
axes[1].grid(True)
axes[1].legend(markerscale=10)

#front view
axes[2].scatter(points[:, 0], points[:, 2],
                s=0.5, c='steelblue', alpha=0.1, label='Sampled points')
bx, by = get_closed_hull_boundary(points[:, [0, 2]], hull_xz)
axes[2].plot(bx, by, 'navy', linewidth=2, label='Convex hull')
axes[2].fill(bx, by, alpha=0.15, color='royalblue')
axes[2].set_xlabel('X [m]')
axes[2].set_ylabel('Z [m]')
axes[2].set_title('Front View (XZ Projection)')
axes[2].axis('equal')
axes[2].grid(True)
axes[2].legend(markerscale=10)

plt.suptitle('Workspace Projections: Point Cloud + Convex Hull Outline', fontsize=14)
plt.tight_layout()
plt.show()