import numpy as np


def transform_points(points1, R, t):
    points2 = (R.T @ (points1 - t).T).T
    return points2


def rotation_matrix_from_rotvec(omega):
    s = len(omega)

    theta = np.linalg.norm(omega)
    n = omega / theta

    x, y, z = n
    n_x = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.eye(s) + np.sin(theta) * n_x + (1 - np.cos(theta)) * (n_x @ n_x)
    return R


def rotvec_from_rotation_matrix(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    omega = (theta / (2 * np.sin(theta))) * np.array(
        [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    )
    return omega


def project_points(points3D, P):
    n = len(points3D)
    points2D = np.hstack((points3D, np.ones((n, 1)))) @ P.T
    lambd = points2D[:, 2].reshape(-1, 1)
    points2 = points2D[:, :2] / lambd
    return points2


def from_image_coordinates_to_world(x, y, c, P):
    P_plus = P.T @ np.linalg.inv(P @ P.T)
    point3D = P_plus @ np.array([x, y, 1])
    point3D = point3D / point3D[3]
    t = -point3D[1] / c[1]
    point3D = point3D + t * np.hstack([c, 1])
    return point3D
