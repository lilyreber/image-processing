import numpy as np


def rotvec_from_rotation_matrix(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    omega = (theta / (2 * np.sin(theta))) * np.array(
        [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    )
    return omega


def solvePnP_DLT(points3D, points2D, K):
    A = []
    for i in range(len(points3D)):
        u, v, w = points3D[i]
        x, y, _ = np.linalg.inv(K) @ np.hstack([points2D[i], [1]])
        A.append([u, v, w, 1, 0, 0, 0, 0, -x * u, -x * v, -x * w, -x])
        A.append([0, 0, 0, 0, u, v, w, 1, -y * u, -y * v, -y * w, -y])
    A = np.array(A)

    U, L, VT = np.linalg.svd(A)
    theta = VT[-1, :]  # берем строку, а не столбец, так как V тут транспонированная
    R = [
        [theta[0], theta[1], theta[2]],
        [theta[4], theta[5], theta[6]],
        [theta[8], theta[9], theta[10]],
    ]

    t = [theta[3], theta[7], theta[11]]

    U, L, VT = np.linalg.svd(R)
    R = U @ VT
    t = t / L[0]

    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    omega = rotvec_from_rotation_matrix(R)
    # возвращаем оценки для вектора вращения omega и трансляции t
    return omega, t


def rotation_matrix_from_rotvec(omega):
    s = len(omega)

    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        return np.eye(3)

    n = omega / theta

    x, y, z = n
    n_x = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.eye(s) + np.sin(theta) * n_x + (1 - np.cos(theta)) * (n_x @ n_x)
    return R


def triangulate_DLT(rotation_vectors, translations, camera_points2D, K):
    w = []
    for p in range(len(camera_points2D[0])):
        A = []
        b = []
        for i in range(len(camera_points2D)):
            r = rotation_matrix_from_rotvec(rotation_vectors[i])
            t = translations[i]
            x, y, _ = np.linalg.inv(K) @ np.hstack([camera_points2D[i][p], [1]])
            A.append(
                [r[2, 0] * x - r[0, 0], r[2, 1] * x - r[0, 1], r[2, 2] * x - r[0, 2]]
            )
            A.append(
                [r[2, 0] * y - r[1, 0], r[2, 1] * y - r[1, 1], r[2, 2] * y - r[1, 2]]
            )
            b.append(t[0] - t[2] * x)
            b.append(t[1] - t[2] * y)

        A = np.array(A)
        b = np.array(b)

        w.append(np.linalg.inv(A.T @ A) @ A.T @ b)

    # возвращаем оценки для 3D-точек сцены
    return w
