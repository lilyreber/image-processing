import numpy as np
import cv2

def compute_homography_dlt(points1, points2):
    """
    Вычисляет матрицу гомографии между двумя наборами соответствующих точек.
    Returns:
        Матрица гомографии размером (3, 3).
    """

    A = []
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    A = np.array(A)

    U, L, VT = np.linalg.svd(A)
    theta = VT[-1, :]  # берем строку, а не столбец, так как V тут транспонированная   

    H = [
        [theta[0], theta[1], theta[2]],
        [theta[3], theta[4], theta[5]],
        [theta[6], theta[7], theta[8]],
    ] 
    H = H / theta[8]

    return H


def stabilize_video_with_homography(frames):
    """
    Стабилизирует видео с помощью гомографии между кадрами.
    Args:
        frames: numpy массив кадров
    Returns:
        Стабилизированные кадры видео (numpy массив).
    """
    stabilized_frames = np.zeros_like(frames)
    stabilized_frames[0] = frames[0]


    for i in range(1, len(frames)):
        image1 = frames[0].copy()
        image2 = frames[i].copy()

        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # вычисление ключевых точек с помощью Shi-Tomasi метода
        points1 = cv2.goodFeaturesToTrack(gray1, 2000, 0.01, 10)
        points1 = points1[:, 0, :]

        # с помощью оптического потока Лукаса-Канаде
        # точки points1 сопоставляются с points2
        points2, _, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None)

        # вычисление гомографии из image2 в image1
        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

        # применение гомографии для преобразования image2 в систему координат image1
        height, width = image1.shape[:2]
        warped_image = cv2.warpPerspective(image2, H, (width, height))
        stabilized_frames[i] = warped_image
    
    assert frames.shape == stabilized_frames.shape

    return stabilized_frames


def get_affine_transform(points1, points2):
    """
    Вычисляет аффинное преобразование и среднюю ошибку проекции.
    Returns:
        A - матрица аффинного преобразования размером (2, 3)
        reproj_error - средняя ошибка проекции
    """
    omega = []
    beta = []
    for i in range(len(points1)):
        x, y = points1[i]
        omega.append([x, y, 1, 0, 0, 0])
        omega.append([0, 0, 0, x, y, 1])
        beta.append(points2[i][0])
        beta.append(points2[i][1])
    omega = np.array(omega)
    beta = np.array(beta)
    theta = (np.linalg.inv(omega.T @ omega) @ omega.T) @ beta

    A = np.array([
        [theta[0], theta[1], theta[2]],
        [theta[3], theta[4], theta[5]]
    ])


    points1_homog = np.hstack([points1, np.ones((len(points1), 1))])
   
    approx_points = (A @ points1_homog.T).T  
    proj_error = np.mean(
                        np.linalg.norm(
                            points2 - approx_points, axis=1))
    
    return A, proj_error