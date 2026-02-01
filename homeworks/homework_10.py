from dataclasses import dataclass
import cv2
import numpy as np
import os 
from pathlib import Path

#DATA_DIR = root / "data"

current_dir = Path(__file__).parent
DATA_DIR = current_dir / "data"

@dataclass
class Config:
    video_file = DATA_DIR / "book.mp4"
    image_file = DATA_DIR / "book.jpg"

    detector = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # коэффициенты дисторсии (k1, k2, p1, p2, k3)
    dist_coeff = np.zeros(5)

    # внутренние параметры камеры
    K = np.array([
        [1000, 0, 320],
        [0, 1000,  240],
        [0, 0, 1.0]
    ])

    # минимальное число соответствующих точек в PnP алгоритме
    min_pnp_num = 100
    
    box_lower = np.array([
        [30, 145, 0], 
        [30, 200, 0], 
        [200, 200, 0], 
        [200, 145, 0]
    ], dtype=np.float32)

    box_upper = np.array([
        [30, 145, -50], 
        [30, 200, -50], 
        [200, 200, -50], 
        [200, 145, -50]
    ], dtype=np.float32)



def main(cfg):
    image = cv2.cvtColor(cv2.imread(cfg.image_file), cv2.COLOR_BGR2GRAY)
    video = cv2.VideoCapture(cfg.video_file)

    keypoints, descriptors = cfg.detector.detectAndCompute(image, None)    

    while True:
        ok, image = video.read()

        if not ok:
            break

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        frame_keypoints, frame_descriptors = cfg.detector.detectAndCompute(frame, None)

        matches = cfg.matcher.match(descriptors, frame_descriptors)
        pts = np.array([keypoints[m.queryIdx].pt for m in matches])
        frame_pts = np.array([frame_keypoints[m.trainIdx].pt for m in matches])
        
        _, mask = cv2.findHomography(pts, frame_pts, method=cv2.RANSAC)
        matches = [m for m, inliner in zip(matches, mask.ravel()) if inliner]

        pts = np.array([keypoints[m.queryIdx].pt for m in matches])
        frame_pts = np.array([frame_keypoints[m.trainIdx].pt for m in matches])

        object_points = np.hstack([
            pts, 
            np.zeros((len(pts), 1))
        ])


        ok, rvec, tvec = cv2.solvePnP(
            object_points, 
            frame_pts, 
            cfg.K, 
            cfg.dist_coeff
        )

        if not ok:
            continue

        image_box_lower, _ = cv2.projectPoints(cfg.box_lower, rvec, tvec, cfg.K, cfg.dist_coeff)
        image_box_upper, _ = cv2.projectPoints(cfg.box_upper,rvec, tvec, cfg.K, cfg.dist_coeff)

        show_image = image.copy()

        cv2.polylines(show_image, [np.int32(image_box_lower)], True, (255, 0, 0), 2)
        cv2.polylines(show_image, [np.int32(image_box_upper)], True, (0, 0, 255), 2)

        pts_lower = image_box_lower[:, 0].astype(int)
        pts_upper = image_box_upper[:, 0].astype(int)

        for (x1, y1), (x2, y2) in zip(pts_lower, pts_upper):
            cv2.line(
                show_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

        cv2.imshow("Book PnP", show_image)

        key = cv2.waitKey(10)

        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)



if __name__ == "__main__":
    cfg = Config()

    main(cfg)
