import random
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class Filter:
    def __init__(self, cfg):
        self.image = cv2.cvtColor(cv2.imread(cfg.input_image), cv2.COLOR_BGR2GRAY)
        self.sigma = cfg.sigma
        self.thresh_percent = cfg.thresh_percent
        self.image_H, self.image_W = self.image.shape
        self.max_trials = cfg.max_trials
        self.seed = cfg.seed
        self.threshold = cfg.threshold
        self.output_image = cfg.output_image
        self.output_edges = cfg.output_edges

    def calc_marr_hildreth_edges(self, log_img, T):
        H, W = log_img.shape

        pad_image = np.zeros((H + 2, W + 2))
        pad_image[1 : H + 1, 1 : W + 1] = log_img

        edges = np.zeros_like(log_img, dtype=np.uint8)

        for y in range(1, H + 1):
            for x in range(1, W + 1):
                window = pad_image[y - 1 : y + 2, x - 1 : x + 2]

                (a, b, c, d, _, e, f, g, h) = window.flatten()

                ok = False

                if d * e < 0 and np.abs(d - e) > T:
                    ok = True
                elif b * g < 0 and np.abs(b - g) > T:
                    ok = True
                elif a * h < 0 and np.abs(a - h) > T:
                    ok = True
                elif f * c < 0 and np.abs(f - c) > T:
                    ok = True

                if ok:
                    edges[y - 1, x - 1] = 255

        return edges

    def marr_hildreth_detection(self):
        n = 2 * np.ceil(3 * self.sigma).astype(int) + 1
        u = cv2.getGaussianKernel(n, sigma=self.sigma)
        G = u @ u.T

        g = cv2.filter2D(self.image.astype(float), ddepth=-1, kernel=G)

        L = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

        log_img = cv2.filter2D(g, ddepth=-1, kernel=L)

        threshold = self.thresh_percent * log_img.max()

        marhil_edges = self.calc_marr_hildreth_edges(log_img, threshold)

        return marhil_edges

    def ransac(self, points, estimation=True):
        random.seed(self.seed)
        max_inliers = 0
        p1_best, p2_best = None, None
        fact_inliers = []

        for _ in range(self.max_trials):
            indices = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[indices[0]], points[indices[1]]

            norm_constant = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)

            distances = (
                np.abs(
                    (p2[1] - p1[1]) * points[:, 0]
                    - (p2[0] - p1[0]) * points[:, 1]
                    + p2[0] * p1[1]
                    - p2[1] * p1[0]
                )
                / norm_constant
            )
            inliers = points[distances < self.threshold]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                p1_best, p2_best = p1, p2
                fact_inliers = inliers

        if estimation:
            p1_best, p2_best = self.ransac(fact_inliers, False)

        return p1_best, p2_best

    def main(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        mh_image = self.marr_hildreth_detection()

        edge_points = np.column_stack(np.where(mh_image > 0))

        p1, p2 = self.ransac(edge_points.copy())
        p1 = [p1[1], p1[0]]
        p2 = [p2[1], p2[0]]

        p1, p2 = self.get_end_points(p1, p2)

        mh_color_image = cv2.cvtColor(mh_image, cv2.COLOR_GRAY2BGR)
        ransac_color_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        cv2.line(mh_color_image, p1, p2, [148, 24, 248], 3, cv2.LINE_AA)
        cv2.line(ransac_color_image, p1, p2, [148, 24, 248], 3, cv2.LINE_AA)

        mh_mask_bgr = cv2.cvtColor(mh_image, cv2.COLOR_GRAY2BGR)
        masked_ransac_image = cv2.bitwise_and(ransac_color_image, mh_mask_bgr)

        cv2.imshow("Marr Hildreth Edges", masked_ransac_image)
        cv2.imwrite(self.output_edges, masked_ransac_image)
        cv2.imshow("Line RANSAC", ransac_color_image)
        cv2.imwrite(self.output_image, ransac_color_image)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()

    def get_end_points(self, p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        norm_constant = np.sqrt(dx**2 + dy**2)
        tx = dx / norm_constant
        ty = dy / norm_constant

        nx = -ty
        ny = tx

        d = p1[0] * nx + p1[1] * ny

        x0, y0 = nx * d, ny * d
        s = max(self.image_W, self.image_H) * 2

        xy1 = int(x0 + s * tx), int(y0 + s * ty)
        xy2 = int(x0 - s * tx), int(y0 - s * ty)

        return xy1, xy2


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    filter = Filter(cfg)
    filter.main()
