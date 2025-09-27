from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class Filter:
    def __init__(self, cfg):
        self.mask_H = None
        self.mask_W = None

        self.image = cv2.cvtColor(cv2.imread(cfg.image_path), cv2.COLOR_BGR2GRAY)

        self.image_H, self.image_W = self.image.shape

    def apply_filter(self, image):
        F = np.fft.fft2(image)
        F = np.fft.fftshift(F)
        abs_F, angle_F = np.abs(F), np.angle(F)

        M = np.ones(image.shape, dtype=np.complex128)

        h, w = image.shape
        center_w = h // 2
        start_v = max(0, center_w - self.mask_W)
        end_v = min(w, center_w + self.mask_W)

        M[0 : self.mask_H, start_v:end_v] = 0

        M[h - self.mask_H : h, start_v:end_v] = 0

        mask_on_F = abs_F * M * np.exp(1j * angle_F)

        mask_on_abs_F = abs_F * M

        g = np.fft.ifft2(np.fft.ifftshift(mask_on_F))
        transformed_image = np.real(g)
        transformed_image_normalized = cv2.normalize(
            transformed_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        spectrum = cv2.normalize(
            np.log(1 + np.real(mask_on_abs_F)), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        return transformed_image_normalized, spectrum

    def on_height_change(self, val):
        self.mask_H = max(1, int((val / 100) * (self.image_H // 2)))

    def on_width_change(self, val):
        self.mask_W = max(1, int((val / 100) * (self.image_W // 2)))

    def create_trackbars(self):
        cv2.createTrackbar("Height", "Filter", 0, 100, self.on_height_change)
        cv2.createTrackbar("Width", "Filter", 0, 100, self.on_width_change)

    def update(self):
        transformed_image, mask_on_abs_F = self.apply_filter(self.image)
        cv2.imshow("Image", transformed_image)
        cv2.imshow("Filter", mask_on_abs_F)

    def main(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)

        self.create_trackbars()
        self.update()

        while True:
            self.update()
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    filter = Filter(cfg)
    filter.main()
