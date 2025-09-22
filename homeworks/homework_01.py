from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class ImageTransformer:
    def __init__(self, cfg):
        self.points = []
        self.points_radius = cfg.radius
        self.points_color = cfg.color
        self.transform_size = cfg.transform_size
        self.cam_frame = cfg.cam_frame
        self.fps = cfg.fps

        self.hue_range = cfg.hue_range
        self.saturation_range = cfg.saturation_range
        self.contrast_range = cfg.contrast_range
        self.brightness_range = cfg.brightness_range

        self.hue_change_param = 0
        self.saturation_change_param = 1.0
        self.contrast_change_param = 1.0
        self.brightness_change_param = 1.0

    def on_hue_change(self, val):
        p_min, p_max = self.hue_range
        self.hue_change_param = (p_max - p_min) * (val / 100) + p_min

    def on_saturation_change(self, val):
        p_min, p_max = self.saturation_range
        self.saturation_change_param = p_max - (p_max - p_min) * (val / 100) + p_min

    def on_contrast_change(self, val):
        p_min, p_max = self.contrast_range
        self.contrast_change_param = p_max - (p_max - p_min) * (val / 100) + p_min

    def on_brightness_change(self, val):
        p_min, p_max = self.brightness_range
        self.brightness_change_param = p_max - (p_max - p_min) * (val / 100) + p_min

    def points_preprocessing(self):
        """
        In my implementation, the points specified in the installation order
        are transferred to the points of the region
        starting from the upper left corner clockwise
        """
        return np.array(self.points).astype(np.float32)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
            else:
                self.points = []

    def draw_points(self, frame):
        for p in self.points:
            cv2.circle(frame, p, self.points_radius, self.points_color, -1)

    def bh_transform(self, frame):
        """bilinear homography transform"""
        quad_before = self.points_preprocessing()
        quad_after = np.array(
            [
                [0, 0],
                [self.transform_size[0] - 1, 0],
                [self.transform_size[0] - 1, self.transform_size[1] - 1],
                [0, self.transform_size[1] - 1],
            ],
            dtype=np.float32,
        )

        H = cv2.getPerspectiveTransform(quad_before, quad_after)

        dst_img = cv2.warpPerspective(frame, H, self.transform_size, cv2.INTER_LINEAR)

        return dst_img

    def cj_augmentation(self, frame):
        """ColorJitter augmentation"""
        frame = self.adjust_hue(frame, self.hue_change_param)
        frame = self.adjust_contrast(frame, self.contrast_change_param)
        frame = self.adjust_saturation(frame, self.saturation_change_param)
        frame = self.adjust_brightness(frame, self.brightness_change_param)
        return frame

    def create_trackbars(self):
        cv2.createTrackbar("Hue", "Image Transform", 0, 100, self.on_hue_change)
        cv2.createTrackbar(
            "Saturation", "Image Transform", 0, 100, self.on_saturation_change
        )
        cv2.createTrackbar(
            "Contrast", "Image Transform", 0, 100, self.on_contrast_change
        )
        cv2.createTrackbar(
            "Brightness", "Image Transform", 0, 100, self.on_brightness_change
        )

    def main(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_frame[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_frame[1])
        cam.set(cv2.CAP_PROP_FPS, self.fps)
        cv2.namedWindow("Image Transform")
        cv2.setMouseCallback("Image Transform", self.mouse_callback)
        self.create_trackbars()

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            display_frame = frame.copy()

            self.draw_points(display_frame)

            cv2.imshow("Image Transform", display_frame)

            if len(self.points) == 4:
                transformed = self.cj_augmentation(self.bh_transform(frame))
                cv2.imshow("Transformed", transformed)
            elif len(self.points) == 0:
                cv2.destroyWindow("Transformed")

            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()

    def adjust_brightness(self, frame, delta):
        dst_float = delta * frame.astype(np.float32)
        dst_img = np.clip(dst_float, 0, 255).astype(np.uint8)
        return dst_img

    def adjust_contrast(self, frame, gamma):
        Y = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mu = Y.mean()
        dst_float = gamma * frame + (1 - gamma) * mu
        dst_img = dst_float.astype(np.uint8)
        return dst_img

    def adjust_saturation(self, frame, beta):
        B, G, R = cv2.split(frame)

        Y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        R = beta * R + (1 - beta) * Y
        G = beta * G + (1 - beta) * Y
        B = beta * B + (1 - beta) * Y

        dst_float = cv2.merge([B, G, R])

        dst_img = dst_float.astype(np.uint8)

        return dst_img

    def adjust_hue(self, frame, alpha):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.remainder(hsv[:, :, 0] + alpha, 180)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    task = Path(__file__).stem

    cfg = OmegaConf.load("../params.yaml")[task]

    bht = ImageTransformer(cfg)
    bht.main()
