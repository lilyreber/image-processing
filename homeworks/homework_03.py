import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class Filter:
    def __init__(self, cfg):
        self.mask_H = None
        self.mask_W = None

        self.image = cv2.cvtColor(cv2.imread(cfg.input_image_path), cv2.COLOR_BGR2GRAY)
        self.sigma_space = cfg.sigma_space
        self.sigma_color = cfg.sigma_color

        self.image_H, self.image_W = self.image.shape

    def compare(self, my_image, cv2_image):
        diff = cv2.absdiff(my_image, cv2_image)

        metrics = {
            "MSE": np.mean(diff**2),
            "MAE": np.mean(diff),
            "Max Difference": np.max(diff),
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics

    def bilateral_filter(self, src_image, sigma_space, sigma_color):
        k = 3 * np.ceil(sigma_space)

        src_image = src_image.astype(np.float32)

        gaussian_kernel = np.exp(-(np.arange(-k, k + 1) ** 2) / (2 * sigma_space**2))
        gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel)

        dst_image = np.zeros((self.image_H, self.image_W))

        src_image = np.pad(src_image, ((k, k), (k, k)), mode="symmetric")

        for x in range(k, self.image_W + k):
            for y in range(k, self.image_H + k):
                func_x_s_y_t = src_image[y - k : y + k + 1, x - k : x + k + 1]
                func_x_y = src_image[y, x]
                kernel_non_linear = np.exp(
                    -((func_x_s_y_t - func_x_y) ** 2) / (2 * sigma_color**2)
                )

                bilateral_kernel = gaussian_kernel_2d * kernel_non_linear

                dst_image[y - k, x - k] = np.sum(
                    bilateral_kernel * func_x_s_y_t
                ) / np.sum(bilateral_kernel)

        return np.clip(dst_image, 0, 255).astype(np.uint8)

    def main(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        start_time = time.time()
        transformed_image = self.bilateral_filter(
            self.image, self.sigma_space, self.sigma_color
        )
        my_runtime = time.time() - start_time
        cv2.imwrite(cfg.output_image_path, transformed_image)

        start_time = time.time()
        cv2_transformed_image = cv2.bilateralFilter(
            self.image,
            2 * (3 * np.ceil(self.sigma_space)) + 1,
            self.sigma_color,
            self.sigma_space,
        )
        cv2_runtime = time.time() - start_time

        cv2.imshow("Image", self.image)
        cv2.imshow("My bilateral filter", transformed_image)
        cv2.imshow("OpenCV bilateral filter", cv2_transformed_image)

        self.compare(transformed_image, cv2_transformed_image)

        print("Время выполнения моей реализации:", my_runtime)
        print("Время выполнения реализации из OpenCV:", cv2_runtime)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    filter = Filter(cfg)
    filter.main()
