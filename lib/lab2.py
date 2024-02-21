import sys
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time


def process_image(image_path, threshold_value, erosion_steps):
    original_image = cv2.imread(image_path)
    intensity_values = np.mean(original_image, axis=2)
    thresholded_image = np.where(intensity_values > threshold_value, 1, 0)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(thresholded_image.astype(np.uint8), kernel, iterations=erosion_steps)
    result_image = np.zeros_like(original_image)
    result_image[eroded_image == 1] = [255, 255, 255]  # White color
    result_image[eroded_image == 0] = [0, 0, 0]  # Black color
    return result_image


def process_image_multiprocessing(image_path, threshold_value, erosion_steps):
    with ProcessPoolExecutor() as executor:
        future = executor.submit(process_image, image_path, threshold_value, erosion_steps)
        result_image = future.result()
    return result_image


def measure_performance(image_path, threshold_value, erosion_steps, num_processes):
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future = executor.submit(process_image, image_path, threshold_value, erosion_steps)
        result_image = future.result()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Performance with {num_processes} processes: {execution_time:.4f} seconds")
    cv2.imwrite(f"/Users/dayveed/Downloads/result{num_processes}.png", result_image)


def main():
    image_path = "/Users/dayveed/Downloads/Dashatars.png"
    threshold_value = 150
    erosion_steps = 2

    for num_processes in [2, 4, 6, 8, 10, 12, 14, 16]:
        measure_performance(image_path, threshold_value, erosion_steps, num_processes)


if __name__ == "__main__":
    main()
