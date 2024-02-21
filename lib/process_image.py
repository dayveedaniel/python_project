import sys

import cv2
import numpy as np


def process_image(image_path, threshold_value, erosion_steps):
    # Загрузка изображения
    original_image = cv2.imread(image_path)

    # Получение интенсивности
    intensity_values = np.mean(original_image, axis=2)
    print(intensity_values)

    # Установка порога
    thresholded_image = np.where(intensity_values > threshold_value, 1, 0)

    # Эрозия
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(thresholded_image.astype(np.uint8), kernel, iterations=erosion_steps)

    # Преобразование в цветное изображение
    result_image = np.zeros_like(original_image)
    result_image[eroded_image == 1] = [255, 255, 255]  # Белый цвет
    result_image[eroded_image == 0] = [0, 0, 0]  # Черный цвет

    return result_image


def main():
    image_path = "/Users/dayveed/Downloads/Dashatars.png"  # Укажите путь к вашему изображению
    threshold_value = 150
    erosion_steps = 2

    result_image = process_image(image_path, threshold_value, erosion_steps)

    # Сохранение результата
    cv2.imwrite("/Users/dayveed/Downloads/result.png", result_image)


if __name__ == "__main__":
    main()
