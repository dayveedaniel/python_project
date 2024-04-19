import multiprocessing as mp
import time

import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.inf)

sizes = [1024, 1280, 1920]
images = {size: np.array(Image.open(f"/Users/dayveed/Downloads/Dashatars copy.png")) for size in sizes}


def processing_row(args):
    y, image, x_shift, y_shift = args
    result = np.zeros(shape=(image.shape[1], 3), dtype="uint8")
    for x in range(image.shape[1]):
        pixel_count = 0
        colors = np.array([0, 0, 0], dtype="uint16")
        for delta_x in range(-1, 2):
            for delta_y in range(-1, 2):
                if 0 <= x + delta_x < image.shape[1] and 0 <= y + delta_y < image.shape[0]:
                    pixel_count += 1
                    if 0 <= x + delta_x - x_shift < image.shape[1] and 0 <= y + delta_y - y_shift < image.shape[0]:
                        colors += image[y + delta_y - y_shift][x + delta_x -
                                                               x_shift]
                    else:
                        colors += np.array([187, 38, 73], dtype="uint8")
        colors //= pixel_count
        result[x] = colors.astype("uint8")
    return result


def processing_image(image, n_threads):
    x_shift = 200
    y_shift = 80
    with mp.Pool(n_threads) as p:
        results = p.map(processing_row, [(i, image, x_shift, y_shift) for i in
                                         range(image.shape[0])])
        results = np.array(results)
        image_result = Image.fromarray(results, 'RGB')
        image_result.save('result_lab2.jpg')


if __name__ == "__main__":
    test_threads = list(range(2, 17, 2))
    for size in sizes:
        for n_threads in test_threads:
            n = 3
            total_time = 0
            for i in range(n):
                image = images[size]
                start = time.perf_counter()
                processing_image(image, n_threads)
                finish = time.perf_counter()
                total_time += finish - start
            total_time /= n
            print(f"File: {size} thread: {n_threads} time: {round(total_time, 3)}s")
