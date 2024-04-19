import multiprocessing as mp
import time

import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.inf)

sizes = [1024, 1280, 1920]
images = {size: np.array(Image.open(f"/Users/dayveed/Downloads/Dashatars copy.png")) for size in sizes}


def processing_row(args):
    y, image = args
    result = np.zeros(shape=(image.shape[1], 3), dtype="uint8")
    for x in range(image.shape[1]):
        central_coeff = 1
        colors = np.array([0, 0, 0], dtype="int16")
        near_pixels = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        for delta_x, delta_y in near_pixels:
            if 0 <= x + delta_x < image.shape[1] and 0 <= y + delta_y < image.shape[0]:
                colors -= 256 - image[y + delta_y][x + delta_x]
                central_coeff += 1
        colors += central_coeff * (256 - image[y][x])
        for i in range(len(colors)):
            if colors[i] < 0:
                colors[i] = 0
            if colors[i] > 255:
                colors[i] = 255
        result[x] = colors.astype("uint8")
    return result


def processing_image(image, n_threads, output):
    with mp.Pool(n_threads) as p:
        results = p.map(processing_row, [(i, image) for i in range(image.shape[0])])
        results = np.array(results)
        image_result = Image.fromarray(results, 'RGB')
        image_result.save(output)


if __name__ == "__main__":
    test_threads = list(range(2, 17, 2))
    for size in sizes:
        for n_threads in test_threads:
            n = 3
            total_time = 0
            for i in range(n):
                image = images[size]
                start = time.perf_counter()
                processing_image(image, n_threads, output=f'result_lab2b{size}.png')
                finish = time.perf_counter()
                total_time += finish - start
            total_time /= n
            print(f"File: {size} Threads: {n_threads} Time: {round(total_time, 3)}s")
