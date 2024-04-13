from numba import cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt


@cuda.jit
def shift_kernel(input_image, output_image, shift_x, shift_y):
    y, x = cuda.grid(2)
    empty_pixels = (73, 38, 187)
    if x < input_image.shape[1] and y < input_image.shape[0]:
        new_x = x - shift_x
        new_y = y - shift_y
        if 0 <= new_x < input_image.shape[1] and 0 <= new_y < input_image.shape[0]:
            for c in range(input_image.shape[2]):
                output_image[y, x, c] = input_image[new_y, new_x, c]
        else:
            for c in range(input_image.shape[2]):
                output_image[y, x, c] = empty_pixels[c]


def shift_image(input_image):
    threadsperblock = (16, 16)
    blockspergrid_y = (input_image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_x = (input_image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    d_input_image = cuda.to_device(input_image.astype(np.float32))
    d_output_image = cuda.device_array(input_image.shape, dtype=np.float32)

    shift_x = 200
    shift_y = 80
    shift_kernel[blockspergrid, threadsperblock](d_input_image,
                                                 d_output_image, shift_x, shift_y)
    output_image = d_output_image.copy_to_host().astype(np.uint8)
    return output_image


@cuda.jit
def blur_kernel(input_image, output_image):
    y, x = cuda.grid(2)  # Поменяли местами x и y
    if x < input_image.shape[1] and y < input_image.shape[0]:
        for c in range(input_image.shape[2]):
            if 0 < x < input_image.shape[1] - 1 and 0 < y < input_image.shape[0] - 1:
                output_image[y, x, c] = (
                                                input_image[y - 1, x - 1, c] + input_image[y - 1, x, c] +
                                                input_image[y - 1, x + 1, c] +
                                                input_image[y, x - 1, c] + input_image[y, x, c] +
                                                input_image[y, x + 1, c] +
                                                input_image[y + 1, x - 1, c] + input_image[y + 1, x, c] +
                                                input_image[y + 1, x + 1, c]) / 9
            else:
                output_image[y, x, c] = input_image[y, x, c]


def blur_image(input_image):
    threadsperblock = (16, 16)
    blockspergrid_y = (input_image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_x = (input_image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    d_input_image = cuda.to_device(input_image.astype(np.float32))
    d_output_image = cuda.device_array(input_image.shape, dtype=np.float32)

    blur_kernel[blockspergrid, threadsperblock](d_input_image,
                                                d_output_image)
    output_image = d_output_image.copy_to_host().astype(np.uint8)
    return output_image


def processing_image(input_image):
    output_image_shifted = shift_image(input_image)
    output_image_blurred = blur_image(output_image_shifted)
    return output_image_blurred


def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


images = {size: cv2.imread(f'/kaggle/input/parallel/{size}.jpg') for size in [1024, 10240, 12800, 20480]}

# %%timeit -n 3 -r 1
# result = processing_image(images[10240])
#
# %%timeit -n 3 -r 1
# result = processing_image(images[12800])
#
# %%timeit -n 3 -r 1
# result = processing_image(images[20480])
#
# result = processing_image(images[1024])
# show_image(result)
