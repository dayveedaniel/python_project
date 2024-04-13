import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import pyopencl.array
import cv2


def processing_image(image, compressed):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    image_gpu = cl.array.to_device(queue, image)
    compressed_gpu = cl.array.to_device(queue, compressed)

    kernel_code = """
    __kernel void subtract_images(__global const uchar* image, __global const 
uchar* compressed,
                                   __global uchar* result, int image_width, 
int image_height,
                                   int compressed_width, int 
compressed_height)
    {
        int x = get_global_id(1);
        int y = get_global_id(0);

        if (x < image_width && y < image_height)
        {
            int compressed_x = x / 4;
            int compressed_y = y / 4;

            for (int c = 0; c < 3; ++c)
            {
                uchar image_pixel = image[(y * image_width + x) * 3 + c]; 
                uchar compressed_pixel = compressed[(compressed_y * 
compressed_width + compressed_x) * 3 + c];

                result[(y * image_width + x) * 3 + c] = image_pixel - 
compressed_pixel;
            }
        }
    }
    """

    program = cl.Program(context, kernel_code).build()
    result_gpu = cl.array.empty_like(image_gpu)
    threads_per_block = (16, 16)
    program.subtract_images(queue, image.shape[:2], threads_per_block, image_gpu.data, compressed_gpu.data,
                            result_gpu.data, np.int32(image.shape[1]), np.int32(image.shape[0]),
                            np.int32(compressed.shape[1]), np.int32(compressed.shape[0]))
    result = image_gpu.get()
    return result


sizes = [1024, 10240, 12800, 20480]
images = {size: [cv2.imread(f'/kaggle/input/parallel/{size}.jpg')] for size
          in sizes}
for size in sizes:
    images[size].append(cv2.resize(images[size][0], list(map(lambda x: x // 4,
                                                             images[size][0].shape[1::-1]))))
compressed = cv2.resize(images[1024][0], list(map(lambda x: x // 4,
                                                  images[1024][0].shape[1::-1])))
plt.imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

result = processing_image(*images[1024])
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# %%timeit -n 3 -r 1
# result = processing_image(*images[10240])
#
# %%timeit -n 3 -r 1
# result = processing_image(*images[12800])
#
# %%timeit -n 3 -r 1
# result = processing_image(*images[20480])