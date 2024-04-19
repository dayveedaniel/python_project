import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def shift_image(input_image, shift_x, shift_y):
    input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
    output_tensor = torch.zeros_like(input_tensor)

    input_tensor = input_tensor.to(device='mps')
    output_tensor = output_tensor.to(device='mps')

    _, _, h, w = input_tensor.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid_y = grid_y.to(device='mps')
    grid_x = grid_x.to(device='mps')

    new_grid_y = grid_y - shift_y
    new_grid_x = grid_x - shift_x

    mask_y = (new_grid_y >= 0) & (new_grid_y < h)
    mask_x = (new_grid_x >= 0) & (new_grid_x < w)
    mask = mask_y & mask_x

    output_tensor[0, :, mask] = input_tensor[0, :, new_grid_y[mask], new_grid_x[mask]]
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return output_image


def blur_image(input_image):
    input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()

    input_tensor = input_tensor.to(device='mps')

    kernel = torch.tensor([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=torch.float32).to(device='mps') / 9

    kernel = kernel.view(1, 1, 3, 3).repeat(input_tensor.shape[1], 1, 1, 1)

    output_tensor = torch.nn.functional.conv2d(input_tensor, kernel, padding=1, groups=input_tensor.shape[1])

    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return output_image


def processing_image(input_image):
    shift_x = 200
    shift_y = 80
    output_image_shifted = shift_image(input_image, shift_x, shift_y)
    output_image_blurred = blur_image(output_image_shifted)
    return output_image_blurred


def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


images = {size: cv2.imread(f'/Users/dayveed/Downloads/Dashatars copy.png') for size in [1024, 10240, 12800, 20480]}

if __name__ == "__main__":
    result = processing_image(images[1024])
    show_image(result)
