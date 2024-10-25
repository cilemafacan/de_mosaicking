import numpy as np
from PIL import Image

def padding(image: np.ndarray) -> np.ndarray:
    padded_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2, image.shape[2]), dtype=image.dtype)
    padded_image[1:-1, 1:-1, :] = image
    padded_image[0, 1:-1] = image[1, :]
    padded_image[-1, 1:-1] = image[-2, :]
    padded_image[1:-1, 0] = image[:, 1]
    padded_image[1:-1, -1] = image[:, -2]
    padded_image[0, 0] = image[1, 1]
    padded_image[0, -1] = image[1, -2]
    padded_image[-1, 0] = image[-2, 1]
    padded_image[-1, -1] = image[-2, -2]

    return padded_image

def check_input(image: Image):
    if image.mode != 'RGB':
        raise ValueError('Input image must be RGB')

def mosaicking(input_image, bayer):
    output_image = np.zeros_like(input_image)
    for i in range(1, input_image.shape[0] - 1, 2):
        for j in range(1, input_image.shape[1] - 1, 2):
            output_image[i:i+2, j:j+2] = input_image[i:i+2, j:j+2] * bayer
    return output_image

def de_mosaicking(input_image):
    output_image = input_image.copy()
    for i in range(0, input_image.shape[0]-2):
        for j in range(0, input_image.shape[1]-2):
            if input_image[i+1, j+1, 0] == 0:
                red_channel = input_image[i:i+3, j:j+3,0]
                if np.count_nonzero(red_channel) == 0:
                    red_channel_avg = 0
                else:
                    red_channel_avg = np.sum(red_channel) / np.count_nonzero(red_channel)
                output_image[i+1, j+1, 0] = red_channel_avg
            
            if input_image[i+1, j+1, 1] == 0:
                green_channel = input_image[i:i+3, j:j+3,1]
                if np.count_nonzero(green_channel) == 0:
                    green_channel_avg = 0
                else:
                    green_channel_avg = np.sum(green_channel) / np.count_nonzero(green_channel)
                output_image[i+1, j+1, 1] = green_channel_avg

            if input_image[i+1, j+1, 2] == 0:
                blue_channel = input_image[i:i+3, j:j+3,2]
                if np.count_nonzero(blue_channel) == 0:
                    blue_channel_avg = 0
                else:
                    blue_channel_avg = np.sum(blue_channel) / np.count_nonzero(blue_channel)
                output_image[i+1, j+1, 2] = blue_channel_avg

    output_image = output_image[1:-1, 1:-1]

    return output_image

