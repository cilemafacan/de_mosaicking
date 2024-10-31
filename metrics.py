import numpy as np
def mse(image1, image2):
    total = 0
    for c in range(3):
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                diff = image1[i, j, c] - image2[i, j, c]
                total += diff ** 2
    mse = total / (image1.shape[0] * image1.shape[1] * image1.shape[2])
    return mse

def psnr(image1, image2):
    mse_val = mse(image1, image2)
    psnr = 10 * np.log10(255**2 / mse_val)
    return psnr

def mae(image1, image2):
    total = 0
    for c in range(3):
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                diff = image1[i, j, c] - image2[i, j, c]
                total += abs(diff)
    mae = total / (image1.shape[0] * image1.shape[1] * image1.shape[2])
    return mae