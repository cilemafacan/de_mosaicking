import numpy as np
def mse(image1, image2):
    for c in range(3):
        for i in range(3):
            for j in range(3):
                squared_diff = (image1[i, j, c] - image2[i, j, c]) ** 2
                mse = np.sum(squared_diff) / (image1.shape[0] * image1.shape[1] * image1.shape[2])
    return mse

def psnr(image1, image2):
    mse_val = mse(image1, image2)
    psnr = 10 * np.log10(255**2 / mse_val)
    return psnr

def mae(image1, image2):
    for c in range(3):
        for i in range(3):
            for j in range(3):
                abs_diff = np.abs(image1[i, j, c] - image2[i, j, c])
                mae = np.sum(abs_diff) / (image1.shape[0] * image1.shape[1] * image1.shape[2])

    return mae