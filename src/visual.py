import cv2
import numpy as np
import os

img_path = '/data3/YG/FRINet/code/data/COD10K/test/image/'
mask_path = '/data3/YG/FRINet/code/pred/COD10K/2-250/'
save_path = '/data3/YG/FRINet/code/visual/COD10K/2-250/'

img_list = os.listdir(img_path)
img_list.sort()

mask_list = os.listdir(mask_path)
mask_list.sort()

num = len(img_list)

for i in range(num):
    original_image = cv2.imread(os.path.join(img_path, img_list[i]))
    mask = cv2.imread(os.path.join(mask_path, mask_list[i]), cv2.IMREAD_GRAYSCALE)

    if mask is None or len(mask.shape) != 2:
        raise ValueError("掩码图必须为单通道黑白图")

    overlay = np.zeros_like(original_image)

    # overlay[mask > 0] = (156, 80, 71) # purple
    overlay[mask > 230] = (250, 134, 192) # orange

    alpha = 0.6
    output_image = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)
    cv2.imwrite(os.path.join(save_path, mask_list[i]), output_image)

# if __name__ == '__main__':
#     img = cv2.imread('/data3/YG/FRINet/code/pred/2-250/COD10K-CAM-1-Aquatic-1-BatFish-2.png')
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             print(img[i, j], end=' ')
