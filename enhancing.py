from torchvision import models
SegModel=models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

img = Image.open('/content/drive/MyDrive/input/1 (678).jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

import cv2
img1=cv2.imread('/content/drive/MyDrive/input/1 (678).jpg').astype('float32')
gray_img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap = plt.cm.gray)
plt.axis("off")
plt.show()

img1=np.uint8(cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX))
plt.axis("off")
img1=cv2.equalizeHist(img1)
plt.imshow(img1,cmap=plt.cm.gray)

clahe=cv2.createCLAHE(clipLimit=40)
gray_img_clahe=clahe.apply(img1)
plt.imshow(gray_img_clahe, cmap = plt.cm.gray)
plt.axis("off")
plt.show()

bilateral = cv2.bilateralFilter(gray_img_clahe, 9, 100, 100)
plt.imshow(bilateral, cmap = plt.cm.gray)
plt.axis("off")
plt.show()

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=gray_img_clahe, ddepth=-1, kernel=kernel)
plt.imshow(image_sharp, cmap = plt.cm.gray)
plt.axis("off")
plt.show()
