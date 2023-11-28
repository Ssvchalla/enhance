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

import torchvision.transforms as T
# trf = T.Compose([T.Resize(256),
#                  T.CenterCrop(224),
#                  T.ToTensor(), 
#                  T.Normalize(mean = [0.485, 0.456, 0.406], 
#                              std = [0.229, 0.224, 0.225])])

trf = T.Compose([#T.Resize(350),
                 #T.CenterCrop(650),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])

inp = trf(img).unsqueeze(0)

out = SegModel(inp)['out']

predicted = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (np.unique(predicted))

def decode_segmap(image, nc=21):

  label_colors = np.array([(0, 0, 0), 
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)
  return rgb
rgb = decode_segmap(predicted)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(rgb,cmap = 'gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()

