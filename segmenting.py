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

