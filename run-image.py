import cv2
import torch
from model import LSCCNN
from matplotlib import pyplot as plt

checkpoint_path = './weights/part_b_scale_4_epoch_24_weights.pth'
checkpoint_path = './weights/part_a_scale_4_epoch_13_weights.pth'
checkpoint_path = './weights/qnrf_scale_4_epoch_46_weights.pth'


network = LSCCNN(checkpoint_path=checkpoint_path)
if torch.cuda.is_available():
    network.cuda()
network.eval()

image = cv2.imread('./source/hands.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pred_dot_map, pred_box_map, img_out = network.predict_single_image(image, nms_thresh=0.10)

plt.figure()
plt.imshow(img_out)
plt.show()