import os
import cv2
import torch
from model import LSCCNN
from matplotlib import pyplot as plt


checkpoint_path = './weights/part_a_scale_4_epoch_13_weights.pth'
checkpoint_path = './weights/part_b_scale_4_epoch_24_weights.pth'
checkpoint_path = './weights/qnrf_scale_4_epoch_46_weights.pth'

network = LSCCNN(checkpoint_path=checkpoint_path)
if torch.cuda.is_available():
    network.cuda()
network.eval()

weights_tag = 'part_a'
nms_thresh = 0.1

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read() # 读取一帧的图像
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    pred_dot_map, pred_box_map, img_out = network.predict_single_image(frame, nms_thresh)
    cv2.imshow('Head Recognition', img_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # 释放摄像头
cv2.destroyAllWindows()

