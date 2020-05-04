import cv2
import os
from model import LSCCNN
from matplotlib import pyplot as plt

checkpoint_path = './weights/part_b_scale_4_epoch_24_weights.pth'
checkpoint_path = './weights/qnrf_scale_4_epoch_46_weights.pth'
checkpoint_path = './weights/part_a_scale_4_epoch_13_weights.pth'
# checkpoint_path = './weights/part_a/scale_4_epoch_13.pth'

network = LSCCNN(checkpoint_path=checkpoint_path)
network.cuda()
network.eval()

image_dir = 'head-detection'
output_dir = 'head-detection-output'
for image_file in os.listdir(image_dir):
    print(image_file)
    if 'jpg' in image_file:
        
        image_filepath = os.path.join(image_dir, image_file)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pred_dot_map, pred_box_map, img_out = network.predict_single_image(image, nms_thresh=0.1)

        plt.figure()
        plt.imshow(img_out)
        output_filepath = os.path.join(output_dir, image_file)
        plt.imsave(output_filepath, img_out)
        # plt.show()