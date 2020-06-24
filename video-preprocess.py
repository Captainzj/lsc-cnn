# -*- coding: UTF-8 -*-
import cv2
import glob
import os
from datetime import datetime
import pdb 
 

def get_video_info(video_path):

    videoCapture = cv2.VideoCapture()
    videoCapture.open(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS) # 帧率
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # (width, height)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    print("fps=", int(fps), "size=", size, "frames=", int(frames))
    
    return fps, size, frames



def video_to_frames(video_path, frames_path='frames', crop_flag=False):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(video_path)
    fps, size, frames = get_video_info(video_path)

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    for i in range(int(frames)):
        ret, frame = videoCapture.read()  # frame (1080, 1440, 3) (height, width, channel)
        frame_height, frame_width, frame_channel = frame.shape
        if crop_flag:
            cropped_frame = frame[:, frame_width//4:-frame_width//4, :]  # crop !!!
            cropped_size = (cropped_frame.shape[1], cropped_frame.shape[0])  # (width, height)
            cv2.imwrite("%s/%d.jpg"%(frames_path, i+1), cropped_frame)
            size = cropped_size
        else:
            cv2.imwrite("%s/%d.jpg"%(frames_path, i+1), frame)

    videoCapture.release()
    


if __name__ == "__main__":

    video_file = "source/head-detection.mp4"
    
    # video → frames
    frames_dir = 'outputs/head-detection-frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    t1 = datetime.now()
    video_to_frames(video_file, frames_path=frames_dir)
    t2 = datetime.now()
    print("Time cost = ", (t2 - t1))
    print("video → frames SUCCEED !!!")

