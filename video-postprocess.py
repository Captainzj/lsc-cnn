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

def frames_to_video(frames_path, fps, size, start_index, end_index, save_path):  
    """
    [start_index, end_index]
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)  # size 不匹配，将导致生成的文件损坏（无法打开）
    for i in range(start_index, end_index+1):
        if os.path.isfile("%s/%d.jpg"%(frames_path, i)):
            frame = cv2.imread("%s/%d.jpg"%(frames_path, i))
            videoWriter.write(frame)
    videoWriter.release()
    return

if __name__ == "__main__":

    video_file = "source/head-detection.mp4"
    
    fps, size, frames = get_video_info(video_path=video_file)

    # frames → video
    output_dir = "outputs"
    for result_frames_dir in os.listdir(output_dir):
        if result_frames_dir.endswith('outputs'):
            print(result_frames_dir)
            frames_path = os.path.join(output_dir, result_frames_dir)
            end_index = len(os.listdir(frames_path))
            result_file = os.path.join(output_dir, result_frames_dir + ".mp4")
            # pdb.set_trace()

            t1 = datetime.now()
            frames_to_video(frames_path, fps=fps, size=size, start_index=1, end_index = end_index, save_path=result_file)
            t2 = datetime.now()
            print("Time cost = ", (t2 - t1))
            print("frames → video SUCCEED !!!")
    
    pass