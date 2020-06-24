# LSC-CNN

### Prepare

```
$ pip install -r requirement.txt
```

* View at `weights/README.md` 下载相应模型


### Simple test on image

```
$ python run-image.py
```

Or view at `simple_example.ipynb`


### 视频检测

```
$ python video-preprocess.py
$ python run-video.py
$ python video-postprocess.py
```

效果如`outputs/head-detection-frames-part_a-0.1-outputs.mp4`所示


### 摄像头实时监测

```
$ python run-real-time.py
```

* 若主机有NVIDIA GPU，强烈建议安装cuda，可极大地提升检测效率
