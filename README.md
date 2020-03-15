# VPCA

Vision-based Posture Correction Application, being developed on NVIDIA Jetson Nano. Pose estimation code and model are referenced and from [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) repository.



### Requirement

Pytorch(torch, torchvision). Pillow, OpenCV, tqdm,

[torch2trt](github.com/NVIDIA-AI_IOT/torch2trt)

[JetCam](github.com/NVIDIA-AI_IOT/jetcam)

[trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)



### Usage

1. Download pre-trained pose estimation model from [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) repository (resnet 224x224) and put the weight file in vpca folder (same folder as main.py)
2. Run main.py

```bash
cd vpca

python main.py
```

