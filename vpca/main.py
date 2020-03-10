import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import cv2
import torchvision.transforms as transforms
import PIL.Image
import time

from util import Utils
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
ut = Utils(topology)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))
WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

# camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)



while True:
    image = camera.read()
    data = ut.preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    # counts, objects, peaks = ut.parseObjects(cmap, paf)
    counts, objects, peaks = ut.parseObjects(cmap, paf)
    ut.drawObjects(image, counts, objects, peaks)
    cv2.imshow("Image", image)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    time.sleep(0.1)

cv2.destroyAllWindows()


