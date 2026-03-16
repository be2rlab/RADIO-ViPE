import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import time
device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
# model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
# model_b = MoGeModel.from_pretrained("Ruicheng/moge-2-vitb-normal").to(device)
# model_s = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").to(device)

model = MoGeModel.from_pretrained("/data/weights/moge/moge-2-vitl-normal.pt").to(device)
model_b = MoGeModel.from_pretrained("/data/weights/moge/moge-2-vitb-normal.pt").to(device)
model_s = MoGeModel.from_pretrained("/data/weights/moge/moge-2-vits-normal.pt").to(device)


# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("/data/Replica/office_0/rgb/000001.png"), cv2.COLOR_BGR2RGB)                    
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
tt0 = tt1 = tt2 = 0
for i in range(1000):
    t0 = time.time()
    output = model.infer(input_image)
    t1 = time.time()
    output_b = model_b.infer(input_image)
    t2 = time.time()
    output_s = model_s.infer(input_image)
    t3 = time.time()
    tt0 += t1 - t0
    tt1 += t2 - t1
    tt2 += t3 - t2


print(f"inference of model L: {tt0/1000}")
print(f"inference of model B: {tt1/1000}")
print(f"inference of model S: {tt2/1000}")

"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""