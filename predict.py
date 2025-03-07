from groundedSam import GroundingSAM
import os
import cv2
from tqdm import tqdm

mpath = "example/images/"
model = GroundingSAM()
mnames = sorted(os.listdir(mpath))
os.makedirs("example/masks", exist_ok=True)
for mp in tqdm(mnames):
    image = cv2.imread(os.path.join(mpath, mp))
    classes = ["A sculpture consisting entirely of a rectangular base"]
    mask,_,_ = model(image, classes)
    cv2.imwrite(f"example/masks/{mp.replace('jpg','png')}", mask)

