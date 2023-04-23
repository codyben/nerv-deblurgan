from PIL import Image
from glob import glob
imgs = glob("*.jpg")

for i in imgs:
    Image.open(i).resize((256,256)).save(i)