import numpy as np
import random
import os, math, re
from glob import glob
from PIL import Image
random.seed(7760)

def get_real_idx(path) -> int:
    idx_finder = re.compile("(.*)(\d{5})")
    matches = idx_finder.match(path)
    return int(matches.group(2).lstrip("0"))

def resize_and_array(path, w, h):
    i = Image.open(path)
    i = i.resize((w,h))
    return np.asarray(i, dtype=np.float32)

def load_dataset(dataset:str = "dynamic_dog", batch_size:int = 64, random:bool = False, width:int = 64, height:int = 64):
    """"
    Loads chunks of 64 images in a sorted order by default, unless the random flag is passed. In that case, non-overlapping subsets of 64 images are returned.
    All images are NOT normalized and come in a numpy array.
    """
    
    if not os.path.exists(dataset):
        raise Exception(f"Folder named {dataset} not found in {os.getcwd()}")
    
    images = glob(f"{dataset}/*.png")
    images.sort(key=get_real_idx)
    total_images = len(images)

    if random:
        random.shuffle(images)

    if batch_size < 0:
        true_indexes = tuple(get_real_idx(p) for p in images)
        image_arrays = np.asarray([resize_and_array(p, width, height) for p in images])
        yield (true_indexes, image_arrays)
    
    batch_count = math.ceil(total_images / batch_size)

    for i in range(batch_count):
        begin = i*batch_size
        end = (i+1)*batch_size
        if end > total_images:
            end = total_images # read until end of array so we don't go out of bounds
        subset = images[begin:end]
        true_indexes = tuple(get_real_idx(p) for p in subset)
        image_arrays = np.asarray([resize_and_array(p, width, height) for p in subset])
        yield (true_indexes, image_arrays)

