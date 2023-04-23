from helper import load_dataset
import tensorflow as tf
from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image, save_image, preprocess_image_7760
import PIL
import numpy as np
g = generator_model()
g.load_weights('generator.h5')
# load in the "dynamic_dog data". Pass batch_size = -1 to load in whole dataset (will be slower)
gen = load_dataset(batch_size=1, width=256, height=256, dataset="bunny")
# access our images
for idx, img in gen:
    img = preprocess_image_7760(img)
    img_out = g.predict(x=img, batch_size=img.shape[0])
    img_out = np.squeeze(img_out) * 127.5 + 127.5
    x = deprocess_image(np.squeeze(img))[:, :, :]
    # save_image(img_out, f"output/frame_{idx}.png")
    output = np.concatenate((x, img_out), axis=1)
    # PIL.Image.fromarray(x.astype(np.uint8)).save(f"output/frame_real_{idx}.jpg")
    PIL.Image.fromarray(output.astype(np.uint8)).save(f"bunny_out/compare_gen_{idx[0]}.png")