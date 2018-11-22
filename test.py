from keras.preprocessing.image import img_to_array
from keras.models import load_model

from PIL import Image
import numpy as np

import argparse

# load image
parser = argparse.ArgumentParser(description='Bird or not')
parser.add_argument('-i', '--image', required=True, )
arg = vars(parser.parse_args())

im = Image.open(arg['image'])
orig = im.copy()
im = im.resize((28, 28), Image.ANTIALIAS)
im = img_to_array(im)
im = im.astype("float") / 255.0
im = img_to_array(im)
im = np.expand_dims(im, axis=0)

# and model 
model = load_model("bird_or_not.model")
(no_bird, bird) = model.predict(im)[0]

print("{} - {}".format(no_bird, bird))