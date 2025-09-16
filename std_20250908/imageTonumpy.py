import numpy as np
from keras.utils import load_img,img_to_array

# path=""

def load_cimg(path,target_size):
    img = load_img(path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    return images