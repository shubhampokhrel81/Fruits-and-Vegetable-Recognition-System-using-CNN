import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from keras.models import Model

from keras.models import load_model

model = load_model('model.h5')

import os, os.path

train_categories = []
train_samples = []
for i in os.listdir("./data/merged/train"):
    train_categories.append(i)

# models.load_weights("finalmodel.hdf5")
img = Image.open("./data/test_images/29.jpg")
original_img = np.array(img, dtype=np.uint8)
plt.imshow(original_img)

if img.size[0] > img.size[1]:
    scale = 100 / img.size[1]
    new_h = int(img.size[1] * scale)
    new_w = int(img.size[0] * scale)
    new_size = (new_w, new_h)
else:
    scale = 100 / img.size[0]
    new_h = int(img.size[1] * scale)
    new_w = int(img.size[0] * scale)
    new_size = (new_w, new_h)

resized = img.resize(new_size)
resized_img = np.array(resized, dtype=np.uint8)
plt.imshow(resized_img)
#plt.show()


left = 0
right = left + 100
up = 0
down = up + 100

cropped = resized.crop((left, up, right, down))
cropped_img = np.array(cropped, dtype=np.uint8)
#plt.imshow(cropped_img)
#plt.show()

cropped_img = cropped_img / 255.0

X = np.reshape(cropped_img, newshape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
#print(X)
prediction_multi = model.predict(x=X)
store = np.argmax(prediction_multi)
#print(np.argmax(prediction_multi))

print("Predicted image is : ", train_categories[store])
plt.show()
