import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import TensorBoard

import os, os.path
import math

train_categories = []
train_samples = []
for i in os.listdir("./data/merged/train"):
    train_categories.append(i)
    train_samples.append(len(os.listdir("./data/merged/train/" + i)))

test_categories = []
test_samples = []
for i in os.listdir("./data/merged/test"):
    test_categories.append(i)
    test_samples.append(len(os.listdir("./data/merged/test/" + i)))

print("No. of Training Samples:", sum(train_samples))
print("No. of Test Samples:", sum(test_samples))

# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 30
# fig_size[1] = 5
# plt.rcParams["figure.figsize"] = fig_size
#
# index = np.arange(len(train_categories))
# plt.bar(index, train_samples)
# plt.xlabel('Categories', fontsize=25)
# plt.ylabel('No. of samples', fontsize=25)
# plt.xticks(index, train_categories, fontsize=15, rotation=90)
# plt.title('Category wise training sample distribution', fontsize=35)
# plt.show()

# index2 = np.arange(len(test_categories))
# plt.bar(index2, test_samples)
# plt.xlabel('Categories', fontsize=25)
# plt.ylabel('No. of samples', fontsize=25)
# plt.xticks(index2, test_categories, fontsize=15, rotation=90)
# plt.title('Category wise test sample distribution', fontsize=35)
# plt.show()

train = []
test = []

for i in os.listdir("./data/merged/train"):
    one_hot = np.zeros(shape=[len(train_categories)])
    actual_index = train_categories.index(i)
    one_hot[actual_index] = 1
    for files in os.listdir("./data/merged/train/" + i):
        img_array = mpimg.imread("./data/merged/train/" + i + "/" + files)
        train.append([img_array, one_hot])
    # print("Train Category Status: {}/{}".format(actual_index+1, len(train_categories)))

for i in os.listdir("./data/merged/test"):
    one_hot = np.zeros(shape=[len(test_categories)])
    actual_index = test_categories.index(i)
    one_hot[actual_index] = 1
    for files in os.listdir("./data/merged/test/" + i):
        img_array = mpimg.imread("./data/merged/test/" + i + "/" + files)
        test.append([img_array, one_hot])
    # print("Test Category Status: {}/{}".format(actual_index+1, len(test_categories)))

train_x = []
train_y = []
for i in range(len(train)):
    train_x.append(train[i][0])
    train_y.append(train[i][1])
    #print("Status {}/{}".format(i+1, len(train)))

test_x = []
test_y = []
for i in range(len(test)):
    test_x.append(test[i][0])
    test_y.append(test[i][1])
    #print("Status {}/{}".format(i+1, len(test)))

idx = np.random.choice(len(train), size=math.floor(len(train)*0.2))
validation_from_test_x = []
validation_from_test_y = []

for i in range(len(idx)):
    validation_from_test_x.append(test[i][0])
    validation_from_test_y.append(test[i][1])
    #print("Status {}/{}".format(i+1, len(idx)))


training_x = np.asarray(train_x, dtype=np.float32)
training_y = np.asarray(train_y, dtype=np.float32)
testing_x = np.asarray(test_x, dtype=np.float32)
testing_y = np.asarray(test_y, dtype=np.float32)
validation_x = np.asarray(validation_from_test_x, dtype=np.float32)
validation_y = np.asarray(validation_from_test_y, dtype=np.float32)

for i in range(len(training_x)):
    training_x[i] = training_x[i]/255

for i in range(len(testing_x)):
    testing_x[i] = testing_x[i]/255

for i in range(len(validation_x)):
    validation_x[i] = validation_x[i]/255


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential

model = Sequential()

# glorot_normal means xavier normal initializer
model.add(Conv2D(input_shape=(100, 100, 3), kernel_size=(7,7), filters=16, padding='same', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(kernel_size=(5,5), filters=32, padding='valid', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(strides=2, padding='valid'))

model.add(Conv2D(kernel_size=(7,7), filters=32, padding='same', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(kernel_size=(5,5), filters=64, padding='valid', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(strides=2, padding='valid'))

model.add(Conv2D(kernel_size=(5,5), filters=128, padding='valid', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(kernel_size=(3,3), filters=256, padding='valid', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(strides=2, padding='valid'))

model.add(Conv2D(kernel_size=(3,3), filters=256, padding='valid', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(kernel_size=(3,3), filters=512, padding='valid', use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(alpha=0.1))
model.add(GlobalAveragePooling2D())


#model.add(Dropout(0.2))
model.add(Dense(256, use_bias=False, activation=None, kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=1))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Dense(len(train_categories), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/{}".format("Fruits_recognition"))
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x=training_x, y=training_y, batch_size=32, epochs=10,  callbacks=[tensorboard], validation_data=(validation_x, validation_y))
model.save("model.h5")

test_accuracy = 0
predict_y = model.predict(x=testing_x, batch_size=32, verbose=1)
for i in range(len(testing_x)):
    if (np.argmax(predict_y[i]) == np.argmax(testing_y[i])):
        test_accuracy += 1

test_accuracy = test_accuracy / len(testing_x)*100
print("Test Accuracy: ", test_accuracy)



