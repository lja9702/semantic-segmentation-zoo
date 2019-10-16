from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from xception import Xception
tf.random.set_seed(777)
from PIL import Image
import numpy as np
from keras.applications.inception_v3 import preprocess_input #InceptionV3, Xception, InceptionResNetV2
#from keras.applications.imagenet_utils import preprocess_input #others

file_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
labels = open(file_path).read().splitlines()
print(len(labels)) #1001
del labels[0]
print(len(labels)) #1000

model1 = Xception(weights='imagenet')
model2 = tf.keras.applications.xception.Xception() #Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
model1.summary()
model2.summary()
##########모델 예측

file_path = tf.keras.utils.get_file('cock.png', 'http://www.farmersonlinemarket.co/wp-content/uploads/2018/01/cock.png')
image = Image.open(file_path)
width = 299
height = 299
image = image.resize((width, height))
image = np.array(image)
print(image.shape)
x_test = [image]
x_test =  np.array(x_test)
#x_test = x_test / 255
x_test = preprocess_input(x_test)

y1_predict = model1.predict(x_test)
y2_predict = model2.predict(x_test)
print(y1_predict) #[[0.20842288 0.41051054 0.38106653]]
print(y2_predict)

print(y1_predict.argmax(axis=1)) #[1]
print(y2_predict.argmax(axis=1)) #[1]

print(y1_predict.argmax(axis=1)[0]) #1
print(y2_predict.argmax(axis=1)[0]) #1

print(labels[y1_predict.argmax(axis=1)[0]]) #B
print(labels[y2_predict.argmax(axis=1)[0]]) #B
