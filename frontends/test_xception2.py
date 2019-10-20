from xception import Xception
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf

model1 = Xception(weights='imagenet') #mine
model2 = model2 = tf.keras.applications.xception.Xception()
model1.summary()
model2.summary()

img_path = 'elephant.jpeg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds1 = model1.predict(x)
preds2 = model2.predict(x)
print(preds1)
print(preds2)

print('Predicted: ', decode_predictions(preds1, top=3)[0])
print('Predicted: ', decode_predictions(preds2, top=3)[0])
