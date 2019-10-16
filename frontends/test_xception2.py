from xception import Xception
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

model = Xception(weights='imagenet')
model.summary()

img_path = 'elephant.jpeg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(preds)
print('Predicted: ', decode_predictions(preds, top=3)[0])
