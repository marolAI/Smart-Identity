import cv2
import numpy as np
import keras
from tensorflow.keras.models import load_model
from keras import backend as K
from tensorflow.keras.utils import img_to_array


def classify(image, model_path):
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
 
	# load the model and classify the image
	print("Load model")
	m = load_model(model_path, compile=False)
	# m = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
	# print(m)
	(notId, Id) = m.predict(image)[0]
	K.clear_session()
	
	# build the label
	label = "ECOWAS ID Card" if Id > notId else "Not ECOWAS ID Card"

	return label
