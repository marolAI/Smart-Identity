from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from keras import backend as K
import numpy as np
import cv2


def id_not_id(image, model="id_not_id.model"):
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# load the trained convolutional neural network
	print("[INFO] loading network...")
	model = load_model(model, compile=False)
	print("Model loaded")
	# classify the input image
	(notId, Id) = model.predict(image)[0]
	K.clear_session()
	
	# build the label
	label = "ID" if Id > notId else "Not ID"
	proba = Id if Id > notId else notId

	return label, proba
