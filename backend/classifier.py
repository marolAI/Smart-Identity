import cv2
import numpy as np

from tensorflow.keras.models import load_model
from keras import backend as K
from tensorflow.keras.utils import img_to_array


def classify(image, model="id_not_id.model"):
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
 
	# load the model and classify the image
	model = load_model(model, compile=False)
	(notId, Id) = model.predict(image)[0]
	K.clear_session()
	
	# build the label
	label = "ID" if Id > notId else "Not ID"

	return label
