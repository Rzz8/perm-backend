import tensorflow
from tensorflow.keras.models import Sequential, load_model, save_model, model_from_json
import numpy as np
model_json = 'model_clsf_20.0.json' # .json file with model architecture
model_weights = 'model_clsf_20.0.h5' # .h5 file with model weights

# This reads the model architecture and creates the "model" instance
with open(model_json, 'r') as jsonfile:
	model = model_from_json(jsonfile.read())

# This loads the model weights
model.load_weights(model_weights)

X = np.array([[
	404.97, 10.8, 82.78, 0.54, 355.7, 1.249, 18.4503, 191.37, 311.6, 124.4, 34.99, 0.4804, 0.04804, 1, 0, 0
]])

print(model.predict(X))