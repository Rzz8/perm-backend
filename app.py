# 1. Import libraries
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
import tensorflow
import tensorflow
from tensorflow.keras.models import Sequential, load_model, save_model, model_from_json
from fastapi.middleware.cors import CORSMiddleware

# Load model for 3-class classification
model_json_3c = 'model_clsf_20.0_80.0.json' # .json file with model architecture
model_weights_3c = 'model_clsf_20.0_80.0.h5' # .h5 file with model weights

# This reads the model architecture and creates the "model" instance
with open(model_json_3c, 'r') as jsonfile:
	model_3c = model_from_json(jsonfile.read())

# This loads the model weights
model_3c.load_weights(model_weights_3c)

# Load model for 2-class classification
model_json_2c = 'model_clsf_20.0.json' # .json file with model architecture
model_weights_2c = 'model_clsf_20.0.h5' # .h5 file with model weights
with open(model_json_2c, 'r') as jsonfile:
	model_2c = model_from_json(jsonfile.read())
model_2c.load_weights(model_weights_2c)

# Load regression model 
pickle_in = open("ann_form_skl.pkl", "rb")
regressor = pickle.load(pickle_in)
pickle_in.close()

class PermValue(BaseModel):
    BD: float
    HL: float
    ROP: float
    IMV: float
    DP: float
    DS: float
    GR: float
    RSS: float
    RSD: float
    RSM: float
    PG: float
    FL: float
    FLR: float
    FG: float

def to_categorical(X, classes=[0,1,2]):
    classes = np.array(classes)
    classes = classes.reshape(1, classes.shape[0])
    X = X.reshape(X.shape[0], 1)
    out = (X.astype(int)==classes).astype(int)
    return out

def add_categorical(X, column=-1, classes=[0,1,2]):
    # Assumes X is 2D
    c = column if column >= 0 else (X.shape[1] + column)
    X_cat = to_categorical(X[:, c], classes=classes)
    X_new = np.concatenate((X[:,:c], X_cat, X[:,(c+1):]), axis=1)
    return X_new

# 2. Create the app object
app = FastAPI()

origins = ['*']

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=False,
                   allow_methods=["*"],
                   allow_headers=["*"])

mmsx_in = open("minmaxscaler_x.pkl", "rb")
mmsx = pickle.load(mmsx_in)
mmsx_in.close()

mmsy_in = open("minmaxscaler_y.pkl", "rb")
mmsy = pickle.load(mmsy_in)
mmsy_in.close()

@app.get('/')
def main():
    return {"hello": "world"}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_permvalue(data: PermValue):
    data = data.dict()
    BD = data['BD']
    HL = data['HL']
    ROP = data['ROP']
    IMV = data['IMV']
    DP = data['DP']
    DS = data['DS']
    GR = data['GR']
    RSS = data['RSS']
    RSD = data['RSD']
    RSM = data['RSM']
    PG =  data['PG']
    FL = data['FL']    
    FLR = data['FLR']
    FG = data['FG']
    X = np.array([[BD, HL, ROP, IMV, DP, DS, GR, RSS, RSD, RSM, PG, FL, FLR, FG]])
    X = add_categorical(X)
    X = mmsx.transform(X)
    prediction1 = np.argmax(model_3c.predict(X))   # 3-class prediction
    prediction2 = np.argmax(model_2c.predict(X))   # 2-class prediction
    prediction3 = np.array([regressor.predict(X)])   # regression prediction  
    prediction3 = mmsy.inverse_transform(prediction3)[0,0]
    prediction3 = round(prediction3, 2)  
    text1 = "The permeability is "
    text2 = "class "
    text3 = " (3-classes with 20-80 md cutoff),"
    text4 = " (2-classes with 20 md cutoff)."
    unit = "md, "
    prediction1 = text2+ str(prediction1) + "  " + text3
    prediction2 = text2+ str(prediction2) + "  " + text4
    prediction3 = text1+ str(prediction3) + "  " + unit
    return {
        "prediction1": prediction1,
        "prediction2": prediction2,
        "prediction3": prediction3
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)