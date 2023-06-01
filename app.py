# 1. Import libraries
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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

# 2. Create the app object
app = FastAPI()

origins = ['*']

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=False,
                   allow_methods=["*"],
                   allow_headers=["*"])

pickle_in = open("ann_skl.pkl", "rb")
regressor = pickle.load(pickle_in)
pickle_in.close()

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
    X = np.array([[BD, HL, ROP, IMV, DP, DS, GR, RSS, RSD, RSM, PG, FL, FLR]])
    X = mmsx.transform(X)
    prediction = np.array([regressor.predict(X)])
    prediction = mmsy.inverse_transform(prediction)[0,0]
    prediction = round(prediction, 2)
    unit = "md"
    prediction = str(prediction) + " " + unit
    return {"prediction": prediction}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)