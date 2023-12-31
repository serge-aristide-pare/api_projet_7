# Importing Necessary modules
from fastapi import FastAPI, Request
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Chargement du modèle
import joblib
from joblib import load
# import uvicorn
# Recuperation de l'identifiant du client
from pydantic import BaseModel
from typing import List
from fastapi.encoders import jsonable_encoder
import shap
 
# Declaring our FastAPI instance
app = FastAPI(
    title="Credit Scoring API",
    description="Un simple API utilisant le modèle de machine learning pour predire le score credit",
    version="0.1",
    )

base_200_clients = pd.read_csv("base_200_clients.csv", index_col='SK_ID_CURR')

loaded_model = load('modele_ok.joblib')
loaded_interpretabilite = load('interpretabilite.joblib')
explainer = load('explainer.joblib')
shap_val_global = explainer.shap_values(base_200_clients.drop(columns=['NAME_CONTRACT_TYPE']))
cols= loaded_model.best_estimator_.named_steps['lgbm'].feature_name_

## API ENPOINTS
# -----------------------#

#########################
# Defining path operation for root endpoint
@app.get('/Loan application scoring dashboard')
def main():
    return {'message': 'Bienvenu dans notre tableau de bord du credit scoring !'}
 
class request_body(BaseModel):
    id_client : int
    
@app.post('/predict') # local : http://127.0.0.1:8000/predict
def predict(data : request_body):
    test_data = data.id_client
    df_client = base_200_clients.loc[[test_data]].drop(columns=['NAME_CONTRACT_TYPE'])#[cols]
    class_idx = loaded_model.predict(df_client)[0]
    class_proba = loaded_model.predict_proba(df_client)
    prob=None
    if class_idx ==0 :
        prob = class_proba[0][0]
    else :
        prob = class_proba[0][1]
    return {"client_id": test_data, "prediction": int(class_idx), "probabilite": round(prob, 3)}

@app.get('/shaplocal/{client_id}')
def shap_values_local(client_id: int):
    """ Calcul les shap values pour un client.
        :param: client_id (int)
        :return: shap values du client (json).
        """
    client_data = base_200_clients.loc[[int(client_id)]].drop(columns=['NAME_CONTRACT_TYPE'])
    shap_val = explainer(client_data)[0][:, 1]
    return {'shap_values': shap_val.values.tolist(),
            'base_value': shap_val.base_values,
            'data': client_data.values.tolist(),
            'feature_names': client_data.columns.tolist()}

@app.get('/shap')
def shap_values():
    """ Calcul les shap values de l'ensemble du jeu de données
    :param:
    :return: shap values
    """
    client_data = base_200_clients.drop(columns=['NAME_CONTRACT_TYPE'])
    return {'shap_values_0': shap_val_global[0].tolist(),
            'shap_values_1': shap_val_global[1].tolist(),
           'feature_names': client_data.columns.tolist()}

# if __name__ == '__main__':
#        port = int(os.environ.get('PORT', 8080))
#        st.write(f"Running on port {port}")
#        app.run(port=port)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0")
    
