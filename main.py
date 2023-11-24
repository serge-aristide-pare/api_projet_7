# Importing Necessary modules
from fastapi import FastAPI, Request
# Chargement du modèle
import joblib
from joblib import load
import uvicorn
import pandas as pd
# Recuperation de l'identifiant du client
from pydantic import BaseModel
from typing import List
from fastapi.encoders import jsonable_encoder
 
# Declaring our FastAPI instance
app = FastAPI(
    title="Credit Scoring API",
    description="Un simple API utilisant lemodèle de machine learning pour predire le score credit",
    version="0.1",
    )

# Load model a serialized .joblib file

loaded_model = load('modele_ok.joblib')
loaded_interpretabilite = load('interpretabilite.joblib')

cols= loaded_model.best_estimator_.named_steps['lgbm'].feature_name_


## API ENPOINTS
# ------------------------------------------------------------

#########################
# Defining path operation for root endpoint
@app.get('/Loan application scoring dashboard')
def main():
    return {'message': 'This is credit scoring !'}
 
# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to GeeksforGeeks!, {name}'}



base_200_clients = pd.read_csv("base_200_clients.csv", index_col='SK_ID_CURR')


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
    

## Retour d'image ou valeur dans l'interpretabilite

@app.get('/interpretability/{client_id}')
def get_interpretability(client_id: int):
    test_data = data.id_client
    df_client = base_200_clients[base_200_clients['SK_ID_CURR']]
    class_idx = loaded_interpretabilite.predict(df_client)[0]
    class_proba = loaded_interpretabilite.predict_proba(df_client)
    
    class_prob = None
    if df_client == 0 :
        class_prob = 'bon client'
    else :
        class_prob = 'mauvais client'
    
    return { "client_id": client_id, "interpretability_info": class_prob }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
    #chargement des données Test
    df_test = pd.read_csv("base_200_clients.csv", index_col='SK_ID_CURR')
    # df_test = pd.read_csv('df_test.csv').drop('Unnamed: 0', axis = 1)
    df_test = df_test.set_index('SK_ID_CURR')
    #chargement du meilleur modèle
    loaded_model = load('modele_ok.joblib')
    loaded_interpretabilite = load('interpretabilite.joblib')
    
# uvicorn app:app --reload

# Supprimer la colonne target dans les 200 clients avant export
# Rajouter un autre point de sortie pour retourner l'interpretabilité du client
    
    
