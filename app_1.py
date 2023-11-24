import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
from joblib import load
from fastapi import FastAPI, Request
from typing import List
from PIL import Image
import requests
import plotly.graph_objects as go

# loaded_model = load('modele_ok.joblib')
# loaded_interpretabilite = load('interpretabilite.joblib')

base_200_clients = pd.read_csv("base_200_clients.csv")
print(list(base_200_clients.columns))
list_id=list(base_200_clients['SK_ID_CURR'])
region = base_200_clients['REGION']
df_test = pd.read_csv("base_200_clients.csv")

st.set_page_config(page_title='Loan application scoring dashboard', 
                   page_icon = "🏡", layout="wide")

# Définition de quelques styles css
st.markdown(""" 
            <style>
            body {font-family:'Roboto Condensed';}
            h1 {font-family:'Roboto Condensed';}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-18e3th9 {padding-top: 1rem; padding-right: 1rem; 
            padding-bottom: 1rem; padding-left: 1rem;}
            </style> """, unsafe_allow_html=True)

url = 'http://127.0.0.1:8000/predict'

#chargement des données Valid/Test
df_valid = pd.read_csv("base_200_clients.csv")
df_valid = df_valid.set_index('SK_ID_CURR')
feats = [f for f in df_valid.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X_valid = df_valid[feats]


#chargement du palier de probabilité
# f = open('LR_params.json')
# thres_dict = json.load(f)
# thres = thres_dict['Threshold']
thres = 0.5 # a verifier et mettre la bonne valeur

#Titre de la page
st.title("Projet 7 - Implémentez un modèle de scoring")

#Menu déroulant
values = list_id
values.insert(0, '<Select>')
num = st.sidebar.selectbox(
    "Veuillez sélectionner un numéro de demande de prêt",
    values
)
if (num != '<Select>') :
    idx = num
    j_idx = {'id_client':idx}
    # j_idx = json.dumps(idx)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request = requests.post(url, json = j_idx, headers = headers)
    req = request.json()
    print(req)
    proba_api = req['probabilite']
    rep_api = req['prediction']

    #Prédiction
    if rep_api == 0:
        t = "<span class='highlight blue'><span class='bold'>Acceptée</span></span>"
    else : 
        t = "<span class='highlight red'><span class='bold'>Rejetée</span></span>"
    
    #Présentation des résultats
    st.markdown(
        """
        <style>
        .header-style {
            font-size:25px;
            font-family:sans-serif;
        }
        </style>
        """,
        unsafe_allow_html = True
    )
    st.markdown(
        """
        <style>
        .font-style {
            font-size:20px;
            font-family:sans-serif;
        }
        </style>
        """, unsafe_allow_html = True )
    st.markdown('<h2> Statut de la demande </h2>', unsafe_allow_html = True)
    column_1, column_2 = st.columns(2)
    column_1.markdown('<h3>Décision</h3>', unsafe_allow_html = True)
    column_1.markdown(t, unsafe_allow_html = True)

    column_2.markdown('<h3>Score</h3>', unsafe_allow_html = True)
    column_2.subheader(f"{proba_api}")
    
    #################################################
    # Affichage du score et du graphique de gauge  #
    #################################################
    
    gauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = proba_api,
        mode = "gauge+number",
        title = {'text': "Score", 'font': {'size': 24}},
        gauge = {'axis': {'range': [None, 1]},
                 'bar': {'color': "grey"},
                 'steps' : [
                     {'range': [0, 0.67], 'color': "lightblue"},
                     {'range': [0.67, 1], 'color': "lightcoral"}],
                 'threshold' :
                     {'line': {'color': "red", 'width': 4}, 
                      'thickness': 1, 'value': thres,
                     }
                }
            ))

    st.plotly_chart(gauge)

#     # Graphique camembert
#     st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)

#     plt.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     st.sidebar.pyplot()
    
    
    ########################
    # Affichage état civil #
    ########################
    # identite_client=base_200_clients
    st.header("**Informations client**")
    #infos = st.checkbox("Afficher les informations du client?")

    if st.checkbox("Afficher les informations du client?"):
        
        
        #st.write("Age client :", infos_client["age"], "ans.")
        st.write("Statut id_client :**", num, "**")
        st.write("Montant credit :**", df_valid.loc[[num]]["AMT_CREDIT"].values[0], "**")
        # st.write("Nombre d'enfant(s) :**", base_200_clients["CNT_CHILDREN"][0], "**")
        # st.write("Age client :", int(base_200_clients["DAYS_BIRTH"].values / -365), "ans.")
        # st.write("Ancienneté service :", int(base_200_clients["DAYS_EMPLOYED"].values / -365), "ans.")
        # st.write("Ancienneté client :", int(base_200_clients["DAYS_REGISTRATION"].values / -365), "ans.")
    
    
    ############################
    #Importance des variables #
    ###########################
    st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )
    st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )
    nb_ = st.slider('Variables à visualiser', 0, 25, 10)
    if 'key' not in st.session_state:
        st.session_state['nb_var'] = nb_
    else : 
        st.session_state['nb_var'] = nb_
            
    nb = st.session_state['nb_var']

    column_1, column_2 = st.columns(2)
    #Détails individuels
    column_1.markdown(
            '<h3>Détails individuels</h3>',
            unsafe_allow_html = True
    )
    
#     coef_loc = feat_loc(idx, df_test, best_model, explainer, nb)
#     column_1.plotly_chart(coef_loc, use_container_width = True)
#     #Détails globaux
#     column_2.markdown(
#             '<h3>Détails globaux</h3>',
#             unsafe_allow_html = True
#     )
#     coef_glo = feat_glo(coef_global, nb)
#     column_2.plotly_chart(coef_glo, use_container_width = True)
        
#     #Choix de la variables à expliquée
#     feats = df_desc['Row'].tolist()
#     var = st.selectbox(
#             "Veuillez sélectionner une variable à définir",
#             feats
#     )
#     d = df_desc.loc[df_desc['Row'] == var]
#     st.write('Variable {} : {}'.format(var, d['Description'].values[0]))
    
    
    
    
#     #Bouton pour voir distribution
    st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )
    st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html = True
    )

    if st.button('Voir distribution par classes'):
        st.markdown(
        '<h2> Distribution des probabilités </h2>',
        unsafe_allow_html = True
        )
        hist = dist_proba(X_valid, best_model, proba_api)
        st.plotly_chart(hist)    
    

# # Prediction
# if st.sidebar.button ("Predire"):
#     prediction, probability = make_prediction(input_data)
#     st.subheader("Probabilités :")
#     prob_df = pd.DataFrame({
#         'Categories': ["No Default", "Default"],
#         'Probabilité': probability[0],
#     })
    
#     fig = px.bar(prob_df, x='Catégories', y = 'Probabilité', labels={'Probabilité': 'Probabilité (%)'})
#     st.plotly_chart(fig)
    
#     st.subheader("Résultat de prediction")
#     if prediction[0] ==1:
#         st.error("le client en defaut")
#     else:
#         st.succes("le clientne sera pas en defaut")


# # Ajouter une barre de défilement pour ajuster le nombre de points de données
# # num_points = st.slider("Nombre de points de données", min_value=10, max_value=1000, value=100)
# # st.markdown(f"Vous avez choisi {list_id} points de données.")

# def show_predict_page():
#     # st.title("credit scoring")
    
#     st.write("""### Please enter the requiered information to predict cerdit scoring""")
    
# #     id_client = list_id #('127633', '251751') #list_id
    
# #     id_clients=st.selectbox("id_clients", list_id)
    
    
#     # ok = st.button("credi scor")      
        

#     # st.subheader(f"La proba du client est ${make_prediction:.2f}")
        
# show_predict_page()






# # Creer un dossier dashboard comme avec l'appli main
# # charger le DF 200 clients 
# # Crer un drop down list où on a la liste de tous les id_client

# # Utiliser requests pour envoyer le numero du client à l'api afin d'avoir une reponse



# # import streamlit as st
# # import streamlit.components.v1 as components

# from PIL import Image
# import time

# st.markdown("<h2 style='text-align:center; color:floralWhite;'> CREDIT SCORE USING MACHINE LEARNING</h2>", unsafe_allow_html=True)

# col1, col2, col3 = st.columns([1,8,1])

# try:
#     img1 = Image.open("image.jpg")
    

#     with col2:
#         st.image(img1, caption = "Credit Risk Analysis")
#         st.markdown('[A Project by L. Serge Aristide PARE](https://github.com/serge-aristide-pare)')

# except:
#     components.html('''
#     <script>
#         alert("Image Not Loading")
#     </script>
#     ''')
#     st.text("Image Not Loading")

# else:
#     pass

# finally:
#     pass

# # Contact Form

# with st.expander("Contact us"):
#     with st.form(key='contact', clear_on_submit=True):
#         email = st.text_input('Contact Email')
#         st.text_area("Query","Please fill in all the information or we may not be able to process your request")
#         submit_button = st.form_submit_button(label='Send Information')