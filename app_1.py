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
import shap
import json
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
# Pour envoyer un mail
import smtplib
from email.mime.text import MIMEText

#chargement des données Valid/Test
df_valid = pd.read_csv("base_200_clients.csv")
list_id = list(df_valid['SK_ID_CURR'])
list_col = df_valid.columns.tolist()
df_valid = df_valid.set_index('SK_ID_CURR')
feats = [f for f in df_valid.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X_valid = df_valid[feats]



## Début du dashboard
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
urli = 'http://127.0.0.1:8000/shaplocal'
url3 = 'http://127.0.0.1:8000/shap'

##########################################################
####### Titre de la page ################################
##########################################################
st.title("Projet 7 - Implémentez un modèle de scoring")
########################################################

# Menu déroulant
values = list_id
values.insert(0, '<Select>')
num = st.sidebar.selectbox(
    "**Veuillez sélectionner un numéro de demande de prêt**",
    values
)
if (num != '<Select>') :
    idx = num
    j_idx = {'id_client':idx}
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request = requests.post(url, json = j_idx, headers = headers)
    req = request.json()
    # print(req)
    proba_api = req['probabilite']
    rep_api = req['prediction']
    # best_threshold =0.828997

    #Prédiction
    if rep_api == 0:
        t = "<span class='highlight blue'><span class='bold'>Acceptée</span></span>"
    else : 
        t = "<span class='highlight red'><span class='bold'>Rejetée</span></span>"
    
    proba_0 = round(1-proba_api, 2)

## Interpretabilité        
    ###############################
    # Présentation des résultats #
    ###############################
    st.markdown(
        """
        <style>
        .header-style { font-size:25px; font-family:sans-serif;}
        </style>
        """, unsafe_allow_html = True
    )
    st.markdown(
        """
        <style>
        .font-style { font-size:20px; font-family:sans-serif; }
        </style>
        """, unsafe_allow_html = True )
    # st.markdown("<h2> Statut de la demande n° </h2>", num #df_valid.loc[[num]]               )
    st.write("Statut de la demande n°",num )
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
        value = proba_api*100,
        mode = "gauge+number",
        title = {'text': "Jauge de Score", 'font': {'size': 24}},
        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
                 'bar': {'color': "blue"},
                 'steps' : [
                     {'range': [0, 20], 'color': "red"},
                     {'range': [20, 50], 'color': "orange"},
                     {'range': [50, 70], 'color': "lightgreen"},
                     {'range': [70, 100], 'color': "green"}],
                 'threshold' :
                 {'line': {'color': "red", 'width': 4}, 'thickness': 1, 'value': proba_api*100,}
                }
            ))
    st.plotly_chart(gauge)
    
    ###############################################################
    
        ## Pretraitement des données
    #***************************************************#
    def prepocessing_var(df, scaler):
        cols = df.select_dtypes(['float64']).columns
        df_scaled = df.copy()
        if scaler == 'minmax':
            scal = MinMaxScaler()
        else:
            scal = StandardScaler()

        df_scaled[cols] = scal.fit_transform(df[cols])
        return df_scaled

    ## Interpretation locale
    #***************************************************#
    def valeur_shape(num):
        url_get_shap_local = urli+"/"+str(num)
        response = requests.get(url_get_shap_local)
        res = response.json()
        print(res)
        shap_val_local = res['shap_values']
        base_value = res['base_value']
        feat_values = res['data']
        feat_names = res['feature_names']

        explanation = shap.Explanation(np.reshape(np.array(shap_val_local, dtype='float'), 
                                                  (1, -1)), base_value,
                                       data=np.reshape(np.array(feat_values, dtype='float'), 
                                                       (1, -1)), feature_names=feat_names)
        return explanation[0]

    ## Interpretation globale
    #***************************************************#
    def get_shap_val():
        url_get_shap = url3
        response = requests.get(url_get_shap)
        content = response.json()
        shap_val_glob_0 = content['shap_values_0']
        shap_val_glob_1 = content['shap_values_1']
        shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])

        return shap_globales, content['feature_names']
    ###############################################################

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

    column_1, column_2 = st.columns(2)
    
    # Détails individuels
    with column_2 :
        
        column_2.markdown(
                '<h3>Détails individuels</h3>',
                unsafe_allow_html = True
        )
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

        st.info("Interprétation locale de la prédiction")
        shap_val = valeur_shape(num)
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 11)
        # Affichage du waterfall plot : shap local
        fig = shap.waterfall_plot(shap_val, max_display=nb_features, show=False)
        st.pyplot(fig)

        with st.expander("Explication du graphique", expanded=False):
            st.caption("Ici sont affichées les caractéristiques influençant de manière locale la décision. "
                           "C'est-à-dire que ce sont les caractéristiques qui ont influençé la décision pour ce client "
                           "en particulier.")
    
    # Détails globaux
    with column_1 :
        column_1.markdown(
                '<h3>Détails globaux</h3>',
                unsafe_allow_html = True
        )
        st.info("Importance globale")
        n_features = st.slider('Nombre de variables à visualiser', 0, 20, 9)
        shap_values, feature_names = get_shap_val()
        fig, ax = plt.subplots()
        # Affichage du summary plot : shap global
        ax = shap.summary_plot(shap_values[1], shap_values[0], plot_type='bar', max_display=n_features, feature_names=feature_names)
        st.pyplot(fig)

        with st.expander("Explication du graphique", expanded=False):
            st.caption("Ici sont affichées les caractéristiques influençant de manière globale la décision.")  
    
##############################################
# Affichage information sur état civil ######
#############################################

if st.checkbox("Afficher les informations supplémentaires du client?"):
    st.header("**Informations client**")
    st.write("Identifiant du client :", num)
    st.write("Age du client :", int(df_valid.loc[[num]]["DAYS_BIRTH"].values[0] / -365), "ans.")
    st.write("Montant credit :", df_valid.loc[[num]]["AMT_CREDIT"].values[0])
    st.write("Nombre d'enfant(s) :", df_valid.loc[[num]]["CNT_CHILDREN"].values[0])
    st.write("Ancienneté du client à la banque :", int(df_valid.loc[[num]]["DAYS_REGISTRATION"].values[0] / -365), "ans.")
    
#############################################    
#### Histogramme de quelques variables ######
#############################################
values_col = list_col
values_col.insert(0, '<Select>')
col = st.sidebar.multiselect(
    # st.multiselect
    "**Veuillez sélectionner une variable dans la base de demande de prêt**",
    values_col
)
if (col != '<Select>') :
    # Créer un histogramme pour la colonne sélectionnée
    for col in col:
        st.header("**Voici la distribution des variables que vous avez selectionner**")
        # print(df_valid[col])
        st.header(f"*Voici la distribution de la variable {col}*")
        
        # Generate an histogram for the current variable
        plt.figure(figsize = (8,6))
        sb.histplot(df_valid[col], kde=False)
        
        # Calculate the average/mean
        avg_line = np.mean(df_valid[col])
        
        #Add a vertical line for the average
        plt.axvline(avg_line, color='red', linestyle='dashed', linewidth=2, label='Moyenne')
        plt.title( f"Distribution de : {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        
        #Show the plot
        plt.legend() # ajout legende avec moyenne
        plt.show()  
        st.pyplot(plt)
    st.markdown('[A Project by L. Serge Aristide PARE](https://github.com/serge-aristide-pare)')
    
# Contact Form
with st.expander("Contact us"):  

    st.title('Send Email 💌 🚀')

    st.markdown("""
    *Enter your email, subject, and email body then hit send to receive an email!*
    """)

    # Taking inputs
    email_sender = st.text_input('From', 'Veuillez mettre votre mail')
    email_receiver = st.text_input('To')
    subject = st.text_input('Subject')
    body = st.text_area('Body')

    # Password input
    password = st.text_input('Password', type="password")

    if st.button("Send Email"):
        try:
            msg = MIMEText(body)
            msg['From'] = email_sender
            msg['To'] = email_receiver
            msg['Subject'] = subject

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()

            # Utiliser le mot de passe d'application ou le mot de passe principal
            server.login(email_sender, password)

            server.sendmail(email_sender, email_receiver, msg.as_string())
            server.quit()

            st.success('Email sent successfully! 🚀')
        except Exception as e:
            st.error(f"Failed to send email: {e}")
