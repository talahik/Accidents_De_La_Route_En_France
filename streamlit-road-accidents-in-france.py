import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import geojson
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
import joblib
import requests, json
import urllib.parse

########   Fichier Caractéristiques    ##########
#on importe le fichier caractéristiques et on fait quelques modifications pour l'exploiter
carac_2022=pd.read_csv('data/caracteristiques-2022.csv',header=0,sep=";")
carac_2021=pd.read_csv('data/caracteristiques-2021.csv',header=0,sep=";")
carac_2020=pd.read_csv('data/caracteristiques-2020.csv',header=0,sep=";")
carac_2019=pd.read_csv('data/caracteristiques-2019.csv',header=0,sep=";")

name={"Accident_Id":"Num_Acc"}
carac_2022=carac_2022.rename(name,axis=1)
frames = [carac_2022, carac_2021, carac_2020,carac_2019]
fusion_carac = pd.concat(frames)

#Pour pouvoir utiliser les coordonnées GPS avec Geopandas je retraite les données et les convertis en float
fusion_carac["lat"]=fusion_carac["lat"].str.replace(",",".")
fusion_carac["lat"]=fusion_carac["lat"].astype("float")
fusion_carac["long"]=fusion_carac["long"].str.replace(",",".")
fusion_carac["long"]=fusion_carac["long"].astype("float")

#Pour pouvoir exploiter les données sur les départements je retraite les départements < à 10 en n°
fusion_carac.loc[(fusion_carac.dep == '1'), 'dep'] = '01'
fusion_carac.loc[(fusion_carac.dep == '2'), 'dep'] = '02'
fusion_carac.loc[(fusion_carac.dep == '3'), 'dep'] = '03'
fusion_carac.loc[(fusion_carac.dep == '4'), 'dep'] = '04'
fusion_carac.loc[(fusion_carac.dep == '5'), 'dep'] = '05'
fusion_carac.loc[(fusion_carac.dep == '6'), 'dep'] = '06'
fusion_carac.loc[(fusion_carac.dep == '7'), 'dep'] = '07'
fusion_carac.loc[(fusion_carac.dep == '8'), 'dep'] = '08'
fusion_carac.loc[(fusion_carac.dep == '9'), 'dep'] = '09'


name={"Accident_Id":"Num_Acc"}
carac_2022=carac_2022.rename(name,axis=1)
frames = [carac_2022, carac_2021, carac_2020,carac_2019]
fusion_carac = pd.concat(frames)

########   Fichier Usagers    ##########
usagers_2022=pd.read_csv('data/usagers-2022.csv',header=0,sep=";")
usagers_2021=pd.read_csv('data/usagers-2021.csv',header=0,sep=";")
usagers_2020=pd.read_csv('data/usagers-2020.csv',header=0,sep=";")
usagers_2019=pd.read_csv('data/usagers-2019.csv',header=0,sep=";")

# on supprime les doublons existants dans les fichiers 2019 et 2020
usagers_2019.drop_duplicates(inplace=True)
usagers_2020.drop_duplicates(inplace=True)

# on supprime la colonne id_usager qui est présente uniquement dans les fichiers 2021 et 2022
usagers_2022=usagers_2022.drop('id_usager', axis=1)
usagers_2021=usagers_2021.drop('id_usager', axis=1)

usagers_2022['year']=2022
usagers_2021['year']=2021
usagers_2020['year']=2020
usagers_2019['year']=2019

frames = [usagers_2022, usagers_2021, usagers_2020,usagers_2019]
fusion_usagers = pd.concat(frames)

# on supprime de la base les valeurs NR de la variable cible
fusion_usagers=fusion_usagers.loc[fusion_usagers['grav']!=-1]

# Créer la variable AGE
if 'year' in fusion_usagers.columns and 'an_nais' in fusion_usagers.columns:
  fusion_usagers['age'] = fusion_usagers['year'] - fusion_usagers['an_nais']

  # Définir les bins et les labels pour les catégories d'âge
  bins = [0, 18, 40, 60, float('inf')]
  labels = ['0-18 ans', '19-40 ans', '41-60 ans', 'plus de 60 ans']

  # Utiliser pd.cut pour catégoriser 'age'
  fusion_usagers['age_cat'] = pd.cut(fusion_usagers['age'], bins=bins, labels=labels, right=True)

  # Supprimer les colonnes originales
  fusion_usagers = fusion_usagers.drop(['an_nais', 'year'], axis=1)


########   Fichier Lieux    ##########
lieux_2019 = pd.read_csv('data/lieux-2019.csv', sep=";")
lieux_2020 = pd.read_csv('data/lieux-2020.csv', sep=";")
lieux_2021 = pd.read_csv('data/lieux-2021.csv', sep=";")
lieux_2022 = pd.read_csv('data/lieux-2022.csv', sep=";")

fusion_lieux = pd.concat([lieux_2019, lieux_2020, lieux_2021, lieux_2022], ignore_index=True)

#fusion du fichier Lieux avec le fichier usagers
fusion_lieux_usagers = pd.merge(fusion_lieux, fusion_usagers[['Num_Acc', 'grav']], on='Num_Acc', how='left')
fusion_lieux_usagers['an'] = fusion_lieux_usagers['Num_Acc'].astype(str).str[:4]

######## Fichier Vehicules   ##########
vehicules_2019 = pd.read_csv('data/vehicules-2019.csv', sep=";")
vehicules_2020 = pd.read_csv('data/vehicules-2020.csv', sep=";")
vehicules_2021 = pd.read_csv('data/vehicules-2021.csv', sep=";")
vehicules_2022 = pd.read_csv('data/vehicules-2022.csv', sep=";")

fusion_vehicules = pd.concat([vehicules_2019, vehicules_2020, vehicules_2021, vehicules_2022], ignore_index=True)

#fusion du fichier Lieux avec le fichier usagers
fusion_vehicules_usagers = pd.merge(fusion_vehicules, fusion_usagers[['Num_Acc', 'grav']], on='Num_Acc', how='left')

#Import du fichier JSOn pour le contours des départements
with open('data/contour_departements.geojson',encoding='UTF-8') as dep:
    departement = geojson.load(dep)

def load_and_process_data():

  ### Fonction permettant de fusionner les 4 jeux de données ###
    base_path = 'data/'
    annees = [2019, 2020, 2021, 2022]
    types_donnees = ['caracteristiques', 'lieux', 'usagers', 'vehicules']
    dataframes_temp = {}

    for type_donnee in types_donnees:
      #on boucle sur les 4 jeux de données par années
        dataframes = []
        for annee in annees:
            file_path = f'{base_path}{type_donnee}-{annee}.csv'
            try:
                df = pd.read_csv(file_path, sep=';')
                if type_donnee == 'caracteristiques' and annee == 2022:
                    df.rename(columns={'Accident_Id': 'Num_Acc'}, inplace=True)
                dataframes.append(df)
            except FileNotFoundError:
                st.error(f'File not found: {file_path}')
                continue
        # on fusionne les 4 années et on ajoute le dataframe fusionné dans le dictionnaire dataframes_temp
        if dataframes:
            df_fusionne = pd.concat(dataframes, ignore_index=True)
            dataframes_temp[type_donnee] = df_fusionne

    if 'usagers' in dataframes_temp:
        # Supprimer les doublons pour les usagers
        ordre_gravite = [2, 3, 4, 1]  # Du plus grave au moins grave
        data_usagers = dataframes_temp['usagers']
        data_usagers_sorted = data_usagers.sort_values(
            by=['Num_Acc', 'grav'],
            ascending=[True, True],
            key=lambda x: pd.Categorical(x, categories=ordre_gravite, ordered=True)
        )
        data_usagers_unique = data_usagers_sorted.drop_duplicates(subset=['Num_Acc'])
        dataframes_temp['usagers'] = data_usagers_unique

    # Fusion
    df_final = dataframes_temp['caracteristiques']

    # Fusion avec 'lieux' et 'usagers'
    for key in ['lieux', 'usagers']:
        if key in dataframes_temp:
            df_final = pd.merge(df_final, dataframes_temp[key], on='Num_Acc', how='left')

    # Fusion avec 'vehicules' en utilisant 'Num_Acc' et 'id_vehicule'
    if 'vehicules' in dataframes_temp:
        df_final = pd.merge(df_final, dataframes_temp['vehicules'], on=['Num_Acc', 'id_vehicule'], how='left')

    return df_final

df = load_and_process_data()

@st.cache_data
def load_and_preprocess_data():
        # Charger les données
        df = pd.read_csv('data/fusion_complete.csv')

        # Créer la variable AGE
        if 'an' in df.columns and 'an_nais' in df.columns:
            df['age'] = df['an'] - df['an_nais']

            # Définir les bins et les labels pour les catégories d'âge
            bins = [0, 5, 12, 18, 40, 60, float('inf')]
            labels = ['0-5 ans', '6-12 ans', '13-18 ans', '19-40 ans', '41-60 ans', 'plus de 60 ans']

            # Utiliser pd.cut pour catégoriser 'age'
            df['age_cat'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

            # Supprimer les colonnes originales
            df = df.drop(['an_nais', 'an', "age"], axis=1)

        return df


df_pre_processing = load_and_preprocess_data()

columns_to_keep = [
        'catr', 'circ', 'NBV_cat', "prof", 'plan', 'lartpc', 'larrout', 'surf_cat', 'situ', 'vma_c', 'place', 'catu',
        'sexe', 'secu1', 'nbr_usager_veh', 'avec_pieton', 'grav_Acc', 'nbr_vehicule', 'nbr_usager_acc', 'agg',
        'Moment_journée', 'Lum_regroupe', 'Intersection', 'Météo_Normale', 'Collision_binaire', 'senc',
        'motor', 'obs_fixe', 'obs_mobile', 'choc_initial', "age_cat", "cat_vehs", "long", "lat"]

# Filtrer les colonnes à conserver qui sont effectivement présentes dans le DataFrame
columns_to_keep = [col for col in columns_to_keep if col in df_pre_processing.columns]

# Créer df_model avec les colonnes filtrées
df_model = df_pre_processing[columns_to_keep]

def traitement_data(X, col_a_traiter):
        for col in col_a_traiter:
            mode_value = X[col].mode()[0]
            X[col] = X[col].replace([-1, pd.NA, np.nan], np.nan)
            X[col] = X[col].astype(str)
        return X

X = df_model.drop('grav_Acc', axis=1)
y = df_model['grav_Acc']

col_a_traiter = columns_to_keep.copy()
col_a_traiter.remove('grav_Acc')

X = traitement_data(X, col_a_traiter)

# Encodage de la variable cible
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Sous-échantillonnage
n_samples = int(np.min(np.bincount(y_encoded)))
sampling_strategy = {0: n_samples, 1: n_samples, 2: n_samples}
undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y_encoded)

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Pipeline pour les colonnes catégorielles
categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# Transformer les colonnes catégorielles
preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, col_a_traiter)
        ],
        remainder='drop'
    )

# Pipeline complet
pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42))
    ])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

import joblib

# Sauvegarde du pipeline
joblib.dump(pipeline, 'accident_severity_model.joblib')

# Prédictions
y_pred = pipeline.predict(X_test)

@st.cache_resource
def load_or_create_model(X=None, y=None):
    try:
        return joblib.load('accident_severity_model.joblib')
    except FileNotFoundError:
        if X is None or y is None:
            raise ValueError("Les données d'entraînement sont nécessaires pour créer le modèle.")

        # Code pour créer et entraîner le modèle
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_pipeline, X.columns)
            ],
            remainder='drop'
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42))
        ])

        pipeline.fit(X, y)
        joblib.dump(pipeline, 'accident_severity_model.joblib')
        return pipeline



# Création des pages et du sommaire
st.title("Projet sur les Accidents routiers en France")
st.sidebar.title("Sommaire")
pages=["Introduction","Exploration", "Pré-processing", "Modélisation" ,"Déploiement du modèle","Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("### Projet réalisé par :")
st.sidebar.markdown("""- Nathalie MORVANT
- Naima TALAHIK
- Gabriel CIFCI
- Stéphane ROY""")
st.sidebar.write("### Lien GitHub :")
st.sidebar.markdown("[Lien vers le projet sur GitHub](https://github.com/talahik/Road-Accidents-in-France)")

############### Création de la page Introduction #########################

if page == pages[0] :
  st.write("# Introduction")
  st.header("Description du projet")
  st.write("""L’objectif de ce projet est d’essayer de prédire la gravité des accidents routiers en France.
         Les prédictions seront basées sur les données historiques. Nous avons fait le choix de nous baser sur 4 années de
         2019 à 2022.""")
  st.write("""Une première étape est d'explorer et de comprendre les jeux de données. Une deuxième étape est
         de rendre les jeux de données propres (valeurs manquantes, doublons, etc..) afin de les exploiter pour la mise en place
        des modèles de machine learning. """)

  st.header("Sources des jeux de données")
  st.write("""Notre source principale de données pour répondre à la problématique est le fichier national des accidents
           corporels de la circulation dit « Fichier BAAC » administré par l’Observatoire national interministériel de
           la sécurité routière "ONISR" et que l'on peut trouver sur data.gouv.""")
  st.write("""Les bases de données, extraites du fichier BAAC, répertorient l'intégralité des
           accidents corporels de la circulation, intervenus durant une année précise en France métropolitaine, dans les DOM
           et  TOM avec une description simplifiée.""")

  st.write("""Ils comprennent des informations de localisation de l’accident, telles que renseignées ainsi que des informations
           concernant les caractéristiques de l’accident et son lieu, les véhicules impliqués et leurs victimes.""")

  st.write("""Les fichiers sont disponibles en open data. Pour mener notre analyse nous avons récupéré les bases de données annuelles
           de 2019 à 2022 composées de 4 fichiers (Caractéristiques – Lieux – Véhicules – Usagers) au format csv.""")

  st.write("""Le n° d'identifiant de l’accident ("Num_Acc") présent dans les 4 fichiers permet d'établir un lien entre toutes
           les variables qui décrivent un accident. Quand un accident comporte plusieurs véhicules, le lien entre le véhicule
           et ses occupants est fait par la variable id_vehicule.""")
  st.link_button("Bases de données annuelles des accidents corporels de la circulation routière - Années de 2005 à 2022", "https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/")

  st.header("Autres sources de données")

  st.write("""Nous avons cherché à utiliser d'autres sources de données pour notre analyse. Ainsi nous avons cherché à mettre
           en rapport les accidents de la route avec le trafic routier quotidien par département. Malheureusement seule l'année 2019 est disponible
           puisque le ministère des transport ne fournit plus ces données depuis. De plus nous nous sommes aperçus que le jeu de données
           fourni comportent des erreurs avec notamment un trafic routier très faible à Paris.""")

  st.write("""Nous avons également cherché à mettre en rapport le nombre d'habitants par département avec les accidents de la route.
           Nous nous sommes procurés toutes ces données sur le site de data.gouv. et le site de l'INSEE.""")
  st.link_button("Trafic routier 2019", "https://www.data.gouv.fr/fr/datasets/trafic-moyen-journalier-annuel-sur-le-reseau-routier-national/")
  st.link_button("Nombre d'habitants par département","https://www.insee.fr/fr/statistiques/1893198")


############### Création de la page Exploration ############################

if page == pages[1] :
    st.write("### Exploration des jeux de données")

    #### Exploration du jeus de données caractéristiques
    st.header("1. Analyse du jeu de données Caractéristiques")
    st.write("Le jeu de données comporte 15 colonnes. En concaténant les 4 années nous obtenons un jeu de données de 218 404 observations")

    st.markdown("On affiche quelques graphiques pour explorer et visualiser ce jeu de données")


    nb_accident_collision=fusion_carac.loc[-(fusion_carac["col"]==-1)].reset_index()
    nb_accident_collision["col"]=nb_accident_collision["col"].map({1:"2 véhicules Frontale",2:"2 véhicules Arrière",3:"2 véhicules côté",4:"3 véhicules et + en chaîne",\
                                                 5:"3 véhicules et + col mulptiples", 6: "Autre collision",7:"Sans colision"})

    fig = px.histogram(nb_accident_collision, x="col", color='an',text_auto="count", labels={"an": "Années"})

    fig.update_layout(title = "Accidents par type de collision et par année",
                  xaxis_title = 'Types de collision',
                  yaxis_title = "Nombre d'accidents",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)


    nb_accident_meteo=fusion_carac.loc[-(fusion_carac["atm"]==-1)].reset_index()
    nb_accident_meteo["atm"]=nb_accident_collision["atm"].map({1:"Normale",2:"Pluie Légère",3:"Pluie forte",4:"Neige_grêle",\
                                                 5:"Brouillard", 6: "Vent fort",7:"Eblouissant",8:"Couvert",9:"Autre"})

    fig = px.histogram(nb_accident_meteo, x="atm", color='an',text_auto="count",labels={"an": "Années"})

    fig.update_layout(title = "Nombre d'accidents par type de condition atmosphérique et par année",
                  xaxis_title = 'Conditions météos',
                  yaxis_title = "Nombre d'accidents",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    nb_accident_loca=fusion_carac.loc[-(fusion_carac["agg"]==-1)].reset_index()
    nb_accident_loca["agg"]=nb_accident_loca["agg"].map({1:"Hors Agglomération",2:"Agglomération"})

    fig = px.histogram(nb_accident_loca, x="agg", color='an',text_auto="count",labels={"an": "Années"})

    fig.update_layout(title = "Nombre d'accidents par localisation et par année",
                  xaxis_title = "Localisations",
                  yaxis_title = "Nombre",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    #on créé une colonne date en fusionnant l'année, le mois et le jour

    fusion_carac["date"]=fusion_carac.apply(lambda row: str(row["an"])+"-" + str(row["mois"])+"-" + str(row["jour"]) ,axis=1)
    # L'argument 'yearfirst=True' est utilisé pour indiquer que le format de la date est "AAAA-MM-JJ"
    fusion_carac["date"]=pd.to_datetime(fusion_carac["date"],yearfirst = True)

    #on compte le nombre d'accidents par date
    values=fusion_carac["date"].value_counts().sort_index().reset_index()
    nom={"count":"Nombre"}
    values=values.rename(nom,axis=1)

    fig = px.line(values,x = "date",y="Nombre")

    fig.add_scatter(x=values["date"], y=[np.median(values["Nombre"])]*len(values["date"]),mode='lines',name="Médiane",marker_color='rgba(255, 182, 193, .9)')


    fig.update_layout(title = "Nombre d'accidents par jour",
                  xaxis_title = "Nombre d'accidents par jour",
                  yaxis_title = "Nombre",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    fig = px.histogram(nb_accident_loca, x="mois", color='an',text_auto="count",labels={"an": "Années"})

    fig.update_layout(title = "Nombre d'accidents par mois et par année",
                  xaxis_title = "Mois",
                  yaxis_title = "Nombre",
                  barmode='group',
                  xaxis = dict(
                  tickvals = [1,2,3,4,5,6,7,8,9,10,11,12],
                  ticktext  = ["janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre"]))
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    #on compte le nombre d'accident par jour de la  semaine et par année
    values_par_jour=fusion_carac.groupby([fusion_carac["date"].dt.weekday,fusion_carac["an"]]).agg({"Num_Acc":"count"}).reset_index()
    nom={"Num_Acc":"Nombre"}
    values_par_jour=values_par_jour.rename(nom,axis=1)
    values_par_jour["date"]=values_par_jour["date"].map({0:"Lundi",1:"Mardi",2:"Mercredi",3:"Jeudi",4:"Vendredi",5:"Samedi",6:"Dimanche"})

    fig = px.histogram(values_par_jour, x="date",y="Nombre", color='an',text_auto="count",labels={"an": "Années"})

    fig.add_scatter(x=values_par_jour["date"], y=[np.median(values_par_jour["Nombre"])]*len(values_par_jour["date"]),mode='lines',name="Médiane")

    fig.update_layout(title = "Nombre d'accidents par jour de la semaine et par année",
                  xaxis_title = "jour",
                  yaxis_title = "Nombre",
                  barmode='group')
    st.plotly_chart(fig)

    fusion_carac['heure'] = fusion_carac['hrmn'].apply(lambda x : x[0:2]).astype("int")
    fig = px.histogram(fusion_carac, x="heure", color='an',text_auto="count",labels={"an": "Années"})


    fig.update_layout(title = "Nombre d'accidents par heure dans la journée et par année",
                  xaxis_title = "Heures dans la journée",
                  yaxis_title = "Nombre",
                  barmode='group')
    st.plotly_chart(fig)

    accident_2022=carac_2022.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2022["année"]=2022
    accident_2021=carac_2021.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2021["année"]=2021
    accident_2020=carac_2020.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2020["année"]=2020
    accident_2019=carac_2019.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2019["année"]=2019

    liste_accident=[accident_2019,accident_2020,accident_2021,accident_2022]
    accident_fusion=pd.concat(liste_accident,axis=0).reset_index().drop(columns="index")

    fig = px.box(accident_fusion,y="Num_Acc", x='année',hover_data=["dep"])

    fig.update_layout(title = "Distribution du nombre d'accidents par année selon le département",
                  xaxis_title = "Années",
                  yaxis_title = "Nombre d'accidents")
    st.plotly_chart(fig)

    fusion_carac_usagers=fusion_carac.merge(right=fusion_usagers,on="Num_Acc",how="left")
    fusion_carac_usagers["agg"]=fusion_carac_usagers["agg"].map({1:"Hors Agglomération",2:"Agglomération"})

    fusion_carac_usagers["grav"]=fusion_carac_usagers["grav"].map({1:"Indemne",2:"Tué",3:"Bléssé hospitalisé",4:"Bléssé léger"})

    fig = px.histogram(fusion_carac_usagers, x="grav", color='agg',text_auto="count",animation_frame="an",labels={"agg": "Localisation"})

    fig.update_layout(title = "Accidents par gravité et en fonction de la localisation",
                  xaxis_title = "Gravité",
                  yaxis_title = "Nombre d'accidents",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')

    st.plotly_chart(fig)





    for feature in departement['features']:
        feature['id']= feature['properties']['code']

       #avec ce bout de code on récupère dans un dictionnaire le Numéro du département et son nom.
    dico_dep={}
    for feature in departement['features']:
        dico_dep[feature['properties']['code']]=feature['properties']['nom']

    df_département = pd.DataFrame(dico_dep.items(), columns=['Département', 'Nom'])

    liste_outre_mer=["971","972","973","974",'977', '978', '975',"976","987","988","986"]
    fusion_carac_dep=fusion_carac[~fusion_carac["dep"].isin(liste_outre_mer)]

    dep_2022=fusion_carac_dep.loc[fusion_carac_dep["an"]==2022]["dep"].value_counts().to_frame().reset_index()
    dep_2022["echelle_accident"]=np.log10(dep_2022['count'])
    dep_2022["an"]=2022


    dep_2021=fusion_carac_dep.loc[fusion_carac_dep["an"]==2021]["dep"].value_counts().to_frame().reset_index()
    dep_2021["echelle_accident"]=np.log10(dep_2021['count'])
    dep_2021["an"]=2021


    dep_2020=fusion_carac_dep.loc[fusion_carac_dep["an"]==2020]["dep"].value_counts().to_frame().reset_index()
    dep_2020["echelle_accident"]=np.log10(dep_2020['count'])
    dep_2020["an"]=2020


    dep_2019=fusion_carac_dep.loc[fusion_carac_dep["an"]==2019]["dep"].value_counts().to_frame().reset_index()
    dep_2019["echelle_accident"]=np.log10(dep_2019['count'])
    dep_2019["an"]=2019



    dep_tous=pd.concat((dep_2022,dep_2021,dep_2020,dep_2019),axis=0)


    dico={"count":"Nombre_accidents","dep":"Département", "an":"Année"}
    dep_tous=dep_tous.rename(dico,axis=1)
    dep_tous["Département"]=dep_tous["Département"].astype("str")
    dep_tous=dep_tous.merge(right=df_département,on="Département",how="left")
    dep_tous["Num_Nom"]=dep_tous["Département"]+ " "+ dep_tous["Nom"]

    fig3 = px.choropleth_mapbox(dep_tous, locations = 'Département',
                            geojson= departement,
                            color='echelle_accident',
                            color_continuous_scale=["green","orange","red"],
                            range_color=[2,3.5],
                            hover_name="Num_Nom",
                            animation_frame='Année',
                            hover_data=['Nombre_accidents'],
                            title="Carte de répartition des accidents en France et par année",
                            mapbox_style="open-street-map",
                            center= {'lat':46.5, 'lon':2},
                            zoom =4.5,
                            opacity= 0.6, width=5000, height=700)

    st.plotly_chart(fig3)
    
    #### Exploration du jeux de données Lieux
    st.header("2. Analyse du jeu de données Lieux")

    # Définition des mappings
    catr_mapping = {
        1: 'Autoroute', 2: 'Route nationale', 3: 'Route départementale',
        4: 'Voie communale', 5: 'Hors réseau public',
        6: 'Parc de stationnement ouvert à la circulation publique',
        7: 'Route de métropole urbaine', 9: 'autre'
    }

    circ_mapping = {
        -1: 'Non renseigné', 1: 'A sens unique', 2: 'Bidirectionnelle',
        3: 'A chaussées séparées', 4: 'Avec voies d affectation variable',
    }

    vosp_mapping = {
        -1: 'Non renseigné', 0: 'Sans objet', 1: 'Piste cyclable',
        2: 'Bande cyclable', 3: 'Voie réservée',
    }

    prof_mapping = {
        -1: 'Non renseigné', 1: 'Plat', 2: 'Pente',
        3: 'Sommet de côte', 4: 'Bas de côte',
    }

    plan_mapping = {
        -1: 'Non renseigné', 1: 'Partie rectiligne', 2: 'En courbe à gauche',
        3: 'En courbe à droite', 4: 'En S',
    }

    surf_mapping = {
        -1: 'Non renseigné', 1: 'Normal', 2: 'Mouillée', 3: 'Flaques',
        4: 'Inondée', 5: 'Enneigée', 6: 'Boue', 7: 'Verglacée',
        8: 'Corps gras – huile', 9: 'Autre'
    }

    infra_mapping = {
        -1: 'Non renseigné', 0: 'Aucun', 1: 'Souterrain-tunnel', 2: 'Pont',
        3: 'Bretelle d échangeur ou de raccordement', 4: 'Voie ferrée',
        5: 'Carrefour aménagé', 6: 'Zone piétonne', 7: 'Zone de péage',
        8: 'Chantier', 9: 'Autre'
    }

    grav_mapping = {
        1: 'Indemne', 2: 'Tué', 3: 'Blessé hospitalisé', 4: 'Blessé léger'
    }



    # Visualisation pour catr
    st.header("Catégorie de route")
    catr_an_compte = fusion_lieux_usagers.groupby(['catr', 'an']).size().reset_index(name='Nombre accidents')
    catr_an_compte['catr'] = catr_an_compte['catr'].map(catr_mapping)
    fig_catr = px.bar(catr_an_compte, x='catr', y='Nombre accidents', color='an', barmode='group',
                      title='Nombre d\'accidents par catégorie de route et par an')
    st.plotly_chart(fig_catr)

    # Visualisation pour circ
    st.header("Régime de circulation")
    circ_an_compte = fusion_lieux_usagers.groupby(['circ', 'an']).size().reset_index(name='Nombre accidents_circ')
    circ_an_compte['circ'] = circ_an_compte['circ'].map(circ_mapping)
    fig_circ = px.bar(circ_an_compte, x='circ', y='Nombre accidents_circ', color='an', barmode='group',
                      title='Nombre d\'accidents par régime de circulation et par an')
    st.plotly_chart(fig_circ)

    # Visualisation pour vosp
    st.header("Voie spéciale")
    vosp_an_compte = fusion_lieux_usagers.groupby(['vosp', 'an']).size().reset_index(name='Nombre accidents_vosp')
    vosp_an_compte['vosp'] = vosp_an_compte['vosp'].map(vosp_mapping)
    fig_vosp = px.bar(vosp_an_compte, x='vosp', y='Nombre accidents_vosp', color='an', barmode='group',
                      title='Nombre d\'accidents par signal voie reservée')
    st.plotly_chart(fig_vosp)

    # Visualisation pour prof
    st.header("Profil de la route")
    prof_an_compte = fusion_lieux_usagers.groupby(['prof', 'an']).size().reset_index(name='Nombre accidents_prof')
    prof_an_compte['prof'] = prof_an_compte['prof'].map(prof_mapping)
    fig_prof = px.bar(prof_an_compte, x='prof', y='Nombre accidents_prof', color='an', barmode='group',
                      title='Nombre d\'accidents par dénivelé')
    st.plotly_chart(fig_prof)

    # Visualisation pour plan
    st.header("Tracé en plan")
    plan_an_compte = fusion_lieux_usagers.groupby(['plan', 'an']).size().reset_index(name='Nombre accidents_plan')
    plan_an_compte['plan'] = plan_an_compte['plan'].map(plan_mapping)
    fig_plan = px.bar(plan_an_compte, x='plan', y='Nombre accidents_plan', color='an', barmode='group',
                      title='Nombre d\'accidents par plan et par an')
    st.plotly_chart(fig_plan)

    # Visualisation pour surf
    st.header("État de la surface")
    surf_an_compte = fusion_lieux_usagers.groupby(['surf', 'an']).size().reset_index(name='Nombre accidents_surf')
    surf_an_compte['surf'] = surf_an_compte['surf'].map(surf_mapping)
    fig_surf = px.bar(surf_an_compte, x='surf', y='Nombre accidents_surf', color='an', barmode='group',
                      title='Nombre d\'accidents par état de surface et par an')
    st.plotly_chart(fig_surf)

    # Visualisation pour infra
    st.header("Infrastructure")
    infra_an_compte = fusion_lieux_usagers.groupby(['infra', 'an']).size().reset_index(name='Nombre accidents_infra')
    infra_an_compte['infra'] = infra_an_compte['infra'].map(infra_mapping)
    fig_infra = px.bar(infra_an_compte, x='infra', y='Nombre accidents_infra', color='an', barmode='group',
                      title='Nombre d\'accidents par infrastructure et par an')
    st.plotly_chart(fig_infra)

    # Visualisation pour vma
    st.header("Vitesse maximale autorisée")
    fusion_lieux_usagers['grav_mapped'] = fusion_lieux_usagers['grav'].map(grav_mapping)
    vma_an_grav_compte = fusion_lieux_usagers.groupby(['vma', 'an', 'grav_mapped']).size().reset_index(name='Nombre accidents')
    pivot_table = vma_an_grav_compte.pivot_table(index=['vma', 'an'], columns='grav_mapped', values='Nombre accidents', fill_value=0)
    pivot_table_mean = pivot_table.groupby('vma').mean()

    scaler = MinMaxScaler()
    pivot_table_mean_normalized = pd.DataFrame(scaler.fit_transform(pivot_table_mean),
                                              columns=pivot_table_mean.columns,
                                              index=pivot_table_mean.index)

    fig_vma = px.line(pivot_table_mean_normalized.reset_index(), x='vma', y=pivot_table_mean_normalized.columns,
                      title='Nombre moyen d\'accidents par VMA et par catégorie de gravité (Normalisé)')
    fig_vma.update_layout(xaxis_title='VMA', yaxis_title='Nombre moyen d\'accidents (Normalisé)',
                          legend_title='Gravité')
    fig_vma.update_xaxes(range=[0, 200])
    st.plotly_chart(fig_vma)



#### exploration des fichiers USAGERS
    st.header("3. Analyse du jeu de données Usagers")
    st.write("""Les fichiers usagers listent l’ensemble des personnes impliquées dans un accident (conducteur, passager ou piéton), et renseignent la variable Gravité de blessure de l'usager.
                Les usagers accidentés sont classés en 4 catégories : Indemne, Tué, Blessé hospitalisé, Blessé léger.""")
    st.write("""Le jeu de données comporte 15 colonnes qualifiant 494 018 personnes impliquées dans un accident routier entre 2019 et 2022 (après suppression de 164 doublons).""")

    st.markdown("On affiche quelques graphiques pour explorer et visualiser ce jeu de données")

    st.subheader("Gravité des accidents")
    fusion_usagers["grav"]=fusion_usagers["grav"].map({1:"Indemne",2:"Tué",3:"Blessé hospitalisé",4:"Blessé léger"})

    fig = px.histogram(fusion_usagers, x="grav", text_auto="count")
    fig.update_layout(title = "Nombre d'usagers selon la gravité de l'accident",
                    xaxis_title = "Gravité de l'accident",
                    yaxis_title = "Nombre",
                    barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    st.markdown("""Selon le graphique ci-dessus, près de 42% sont des personnes indemnes et 58% sont des victimes.""")


    st.subheader("Catégorie de l'usager")
    fusion_usagers["catu"]=fusion_usagers["catu"].map({1:"Conducteur",2:"Passager", 3:"Piéton"})

    fig1 = px.pie(fusion_usagers, names="catu", title="Répartition des usagers par catégorie")
    st.plotly_chart(fig1, use_container_width=True)

    fusion_usagers["sexe"]=fusion_usagers["sexe"].map({1:"Masculin",2:"Féminin"})

    fig2 = px.histogram(fusion_usagers, x="catu", color="sexe", text_auto="count", labels={"sexe": "Sexe"})
    fig2.update_layout(title = "Sexe des usagers selon leur catégorie",
                  xaxis_title = "Catégorie de l'usager",
                  yaxis_title = "Nombre d'usagers",
                  barmode='group')
    fig2.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig2)

    temp=fusion_usagers.loc[fusion_usagers['age']!='NaN'].reset_index()
    temp=temp.loc[temp['sexe']!='NaN'].reset_index()
    #temp["grav"]=temp["grav"].map({1:"Indemne",2:"Tué",3:"Blessé hospitalisé",4:"Blessé léger"})
    #temp["sexe"]=temp["sexe"].map({1:"Masculin",2:"Féminin"})
    #temp["catu"]=temp["catu"].map({1:"Conducteur",2:"Passager", 3:"Piéton"})

    temp_grouped1 = temp.groupby(['catu','sexe'])['age'].mean().reset_index()
    fig1 = sns.catplot(x="catu", y="age", kind ='point', hue='sexe', data=temp_grouped1);
    plt.xlabel("gravité de l'accident")
    plt.ylabel("âge")
    plt.title("Age moyen selon la catégorie de l'usager et son sexe")
    st.pyplot(fig1)

    catu_normalized_by_grav = pd.crosstab(fusion_usagers["catu"], fusion_usagers["grav"], normalize='index') * 100
    fig3 = plt.figure(figsize=(6, 6))
    heatmap_normalized_by_gravity = sns.heatmap(catu_normalized_by_grav, annot=True, fmt=".1f", cmap="YlOrRd")
    heatmap_normalized_by_gravity.set_title("Pourcentage de la gravité des blessures selon la catégorie de l'usager")
    heatmap_normalized_by_gravity.set_xlabel("Gravité des blessures")
    heatmap_normalized_by_gravity.set_ylabel("Catégorie de l'usager")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

    temp_grouped2 = temp.groupby(['grav','catu'])['age'].mean().reset_index()
    fig2 = sns.catplot(x="catu", y="age", kind ='point', hue='grav', data=temp_grouped2);
    fig2._legend.set_title('Gravité des blessures')
    plt.xlabel("gravité de l'accident")
    plt.ylabel("âge")
    plt.title("Age moyen selon la catégorie de l'usager et la gravité de ses blessures")
    st.pyplot(fig2)


    st.subheader("Equipement de sécurité")
    #fig = plt.figure(figsize=(3, 3))
    #sns.countplot(data=fusion_usagers,x="secu1",order=fusion_usagers['secu1'].value_counts(ascending=False).index)
    #plt.xlabel("1er équipement de sécurité")
    #plt.ylabel("Nombre")
    #plt.title("Répartition du type de 1er équipement de sécurité")
    #plt.xticks(range(11),["Ceinture","Casque","Non déterminable","Aucun équipement","NR","Dispositif enfant","Autre","Gants","Gilet réfléchissant","Airbag","Gants+Airbag"],rotation=45)
    #st.pyplot(fig)

    fusion_usagers["secu1"]=fusion_usagers["secu1"].map({1:"Ceinture",2:"Casque",8:"Non déterminable",0:"Aucun équipement",-1:"NR",3:"Dispositif enfant",9:"Autre",6:"Gants",4:"Gilet réfléchissant",5:"Airbag",7:"Gants+Airbag"})
    fig = px.histogram(fusion_usagers, x="secu1", text_auto="count")
    fig.update_layout(title = "Répartition du type de 1er équipement de sécurité",
                    xaxis_title = "1er équipement de sécurité",
                    yaxis_title = "Nombre",
                    barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)


    st.subheader("Focus sur les piétons et l’analyse des données relatives")
    data_pietons=fusion_usagers.loc[fusion_usagers['catu']=='Piéton'].reset_index()
    #data_pietons["grav"]=data_pietons["grav"].map({1:"Indemne",2:"Tué",3:"Blessé hospitalisé",4:"Blessé léger"})

    conditions_list=[(data_pietons['actp']=='1'), (data_pietons['actp']=='2'), (data_pietons['actp']=='3'), (data_pietons['actp']=='4'), (data_pietons['actp']=='5'), (data_pietons['actp'].isin(['6','7','8','9','A']))]
    choice_list=["se déplaçant sens véhic", "se déplaçant sens inverse véhic", "traversant","masqué","courant-jouant","autre"]
    data_pietons['actp_g']=np.select(conditions_list, choice_list, default="NR ou sans objet")
    fig1 = px.histogram(data_pietons, x="grav", color='actp_g',text_auto="count",labels={"actp_g": "Action du piéton"})
    fig1.update_layout(title = "Action des piétons au moment de l'accident par gravité des blessures",
                          xaxis_title = "Gravité de l'accident",
                          yaxis_title = "Nombre d'usagers",
                          barmode='group')
    fig1.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig1)

    conditions_list=[(data_pietons['locp'] <1), (data_pietons['locp'] >0) & (data_pietons['locp'] <3), (data_pietons['locp'] >2) & (data_pietons['locp'] <5), (data_pietons['locp'] >4)]
    choice_list=["NR ou sans objet", "sur chaussée", "sur passage piéton", "divers"]
    data_pietons['locp_g']=np.select(conditions_list, choice_list, default="Not specified")
    locp_normalized_by_grav = pd.crosstab(data_pietons['locp_g'], data_pietons['grav'], normalize='index') * 100
    fig2 = plt.figure(figsize=(6, 4))
    heatmap_normalized_by_gravity = sns.heatmap(locp_normalized_by_grav, annot=True, fmt=".1f", cmap="YlOrRd")
    heatmap_normalized_by_gravity.set_title("Pourcentage de la gravité des blessures selon la localisation du piéton")
    heatmap_normalized_by_gravity.set_xlabel("Gravité des blessures")
    heatmap_normalized_by_gravity.set_ylabel("Localisation du piéton")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    #### Exploration du jeux de données vehicules
    st.header("3. Analyse du jeu de données Vehicules")
    #Mapping des catégories de vehicules
    catv_mapping = {
    1: "Bicyclette",
    2: "Cyclomoteur <50cm3",
    3: "Voiturette",
    7: "Voiture (VL seul)",
    10: "Véhicule Utilitaire (VU seul)",
    13: "Poids Lourd 3,5T < PTCA <= 7,5T",
    14: "Poids Lourd > 7,5T",
    16: "Tracteur Routier seul",
    17: "Tracteur Routier + Semi-Remorque",
    20: "Engin spécial",
    21: "Tracteur Agricole",
    30: "Scooter < 50 cm3",
    31: "Motocyclette > 50 cm3 et <= 125 cm3",
    32: "Scooter > 50 cm3 et <= 125 cm3",
    33: "Motocyclette > 125 cm3",
    34: "Scooter > 125 cm3",
    37: "Autobus",
    38: "Autocar",
    80: "Vélo à Assistance Électrique (VAE)",
    99: "Autre véhicule"
    }

    # Mapping des manœuvres 
    manv_mapping = {
    1: "Pas de changement",
    2: "Pas de changement",
    3: "Pas de changement",
    4: "Pas de changement",
    9: "Changement de direction ou de file",
    11: "Changement de direction ou de file",
    12: "Changement de direction ou de file",
    15: "Changement de direction ou de file",
    16: "Changement de direction ou de file",
    17: "Changement de direction ou de file",
    18: "Changement de direction ou de file",
    6: "Manœuvres spéciales ou d'arrêt",
    7: "Manœuvres spéciales ou d'arrêt",
    8: "Manœuvres spéciales ou d'arrêt",
    20: "Manœuvres spéciales ou d'arrêt",
    21: "Manœuvres spéciales ou d'arrêt",
    22: "Manœuvres spéciales ou d'arrêt",
    23: "Manœuvres spéciales ou d'arrêt",
    24: "Manœuvres spéciales ou d'arrêt",
    25: "Manœuvres spéciales ou d'arrêt",
    26: "Manœuvres spéciales ou d'arrêt",
    5: "Déplacement à contresens",
    -1: "Inconnu ou non renseigné",
    0: "Inconnu ou non renseigné"
    }
    # Mapping des points de choc 
    choc_mapping = {
    1: "Avant",
    2: "Avant",
    3: "Avant",
    4: "Arrière",
    5: "Arrière",
    6: "Arrière",
    7: "Côté",
    8: "Côté",
    9: "Chocs multiples (tonneaux)",
    -1: "Aucun/Non renseigné",
    0: "Aucun/Non renseigné"
    }



    # Appliquer le mapping catv_mapping
    fusion_vehicules_usagers["catv"] = fusion_vehicules_usagers["catv"].map(catv_mapping)
    # Appliquer le mapping manv_mapping
    fusion_vehicules_usagers["manv"] = fusion_vehicules_usagers["manv"].map(manv_mapping)
    # Appliquer le mapping choc
    fusion_vehicules_usagers["choc_initial"] = fusion_vehicules_usagers["choc"].map(choc_mapping)

    fig_catv = px.histogram(fusion_vehicules_usagers, x="catv", text_auto="count",
                            labels={"catv_simplified": "Catégories de véhicules"})
    fig_catv.update_layout(title="Répartition des Catégories de Véhicules impliquées dans les Accidents",
                            xaxis_title="Catégorie de Véhicule",
                            yaxis_title="Nombre d'Accidents",
                            barmode='group')
    fig_catv.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_catv)

    fig_manv = px.histogram(fusion_vehicules_usagers, x="manv", text_auto="count",
                    labels={"manv_super_simplified": "Manœuvre principale "})
    fig_manv.update_layout(title="Répartition des Manœuvres Principales des Véhicules avant l'Accident",
                   xaxis_title="Manœuvre",
                   yaxis_title="Nombre d'Accidents",
                   barmode='group')
    fig_manv.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_manv)
    
    fig_choc = px.histogram(fusion_vehicules_usagers, x="choc_initial", text_auto="count",
                            labels={"choc_initial_simplified": "Point de Choc Initial "})
    fig_choc.update_layout(title="Répartition des Points de Choc Initiaux des Véhicules ",
                           xaxis_title="Point de Choc",
                           yaxis_title="Nombre d'Accidents",
                           barmode='group')
    fig_choc.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_choc)

    # Appliquer le mapping
    fusion_vehicules['vehicule_simplified'] = fusion_vehicules['catv'].map(catv_mapping)
    # Fusion avec les données usagers pour obtenir la gravité des accidents
    fusion_vehicules_usagers = pd.merge(fusion_vehicules, fusion_usagers[['Num_Acc', 'grav']], on='Num_Acc', how='left')
    # Créer un graphique pour la répartition de la gravité d'accident par type de véhicule simplifié
    fig_gravite_vehicule = px.histogram(
        fusion_vehicules_usagers, 
        x="vehicule_simplified", 
        color='grav',
        text_auto="count", 
        labels={"grav": "Gravité de l'Accident"},
        category_orders={"grav": ["Indemne", "Blessé léger", "Blessé hospitalisé", "Tué"]}
        )

    fig_gravite_vehicule.update_layout(
        title="Répartition de la Gravité des Accidents selon le Type de Véhicule ",
        xaxis_title="Type de Véhicule",
        yaxis_title="Nombre d'Accidents",
        barmode='stack'
        )
    
    fig_gravite_vehicule.update_xaxes(categoryorder='total descending',
                                      tickangle=-45)
    st.plotly_chart(fig_gravite_vehicule)



############### Création de la page Préprocessing #########################

if page == pages[2]:
    st.write("# Preprocessing")
    st.write("""Avant l'étape d'entraînement du modèle, nous avons travaillé à la préparation des données : structuration du dataset et nettoyage des données.""")

    st.header("1. Structuration des données")

    st.subheader("Fusion des jeux de données")

    st.write("""Les données que l'on souhaite utiliser se trouvent dans 4 sources de données (caractéristiques, lieux, véhicules, usagers) que nous avons donc rassembler à l'aide de jointures. """)

    st.subheader("Traitement de la variable cible")

    st.write("""Nous avons construit notre variable cible à partir de la variable ‘gravité' du jeu de données USAGERS, qui renseigne la gravité des blessures de chaque personne impliquée dans un accident de la route.""")
    st.write("""Notre variable cible (à prédire par le modèle) qualifie la gravité d'un accident selon 3 valeurs :""")
    st.write("""• Un accident LÉGER est un accident corporel comptant au moins un blessé léger (hospitalisé moins de 24h), aucun blessés hospitalisés (plus de 24h) et aucun tué (jusqu'à 30 jours après l'accident)""")
    st.write("""• Un accident GRAVE non mortel est un accident corporel comptant au moins un blessé hospitalisé (plus de 24h) et aucun tué (jusqu'à 30 jours après l'accident).""")
    st.write("""• Un accident MORTEL est un accident corporel comptant au moins un tué (jusqu'à 30 jours après l'accident).""")

    st.write("""Par exemple : un accident comptant 1 blessé léger et 2 blessés hospitalisés, est un accident GRAVE non mortel.""")
    st.write("""Il s’agit d’une variable qualitative ordinale.""")

    # Gravité des accidents
    temp = fusion_usagers.loc[:, ['Num_Acc', 'grav']]
    temp['tue'] = np.where(temp['grav'] == 2, 1, 0)
    temp['blesse_leger'] = np.where(temp['grav'] == 4, 1, 0)
    temp['blesse_hospitalise'] = np.where(temp['grav'] == 3, 1, 0)
    grav_groupby = temp.groupby('Num_Acc').agg({'tue': 'sum', 'blesse_leger': 'sum', 'blesse_hospitalise': 'sum'})

    conditionlist = [
        (grav_groupby['tue'] > 0),
        (grav_groupby['blesse_hospitalise'] > 0) & (grav_groupby['tue'] == 0),
        (grav_groupby['blesse_leger'] > 0) & (grav_groupby['blesse_hospitalise'] == 0) & (grav_groupby['tue'] == 0)
    ]
    choicelist = ["mortel", "grave", "leger"]
    grav_groupby["grav_Acc"] = np.select(conditionlist, choicelist, default="NR")

    grav_counts = grav_groupby["grav_Acc"].value_counts().reset_index()
    grav_counts.columns = ['gravite', 'count']
    fig_severity = px.bar(grav_counts, x="gravite", y="count",
                          title="Distribution de la gravité des accidents")
    st.plotly_chart(fig_severity, use_container_width=True)

    st.subheader("Nouvelles variables explicatives")

    st.write("""A partir des données brutes, nous avons construit 3 nouvelles variables relatives à un accident :""")
    st.write("""• Nombre de véhicules par accident""")
    st.write("""• Nombre d'usagers par accident""")
    st.write("""• Présence de piétons dans un accident (oui/non)""")

    # Nombre de véhicules par accident
    temp = fusion_usagers.loc[:, ['Num_Acc', 'id_vehicule']]
    temp = temp.drop_duplicates(keep='first')
    count_vehicule = temp.groupby('Num_Acc').agg({'id_vehicule': 'count'}).reset_index()
    count_vehicule = count_vehicule.rename(columns={'id_vehicule': 'nbr_vehicule'})

    # Nombre d'usagers par accident
    count_usager = fusion_usagers["Num_Acc"].value_counts().sort_index().reset_index()
    count_usager = count_usager.rename(columns={'count': 'nbr_usager_acc'})

    # Présence de piétons
    temp = fusion_usagers.loc[:, ['Num_Acc', 'catu']]
    accidents_avec_pietons = temp[temp['catu'] == 3]['Num_Acc'].unique()

    total_accidents = fusion_usagers['Num_Acc'].nunique()
    accidents_avec_pietons_count = len(accidents_avec_pietons)
    accidents_sans_pietons_count = total_accidents - accidents_avec_pietons_count

    pieton_data = pd.DataFrame({
        'categorie': ['Avec piéton', 'Sans piéton'],
        'nombre': [accidents_avec_pietons_count, accidents_sans_pietons_count]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig_vehicles = px.histogram(count_vehicule, x="nbr_vehicule",
                                    title="Distribution du nombre de véhicules par accident")
        st.plotly_chart(fig_vehicles, use_container_width=True)

    with col2:
        fig_users = px.histogram(count_usager, x="nbr_usager_acc",
                                 title="Distribution du nombre d'usagers par accident")
        st.plotly_chart(fig_users, use_container_width=True)

    fig_pedestrians = px.pie(pieton_data, names="categorie", values="nombre",
                         title="Proportion d'accidents avec piéton")
    st.plotly_chart(fig_pedestrians, use_container_width=True)



    # Chemin d'accès aux fichiers
    file_paths = {
        'Caractéristiques 2022':'data/caracteristiques-2022.csv',
        'Caractéristiques 2021': 'data/caracteristiques-2021.csv',
        'Caractéristiques 2020': 'data/caracteristiques-2020.csv',
        'Caractéristiques 2019': 'data/caracteristiques-2019.csv',
        'Usagers 2022': 'data/usagers-2022.csv',
        'Usagers 2021': 'data/usagers-2021.csv',
        'Usagers 2020': 'data/usagers-2020.csv',
        'Usagers 2019': 'data/usagers-2019.csv'
}

####################################################################################################


    st.header("2. Gestion des valeurs manquantes")

    liste_col_valeur_manquantes = [col for col in df.columns if (df[col] == -1).any()]
    #st.write("Colonnes contenant des valeurs '-1 - Non renseigné' :")
    #st.write(liste_col_valeur_manquantes)

    for col in liste_col_valeur_manquantes:
        df[col] = df[col].replace(-1, np.nan)

    missing_percentages = (df.isnull().sum() / len(df)) * 100
    fig = px.bar(x=missing_percentages.index, y=missing_percentages.values,
                labels={'x': 'Colonnes', 'y': 'Pourcentage de valeurs manquantes'},
                title="Pourcentage de valeurs manquantes par colonne")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    st.write("""Après analyse des taux de valeurs manquantes de chaque variable, nous avons choisi de les remplacer par le mode.""")
    st.write("""Les variables avec plus de 80% de valeurs manquantes ont été supprimées.""")

    st.header("3. Recodage des variables")

    st.write("""Nous avons aussi transformé certaines variables en procédant à des regroupements de modalités pour plus de représentativité des données et de simplicité dans l'interprétation du modèle.""")

    st.subheader("Exemple de la variable : Catégorisation des véhicules")

    st.write("""La donnée brute Catégorisation du véhicule possède plus de 30 valeurs possibles. Nous avons choisi de regrouper les modalités en 7 types de véhicules.""")

    #@st.cache_data
    def cat(cat_vehs):
        if pd.isna(cat_vehs) or cat_vehs == -1:
            return "Non renseigné"
        if cat_vehs == 0:
            return "Engin spécial"
        if cat_vehs in [2, 4, 5, 6, 30, 31, 32, 33, 34, 41, 42, 43]:
            return "Deux Roues"
        elif cat_vehs in [3, 7, 8, 9, 10, 11, 12]:
            return "Véhicule léger"
        elif cat_vehs in [13, 14, 15, 16, 17]:
            return "Poids lourd"
        elif cat_vehs in [18, 19, 37, 38, 39, 40]:
            return "Transport en commun"
        elif cat_vehs == 1:
            return "Vélo"
        else:
            return "Autre"

    df["cat_vehs"] = df["catv"].apply(cat)
    st.write("Distribution des catégories de véhicules :")
    st.bar_chart(df_pre_processing["cat_vehs"].value_counts())


    st.subheader("Exemple du traitement des variables temporelles")

    st.write("""Les données nous renseignent l'heure et les minutes de l'accident. A partir de celles-ci, nous avons construit une variable qui renseigne le moment de la journée selon : matin, après-midi, soir, nuit.""")

    df['hrmn'] = df['hrmn'].astype(str)
    df['hrmn'] = df['hrmn'].str.zfill(4)

    df['heure'] = df['hrmn'].apply(lambda x : x[0:2]).astype("int")

    def cat_heures(cat_heure):
        if cat_heure >= 0 and cat_heure < 6:
            return "Nuit"
        elif cat_heure >= 6 and cat_heure < 12:
            return "Matin"
        elif cat_heure >= 12 and cat_heure < 18:
            return "Après-midi"
        else:
            return "Soir"

    df["Moment_journée"] = df['heure'].apply(cat_heures)
    st.write("Distribution des moments de la journée :")
    st.bar_chart(df["Moment_journée"].value_counts())


    st.subheader("Catégorisation de l'âge")

    st.write("""Nous avons discrétisé la variable âge en 7 catégories :""")

    def categorize_age(age):
        if pd.isna(age):
            return "Non renseigné"
        elif age < 0 or age > 120:  # Vérification de validité
            return "Invalide"
        elif age <= 5:
            return "0-5 ans"
        elif age <= 12:
            return "6-12 ans"
        elif age <= 18:
            return "13-18 ans"
        elif age <= 40:
            return "19-40 ans"
        elif age <= 60:
            return "41-60 ans"
        else:
            return "plus de 60 ans"

    df['age'] = df['an'] - df['an_nais']
    df['age_cat'] = df['age'].apply(categorize_age)

    st.write("Distribution des catégories d'âge :")
    st.bar_chart(df['age_cat'].value_counts())


    st.success("Prétraitement terminé ! Les données sont maintenant prêtes pour la modélisation.")

    st.write("""
        Ces opérations de prétraitement nous ont permis de structurer et de nettoyer les données,
        réduisant la complexité du jeu de données tout en préservant les informations essentielles
        pour l'analyse et la modélisation ultérieures.
        L'ensemble de nos variables explicatives sont des variables catégorielles que nous avons ensuite réencodées, les algorithmes de ML ne prennant en entrée que des variables numériques.
        """)


############### Création de la Page Modélisation #########################

if page == pages[3]:  
    st.write("### Méthodologie de modélisation + performance")
    # Fonction pour charger et prétraiter les données
    ##### j'ai enlevé le truc pour un test

    # Charger et prétraiter les données

    st.title("Modèle de Prédiction de la Gravité des Accidents")

    st.header("1. Présentation du Projet")

    st.write("""Ce projet vise à prédire la gravité des accidents de la route en utilisant des techniques de machine learning.""")
    st.write("""Pour rappel, notre variable cible est une variable qualitative ordinale : Mortel, Grave, Léger.""")
    st.write("""Pour cette partie de modélisation, nous sommes donc sur un cas de classification. Plus précisément, un cas de classification multi-classes.
    Nous avons testé plusieurs algorithmes, notamment la Régression logistique, Random Forest, Arbre de décision, KNN, Bagging et Boosting.""")

    st.write("Aperçu des données utilisées :")
    st.dataframe(X_train.head())
    st.write(f"Nombre total d'observations : {len(df_model)}")

    st.write("""
    Rappel de la représentativité de la gravité des 218 404 accidents entre 2019 et 2022 :
    - 64,3% dits LÉGER
    - 30,1% dits GRAVE
    - 5,6% dits MORTEL
    """)

    st.write("""Pour faire face à la forte disparité entre les classes à prédire, nous avons procédé à un rééquilibrage.""")
    st.write("""
    Nous avons testé différentes techniques de rééquilibrage des classes :
    - Oversampling aléatoire
    - Oversampling SMOTE
    - Undersampling aléatoire
    - Class_weight='balanced'

    Nous avons observé des résultats quasi similaires quelle que soit la technique utilisée.
    Pour ce modèle, nous utilisons l'undersampling de la classe majoritaire.
    """)

    st.write(f"Nombre d'observations après sous-échantillonnage : {len(X_resampled)}")


    st.header("2. Modélisation")

    st.write("""
    Nous avons choisi le modèle d'arbre de décision pour les raisons suivantes :
    1. Performance : L'arbre de décision a montré des performances satisfaisantes sur notre jeu de données équilibré.
    2. Interprétabilité : Il offre une grande transparence dans son processus de décision, crucial pour l'analyse des accidents de la route.
    3. Capacité à gérer des données mixtes : Notre jeu de données contient à la fois des variables catégorielles et numériques.
    """)

    st.header("3. Évaluation du Modèle")

    # Matrice de confusion
    st.subheader("Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    st.pyplot(fig)
    
    # Rapport de classification
    st.subheader("Rapport de Classification")
    cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    st.table(cr_df)
    
    # Extraire l'importance des caractéristiques
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = pipeline.named_steps['classifier'].feature_importances_

    # Créer un DataFrame pour l'importance des caractéristiques
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    # Afficher le top 15 des caractéristiques les plus importantes
    st.subheader("Top 15 des Caractéristiques les Plus Importantes")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), ax=ax)
    plt.title('Top 15 des Caractéristiques les Plus Importantes')
    plt.xlabel('Importance')
    plt.ylabel('Caractéristiques')
    st.pyplot(fig)

    # Calcul de la MAE
    y_pred_train = pipeline.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred)

    st.write(f"MAE sur le jeu d'entraînement : {mae_train}")
    st.write(f"MAE sur le jeu de test : {mae_test}")

    st.write("""
    L'échelle de gravité va de 0 à 2, avec 3 catégories distinctes.
    Un MAE d'environ 0.67-0.68 signifie que, en moyenne, les prédictions du modèle s'écartent de 0.67 à 0.68 unité sur cette échelle.
    """)

    st.header("4. Comparaison des Modèles")
    #cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    st.write("""
    Nous avons également testé un modèle utilisant seulement les 4 caractéristiques les plus importantes :
    agg, secu1, age_cat, et circ.
    """)

    # Définir les 4 caractéristiques les plus importantes
    important_features = ['agg', 'secu1', 'age_cat', 'circ']

    # Préparer les données pour le modèle simplifié
    X_simple = X_resampled[important_features]
    X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(X_simple, y_resampled, test_size=0.2, random_state=42)

    # Créer et entraîner le pipeline pour le modèle simplifié
    simple_pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', categorical_pipeline, important_features)
        ])),
        ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42))
    ])

    simple_pipeline.fit(X_simple_train, y_simple_train)

    # Faire des prédictions avec le modèle simplifié
    y_simple_pred = simple_pipeline.predict(X_simple_test)

    # Évaluer le modèle simplifié
    simple_accuracy = simple_pipeline.score(X_simple_test, y_simple_test)
    simple_report = classification_report(y_simple_test, y_simple_pred, target_names=label_encoder.classes_, output_dict=True)
    simple_mae = mean_absolute_error(y_simple_test, y_simple_pred)

    # Comparer les performances
    st.write("Comparaison des performances :")
    st.write(f"Précision du modèle complet : {pipeline.score(X_test, y_test):.3f}")
    st.write(f"Précision du modèle simplifié : {simple_accuracy:.3f}")

    st.write("\nRapport de classification du modèle complet :")
    st.table(pd.DataFrame(cr).transpose())

    st.write("\nRapport de classification du modèle simplifié :")
    st.table(pd.DataFrame(simple_report).transpose())

    st.write(f"\nMAE du modèle complet : {mae_test:.3f}")
    st.write(f"MAE du modèle simplifié : {simple_mae:.3f}")

   
    ##############################################
    st.write("""
    Analyse de la comparaison :
    1. Le modèle simplifié utilise seulement 4 caractéristiques au lieu de toutes les caractéristiques du modèle complet.
    2. Comparez les précisions, les rapports de classification et les MAE pour voir comment la simplification affecte les performances.
    3. Notez que le modèle simplifié peut être plus facile à interpréter et plus rapide à exécuter, mais il peut perdre en précision.
    4. Observez si certaines classes (léger, grave, mortel) sont mieux ou moins bien prédites par le modèle simplifié.
    """)

    st.header("5. Visualisation de l'Arbre de Décision")
    fig, ax = plt.subplots(figsize=(50, 50))
    plot_tree(pipeline.named_steps['classifier'],
              feature_names=feature_names,
              class_names=label_encoder.classes_,
              filled=True,
              ax=ax,
              max_depth=3)  # Limiter la profondeur pour la lisibilité
    st.pyplot(fig)

    st.header("6. Analyse Détaillée des Caractéristiques Importantes")
    st.write("""
    Interprétation des caractéristiques les plus importantes :
    1. agg (agglomération) : Influence de la localisation de l'accident (en ville ou hors agglomération).
    2. secu1 (équipement de sécurité) : Impact de l'utilisation des équipements de sécurité.
    3. age_cat (catégorie d'âge) : Effet de l'âge des usagers sur la gravité des accidents.
    4. circ (régime de circulation) : Influence du type de route sur la gravité des accidents.
    """)

############### Création de la page Déploiment du modèle #########################

if page == pages[4]:
    st.write("### Déploiement du modèle")

    st.write("""
    Dans cette section, nous allons déployer notre modèle pour permettre aux utilisateurs de prédire la gravité d'un accident en fonction de différents paramètres.
    """)
    st.write("Aperçu des données utilisées :")
    st.dataframe(df_model.head())
    mode_values = df_model.mode().iloc[0]

    # Entrées utilisateur pour les paramètres
    adresse_accident=st.text_input("""Adresse de l'accident sous le format : 200, avenue salvador allende, 79000 Niort""","200 avenue salvador allende 79000 Niort")
    st.write("Cordonnées GPS de l'accident :")
    api_url = "https://api-adresse.data.gouv.fr/search/?q="
    adr = adresse_accident
    r = requests.get(api_url + urllib.parse.quote(adr))
    results=json.loads(r.content.decode('unicode_escape'))
    lat=results["features"][0]["geometry"]["coordinates"][0]
    long=results["features"][0]["geometry"]["coordinates"][1]
    st.write("latitude " ,lat)
    st.write("longitude " ,long)

    agg = st.selectbox('Agglomération', ['Hors Agglomération', 'Agglomération', 'Inconnue'])
    secu1 = st.selectbox('Équipement de sécurité', ['Oui', 'Non', 'Inconnue'])
    age_cat = st.selectbox('Catégorie d\'âge', ['0-5 ans', '6-12 ans', '13-18 ans', '19-40 ans', '41-60 ans', 'plus de 60 ans', 'Inconnue'])
    circ = st.selectbox('Régime de circulation', ['A sens unique', 'Bidirectionnelle', 'A chaussées séparées', 'Avec voies d\'affectation variable', 'Inconnue'])
    surf_cat = st.selectbox('État de la surface', ['Normale', 'Humide', 'Gel', 'Corps gras', 'Autre', 'Inconnue'])
    lum = st.selectbox('Conditions de luminosité', ['Plein jour', 'Nuit avec éclairage', 'Nuit sans éclairage', 'Inconnue'])
    catr = st.selectbox('Catégorie de route', ['Autoroute', 'Route nationale', 'Route Départementale', 'Voie Communales', 'Hors réseau public', 'Parc de stationnement ouvert à la circulation publique', 'Routes de métropole urbaine', 'autre', 'Inconnue'])
    NBV_cat = st.selectbox('Nombre de voies', ['1 voie', '2 voies', '3 voies', '4 voies', '5 voies ou plus', 'Inconnue'])
    prof = st.selectbox('Profil de la route', ['Plat', 'Pente', 'Sommet de côte', 'Bas de côte', 'Inconnue'])
    plan = st.selectbox('Tracé en plan', ['Partie rectiligne', 'En courbe à gauche', 'En courbe à droite', 'En S', 'Inconnue'])
    lartpc = 'Inconnue' #Trop compliqué à renseigné et il faudrait faire un contrôle de l'entrée des données utilisateurs pour avoir un chiffre et non du texte
    larrout = 'Inconnue' #Trop compliqué à renseigné et il faudrait faire un contrôle de l'entrée des données utilisateurs pour avoir un chiffre et non du texte
    situ = st.selectbox('Situation de l\'accident', ['Chaussée','bande arrêt d\'urgence','Accotement','Trottoir','Piste Cyclable','Voie spéciale','Autre','Inconnue'])
    if situ == 'Chaussée':
      situ=1
    elif situ == 'Bande arrêt d\'urgence':
      situ=2
    elif situ == 'Accotement':
      situ=3
    elif situ == 'Trottoir':
      situ=4
    elif situ == 'Piste Cyclable':
      situ=5
    elif situ == 'Voie spéciale':
      situ=6
    elif situ == 'Autre':
      situ=8
    else: situ='Inconnue'
    vma_c = st.selectbox('Vitesse maximale autorisée', [20, 30, 50, 70, 80, 90, 100, 110, 130, 'Autre', 'Inconnue'])
    place = 'Inconnue' #Trop compliqué à renseigné et il faudrait faire un contrôle de l'entrée des données utilisateurs pour avoir un chiffre et non du texte
    catu = st.selectbox('Catégorie de l\'usager', ['Conducteur', 'Passager', 'Piéton', 'Inconnue'])
    sexe = st.selectbox('Sexe de l\'usager', ['Masculin', 'Féminin', 'Inconnue'])
    nbr_usager_veh = st.selectbox('Nombre d\'usagers par véhicule',[1, 2, 3,4,5,6,7,8])
    avec_pieton = st.selectbox('Présence de piétons', ['Oui', 'Non', 'Inconnue'])
    nbr_vehicule = st.selectbox('Nombre de véhicules',[1,2, 3,4,5,6,7,8])
    nbr_usager_acc = st.selectbox('Nombre d\'usagers impliqués dans l\'accident',[1, 2, 3,4,5,6,7,8])
    Moment_journée = st.selectbox('Moment de la journée', ['Nuit', 'Matin', 'Après-midi', 'Soir', 'Inconnue'])
    Intersection = st.selectbox('Présence d\'intersection', ['Oui', 'Non', 'Inconnue'])
    Météo_Normale = st.selectbox('Conditions météorologiques normales', ['Oui', 'Non', 'Inconnue'])
    Collision_binaire = st.selectbox('Présence de collision', ['Oui', 'Non', 'Inconnue'])
    senc = 'Inconnue' #Trop compliqué à renseigné et il faudrait faire un contrôle de l'entrée des données utilisateurs pour avoir un chiffre et non du texte
    motor = 'Inconnue' #Trop compliqué à renseigné et il faudrait faire un contrôle de l'entrée des données utilisateurs pour avoir un chiffre et non du texte
    obs_fixe = st.selectbox('Présence d\'obstacles fixes', ['Oui', 'Non', 'Inconnue'])
    obs_mobile = st.selectbox('Présence d\'obstacles mobiles', ['Oui', 'Non', 'Inconnue'])
    choc_initial = st.selectbox('Type de choc initial', ['Avant', 'Arrière', 'Côté', 'Aucun', 'Inconnue'])
    cat_vehs = st.selectbox('Catégorie du véhicule', ['Engin spécial', 'Deux Roues', 'Véhicule léger', 'Poids lourd', 'Transport en commun', 'Vélo', 'Autre', 'Inconnue'])


    # Collecter toutes les entrées utilisateur dans un dictionnaire
    user_inputs = {
        'agg': agg,
        'secu1': secu1,
        'age_cat': age_cat,
        'circ': circ,
        'surf_cat': surf_cat,
        'Lum_regroupe': lum,
        'catr': catr,
        'NBV_cat': NBV_cat,
        'prof': prof,
        'plan': plan,
        'lartpc': lartpc,
        'larrout': larrout,
        'situ': situ,
        'vma_c': vma_c,
        'place': place,
        'catu': catu,
        'sexe': sexe,
        'nbr_usager_veh': nbr_usager_veh,
        'avec_pieton': avec_pieton,
        'nbr_vehicule': nbr_vehicule,
        'nbr_usager_acc': nbr_usager_acc,
        'Moment_journée': Moment_journée,
        'Intersection': Intersection,
        'Météo_Normale': Météo_Normale,
        'Collision_binaire': Collision_binaire,
        'senc': senc,
        'motor': motor,
        'obs_fixe': obs_fixe,
        'obs_mobile': obs_mobile,
        'choc_initial': choc_initial,
        'cat_vehs': cat_vehs,
        'long': long,
        'lat': lat
    }

    # Remplacer 'Inconnue' par la valeur la plus fréquente
    for key, value in user_inputs.items():
        if value == 'Inconnue':
            user_inputs[key] = mode_values[key]

    # Convertir les entrées utilisateur en DataFrame
    input_data = pd.DataFrame(user_inputs, index=[0])

    # Bouton pour effectuer la prédiction
    if st.button("Prédire la gravité de l'accident"):
        # Utilisation du pipeline chargé pour faire la prédiction
        prediction = pipeline.predict(input_data)
        proba = pipeline.predict_proba(input_data)

        # Affichage du résultat
        st.subheader("Résultat de la prédiction :")
        gravity = prediction[0]
        gravity_mapping = {0: "Légère", 1: "Grave", 2: "Mortel"}
        st.write(f"La gravité prédite de l'accident est : **{gravity}** c'est à dire {gravity_mapping[gravity]}")

        # Affichage des probabilités
        st.write("Probabilités pour chaque classe :")
        classes = pipeline.classes_
        for i, classe in enumerate(classes):
            st.write(f"{classe} {gravity_mapping[classe]} : {proba[0][i]:.2%}")

        # Visualisation des probabilités
        fig = px.bar( x=classes,y=proba[0])
        fig.update_layout(title = "Probabilités pour chaque classe de gravité",
                  xaxis_title = "Classes",
                  yaxis_title = "Probabilités",
                  xaxis = dict(
                  tickvals = [0,1,2],
                  ticktext  = ["Lègère","Grave","Mortel"]))
        st.plotly_chart(fig)



############### Création de la page Conclusion #########################

if page == pages[5] :
        st.write("# Conclusion ")
    
        st.write("""
        Pistes d'amélioration du modèle :
        1. Optimiser les hyperparamètres du modèle
        2. Collecter plus de données, particulièrement pour les accidents graves et mortels
        3. Explorer d'autres caractéristiques potentiellement pertinentes

        La prochaine étape serait d'implémenter ces améliorations et de comparer les résultats avec notre modèle actuel.
                 
        Recommandations :
        - Promotion des équipements de sécurité : Renforcer les campagnes de sensibilisation.
        - Amélioration de la sécurité routière : Focus sur les zones à haut risque.
        - Vigilance à avertir rapidement les secours en cas d'accident : Renforcer les dispositifs d'alerte.
    """)
