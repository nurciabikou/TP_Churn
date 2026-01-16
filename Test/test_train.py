
import os
import glob
import pytest
import joblib
import numpy as np
import pandas as pd
import subprocess
from sklearn.linear_model import LogisticRegression

def test_model_file_generated():
    """Vérifie qu'au moins un fichier .pkl est présent dans data/"""
    
    # Cherche tous les fichiers .pkl dans data/
    pkl_files = glob.glob("data/*.pkl")
    
    print("Fichiers .pkl trouvés :", pkl_files)
    
    # Vérifie qu'au moins un fichier a été généré
    assert len(pkl_files) > 0, "❌ Aucun fichier .pkl généré dans le dossier data/"

    #verifie qu'il fait bien les predictions 




def test_model_can_predict():
    """Vérifie que le modèle .pkl généré peut faire des prédictions"""
    
    pkl_files = glob.glob("data/*.pkl")
    assert len(pkl_files) > 0, "❌ Aucun fichier .pkl généré dans le dossier data/"
    
    model_file = pkl_files[0]
    model = joblib.load(model_file)
    
    # Exemple de données sous forme de DataFrame avec colonnes correctes
    example_features = pd.DataFrame(
        [[30, 1, 2, 5]],
        columns=["Age", "Account_Manager", "Years", "Num_Sites"]
    )
    
    prediction = model.predict(example_features)
    assert prediction[0] in [0, 1], f"❌ La prédiction n'est pas correcte : {prediction[0]}"
    
    print(f"✅ Le modèle {model_file} prédit correctement : {prediction[0]}")

#corrigé
 
def test_train_model_file_exists():
    """Vérifie que le fichier churn_model_clean.pkl est créé après exécution de train.py"""
    assert os.path.exists('data/churn_model_clean.pkl'), (
        "Le fichier churn_model_clean.pkl n'existe pas après l'exécution de train.py."
    )
 
def test_train_model_loading():
    """Vérifie que le fichier sauvegardé contient un modèle Random Forest"""
    model = joblib.load('data/churn_model_clean.pkl')
    assert isinstance(model, LogisticRegression), (
        "Le fichier churn_model_clean.pkl ne contient pas un modèle LogisticRegression."
    )
 
def test_train_model_prediction():
    """Vérifie que le modèle entraîné peut prédire sur un sous-ensemble des données"""
    model = joblib.load('data/churn_model_clean.pkl')
    data = pd.read_csv('data/train_data.csv')
    X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
 
    prediction = model.predict(X[:1])
    assert prediction is not None, "Le modèle n'a pas retourné de prédiction."