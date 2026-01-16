import pandas as pd
import joblib

dataset=pd.read_csv('data/train_data.csv')
cols_to_drop = ['Onboard_date', 'Location', 'Company','Total_Purchase']
df = dataset.drop(columns=cols_to_drop)


# Entrainer un modele de regression logistique avec  Skearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Entraînement du modèle
X = df[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
 
y = df['Churn']
 
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
# Sauvegarder le modèle entrainé avec joblib
joblib.dump(model, 'data/churn_model_clean.pkl')

print("Modèle de régression entrainé")
