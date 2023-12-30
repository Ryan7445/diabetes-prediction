import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data()
def load_data():
    df = pd.read_csv('data.csv')
    x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df[['Outcome']]
    return df, x, y

@st.cache_data()
def train_model(x,y):
  knn = KNeighborsClassifier(n_neighbors = 9)
  knn.fit(x, y)
  score = knn.score(x, y)

  return knn, score

def predict(x, y, features):
  knn, score = train_model(x,y)

  pred = knn.predict(np.array(features).reshape(1,-1))

  return pred, score

