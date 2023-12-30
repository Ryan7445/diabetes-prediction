import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import streamlit as st
from web_function import train_model

def app(df, x, y):
  warnings.filterwarnings("ignore")
  st.set_option('deprecation.showPyplotGlobalUse', False)

  st.title("Visualisasi Data")

  if st.checkbox("Plot Confusion Matrix"):
    model, score = train_model(x,y)
    plt.figure(figsize=(10,10))
    pred = model.predict(x)
    cm = confusion_matrix(y, pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
    disp.plot()
    st.pyplot()
  
  if st.checkbox("Plot Scatter KNN"):
    model, score = train_model(x,y)
    plt.figure(figsize=(10,10))
    sns.scatterplot(x="Glucose", y="BloodPressure", hue="Outcome", data=df)
    plt.xlabel("Glucose")
    plt.ylabel("BloodPressure")
    plt.legend(loc='upper left')
    st.pyplot()