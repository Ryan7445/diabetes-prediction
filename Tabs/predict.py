import streamlit as st
from web_function import predict

def app(df, x, y):
  st.title("Prediksi Diabetes")
  st.write("Masukkan data anda")
  
  col1, col2 = st.columns(2)

  with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 200, 0)
    bloodpressure = st.number_input("BloodPressure", 0, 200, 0)
    skinthickness = st.number_input("SkinThickness", 0, 200, 0)
  with col2:
    insulin = st.number_input("Insulin", 0, 200, 0)
    bmi = st.number_input("BMI", 0.00, 200.00, 0.00)
    diabetespedigreefunction = st.number_input("DiabetesPedigreeFunction", 0.000, 200.000, 0.000)
    age = st.number_input("Age", 0, 200, 0)

  features = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]
  
  if st.button("Predict"):
    pred, score = predict(x, y, features)

    if pred == 0:
      st.success("Anda tidak terkena diabetes")
    else:
      st.error("Anda terkena diabetes")

    st.write("Akurasi: ", (score*100), "%")