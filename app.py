import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0)
fare = st.number_input("Fare", min_value=0.0)
family = st.number_input("Family Size", min_value=0)

sex_val = 0 if sex == "male" else 1

if st.button("Predict"):
    input_data = np.array([[pclass, sex_val, age, fare, family]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("🎉 Survived!")
    else:
        st.error("💀 Did Not Survive")