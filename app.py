import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
family = st.slider("Family Size", 1, 10, 1)

sex = 0 if sex == "male" else 1

input_data = np.array([[pclass, sex, age, fare, family]])

if st.button("Predict Survival"):
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("🎉 Survived!")
    else:
        st.error("💀 Did Not Survive")