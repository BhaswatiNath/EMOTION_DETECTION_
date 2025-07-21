import streamlit as st
import joblib

model = joblib.load('model3.pkl')
vectorizer = joblib.load('vectorizer3.pkl')

st.title("Emotion Detection App")

user_input = st.text_input("Enter your text:")

if user_input:
    user_input = user_input.strip()
    user_input_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vector)
    st.write(f"Prediction: {prediction[0]}")
