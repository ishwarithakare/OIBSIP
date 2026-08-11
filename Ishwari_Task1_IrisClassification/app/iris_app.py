import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('../models/iris_classifier.pkl')

# Page setup
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("Iris Flower Species Classifier")
st.write("This app predicts the species of an iris flower based on its measurements. Adjust the sliders below and see the prediction update instantly.")

# Sidebar sliders for input
st.sidebar.header("Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 3.8)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prepare input for prediction
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Predict
species_names = ['Setosa', 'Versicolor', 'Virginica']
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# Display result
st.subheader("Prediction")
st.write(f"Predicted species: **{species_names[prediction]}**")

st.subheader("Prediction Confidence")
proba_df = pd.DataFrame({
    'Species': species_names,
    'Confidence': prediction_proba
})
st.bar_chart(proba_df.set_index('Species'))

st.subheader("Your Input")
st.write(input_data)