import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('../models/sales_prediction_model.pkl')

st.set_page_config(page_title="Sales Predictor", page_icon="📈")

st.title("Advertising Sales Predictor")
st.write("This app predicts product sales based on advertising budget allocated across TV, Radio, and Newspaper. Adjust the sliders below to see the predicted sales update instantly.")

st.sidebar.header("Advertising Budget")
tv = st.sidebar.slider("TV Budget ($ thousands)", 0.0, 300.0, 150.0)
radio = st.sidebar.slider("Radio Budget ($ thousands)", 0.0, 50.0, 25.0)
newspaper = st.sidebar.slider("Newspaper Budget ($ thousands)", 0.0, 120.0, 30.0)

# Calculate the interaction feature, same as during training
tv_radio_interaction = tv * radio

input_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper],
    'TV_Radio_Interaction': [tv_radio_interaction]
})

prediction = model.predict(input_data)[0]

st.subheader("Predicted Sales")
st.write(f"Estimated sales: **${prediction:.2f} thousand**")

st.subheader("Your Input")
st.write(input_data)

st.subheader("Insight")
st.write("Notice how sales respond much more strongly when both TV and Radio budgets are increased together, compared to increasing either one alone. This reflects a synergy effect found during analysis, where the combined TV and Radio spend was the strongest predictor of sales.")