# coffee price predictor web app
import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

# page config
st.set_page_config(page_title="Coffee Price AI", layout="centered")

# --- load data and artifacts ---
@st.cache_resource # this keeps it fast by loading only once
def load_artifacts():
    model = tf.keras.models.load_model('coffee_price_model.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model_columns = json.load(open('model_columns.json'))
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except:
    st.error("Error: Couldn't load model files. Make sure .h5, .pkl, and .json files are here.")
    st.stop()

# --- helper to extract options from model columns ---
# this ensures the dropdowns only show options the model actually knows about
def get_options(prefix):
    return [col.split(f'{prefix}_')[1] for col in model_columns if col.startswith(prefix)]

# --- user interface ---
st.title("☕ Coffee Price Predictor")
st.write("Select the details below to estimate the price using our Neural Network.")

# create three columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    coffee_options = get_options('coffee_name')
    coffee = st.selectbox("Coffee Type", coffee_options)

with col2:
    time_options = get_options('Time_of_Day')
    time_day = st.selectbox("Time of Day", time_options)

with col3:
    month_options = get_options('Month_name')
    month = st.selectbox("Month", month_options)

# --- prediction logic ---
if st.button("Predict Price", type="primary"):
    
    # create a single row dataframe with 0s
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0
    
    # turn on the specific columns matching user input
    try:
        input_data[f'coffee_name_{coffee}'] = 1
        input_data[f'Time_of_Day_{time_day}'] = 1
        input_data[f'Month_name_{month}'] = 1
    except KeyError:
        st.error("Something went wrong mapping the inputs.")
        st.stop()

    # scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)[0][0]

    # --- display results ---
    st.metric(label="Predicted Price", value=f"${prediction:.2f}")

    # --- visualization ---
    st.subheader("Price in Context")
    
    try:
        df = pd.read_csv('Coffe_sales.csv')
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['money'], kde=True, color="skyblue", bins=20, ax=ax)
        ax.axvline(x=prediction, color='red', linestyle='--', linewidth=3, label=f'Prediction: ${prediction:.2f}')
        ax.set_title("Where this price falls compared to history")
        ax.set_xlabel("Price ($)")
        ax.legend()
        
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Could not load CSV to show the graph.")