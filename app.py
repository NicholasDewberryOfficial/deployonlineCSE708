import streamlit as st, pandas as pd, tensorflow as tf, pickle, json
import matplotlib.pyplot as plt, seaborn as sns

st.set_page_config(layout="centered")

@st.cache_resource
def ld():
    m=tf.keras.models.load_model('coffee_price_model.h5')
    s=pickle.load(open('scaler.pkl','rb'))
    c=json.load(open('model_columns.json'))
    return m,s,c

# Change this part in app.py
try: 
    m,s,cols=ld()
except Exception as e: 
    st.error(f"Error loading files: {e}")
    st.stop()

def get(k): return [x.split(f'{k}_')[1] for x in cols if x.startswith(k)]

st.title("Coffee Price AI")
c1,c2,c3=st.columns(3)
cf=c1.selectbox("Coffee", get('coffee_name'))
td=c2.selectbox("Time", get('Time_of_Day'))
mn=c3.selectbox("Month", get('Month_name'))

if st.button("Predict"):
    row=pd.DataFrame(columns=cols); row.loc[0]=0
    try:
        row[f'coffee_name_{cf}']=1
        row[f'Time_of_Day_{td}']=1
        row[f'Month_name_{mn}']=1
    except: pass
    
    p=m.predict(s.transform(row),verbose=0)[0][0]
    st.metric("Price", f"${p:.2f}")

    try:
        d=pd.read_csv('Coffe_sales.csv')
        fig,ax=plt.subplots(figsize=(10,4))
        sns.histplot(d['money'],kde=True,ax=ax,color='skyblue')
        ax.axvline(p,color='red',ls='--')
        st.pyplot(fig)
    except: pass