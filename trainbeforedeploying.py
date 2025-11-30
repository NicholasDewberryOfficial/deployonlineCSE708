# minimal training script to create artifacts for the web app

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json

print("training model...")

# 1. load data
try:
    df = pd.read_csv('Coffe_sales.csv')
except FileNotFoundError:
    print("error: coffe_sales.csv not found.")
    exit()

# 2. select only the features the app will use
features = df[['Time_of_Day', 'coffee_name', 'Month_name']]
target = df['money']

# 3. process data (one-hot encoding)
X = pd.get_dummies(features)

# --- SAVE ARTIFACT 1: COLUMN NAMES ---
# the app needs to know exactly what columns the model expects
model_columns = X.columns.tolist()
with open('model_columns.json', 'w') as f:
    json.dump(model_columns, f)

# 4. split and scale
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- SAVE ARTIFACT 2: SCALER ---
# save the scaler so we can transform user input exactly like training data
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 5. build and train neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mae')
model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

# --- SAVE ARTIFACT 3: MODEL ---
model.save('coffee_price_model.h5')

print("success! created: coffee_price_model.h5, scaler.pkl, model_columns.json")