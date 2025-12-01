import pandas as pd, tensorflow as tf, pickle, json
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss

try: d=pd.read_csv('Coffe_sales.csv')
except: exit()

#X=pd.get_dummies(d[['Time_of_Day','coffee_name','Month_name']])
X=pd.get_dummies(d[['Weekday','coffee_name','Month_name']])
y=d['money']

json.dump(X.columns.tolist(), open('model_columns.json','w'))

xt,xv,yt,yv=tts(X,y,test_size=.25,random_state=42)
s=ss()
xt=s.fit_transform(xt)
pickle.dump(s, open('scaler.pkl','wb'))

m=tf.keras.Sequential([
    tf.keras.layers.Dense(32,'relu',input_shape=[xt.shape[1]]),
    tf.keras.layers.Dense(16,'relu'),
    tf.keras.layers.Dense(1)
])
m.compile('adam','mae')
m.fit(xt,yt,epochs=50,verbose=0)
m.save('coffee_price_model.h5')
print("done.")