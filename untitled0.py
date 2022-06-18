import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##Importamos Datos
house_df= pd.read_csv('precios_hogares.csv')

#Visualizacion
sns.scatterplot(x ='sqft_living', y='price',data = house_df)

#correlación
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(house_df.corr(),annot=True)

##Limpieza de Datos
selected_features = ['bedrooms','bathrooms','sqft_living','sqft_lot',
                     'floors','sqft_above','sqft_basement']


x= house_df[selected_features]
y=house_df['price']

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
X_Scaler= scaler.fit_transform(x)


#Normalizamos output
y= y.values.reshape(-1,1)
Y_Scaler = scaler.fit_transform(y)

##Entrenamiento de Modelo
from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Scaler,Y_Scaler,test_size= 0.25)

##Definiendo Modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu',input_shape=(7,)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer='Adam', loss='mean_squared_error')

epoch_hist= model.fit(X_Train,Y_Train, epochs=100, batch_size= 50, validation_split=0.2)

#Evaluando modelos
epoch_hist.history.keys()

#Grafico
plt.plot(epoch_hist.history['loss'])
plt.plot(epoch_hist.history['val_loss'])
plt.title('Progreso durante entrenamiento')
plt.legend('Training loss, Validation loss')
plt.xlabel('epochs')
plt.ylabel('Entrenamiento y Validacion loss')

#Prediccion
#Definir casa con sus respectivos inputs 
#bedrooms','bathrooms','sqft_living','sqft_lot',
#'floors','sqft_above','sqft_basement'

X_Test_1= np.array([[4,3,1960,5000,1,2000,3000]])

scaler_1= MinMaxScaler()
test_1= scaler_1.fit_transform(X_Test_1)

#Prediccion
Y_Predict_1= model.predict(test_1) 

#Revirtiendo Escalado para apreciar el precio correctamente escalado
Y_Predict_1= scaler.inverse_transform(Y_Predict_1)






