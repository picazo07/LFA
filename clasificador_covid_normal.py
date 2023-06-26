import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
import posicionador



class clasificador_covid_normal:
    def __init__(self):
        self.meanface= pd.read_csv("clasificador/meanface.csv",header=None) #Cargar cara media
        self.meanface=np.array(self.meanface)
        self.eigenfaces= pd.read_csv("clasificador/eigenfaces.csv",header=None) #Cargar eigenfaces
        self.eigenfaces=np.array(self.eigenfaces)
        self.J= pd.read_csv("clasificador/J_caracteristicas.csv",header=None) #Cargar eigenfaces
        self.J=np.array(self.J)
    
    def clasificacion(self, test):
        meanface = self.meanface 
        eigenfaces = self.eigenfaces   
        J = self.J
        width, height = 256,256
        k=5 # Posicionador
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) #CLAHE
        #%% Posicionador      
        test = cv2.resize(test,(width,height)) #Verificar tamaño de la imagen
        image = cv2.equalizeHist(test) #Imagen para posicionador
        image = cv2.resize(image,(64,64))
        image = np.reshape(image,(64*64,1))
        prueba = posicionador.posicionador(image, k) #Posicionador
        coordenadas_regresion= prueba.regresion() #Coordenadas Regresion   
        coordenadas_warp= prueba.cordenadas_warp(coordenadas_regresion.T, width, height)#coordenadas en 256X256        
        output_warp = prueba.warp_image(test, coordenadas_warp, width, height)
        #%% Proyeccion en el espacio de eigenfaces
        test = output_warp
        test = clahe.apply(test) #CLAHE
        test = np.array(test)
        test = np.reshape(test,(width*height,1))
        test = test/255 #Normalización de pixel 
        test = test - meanface #Restar cara media
        test_w = np.matmul(eigenfaces.T,test) #Proyección en las eigenfaces 
        #%% Seleccion de características
        n_caracteristicas = 600
        test_W_reducido = np.zeros([n_caracteristicas,test_w.shape[1]])
        for i in range (0,n_caracteristicas):   
            test_W_reducido[i,:]=test_w[int(J[i,1]),:]
        #%% Amplificacion J
        J_amp = J[0:n_caracteristicas,:] #Toma de las primeras J
        J_amp[:,0]=np.sqrt(J_amp[:,0]) #Raiz cuadrada de J
        J_amp[:,0]=J_amp[:,0]/J_amp[:,0].sum() # Ponderación de las J
        for i in range (0,n_caracteristicas):   
            test_W_reducido[i,:]=test_W_reducido[i,:]* J_amp[i,0]
        test_W_reducido = test_W_reducido.T
        test_W_reducido = tf.convert_to_tensor(test_W_reducido)
        #print(test_W_reducido.shape)
        #%% Cargar modelo de MLP y predicción
        capa_oculta = 120
        model = Sequential()
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(test_W_reducido.shape[1],)), #Capa de entrada
            keras.layers.Dense(units=capa_oculta, activation='relu'), #Capa Oculta 
            keras.layers.Dense(units=capa_oculta, activation='relu'), # Capa Oculta 
            keras.layers.Dense(units=capa_oculta, activation='relu'), # Capa Oculta
            keras.layers.Dense(units=capa_oculta, activation='relu'), # Capa Oculta  
            keras.layers.Dense(units=1, activation='sigmoid') #Capa de salida  
        ])
        model.load_weights("clasificador/best_model.h5")
        predictions = model.predict(test_W_reducido)
        #%%
        predicciones_binarias = [1 if x >= 0.5 else 0 for x in predictions]
        clases=["COVID_19","Normal"]
        prediccion_final= clases[int(predicciones_binarias[0])]
        porcentaje = (predictions[0]-1) * 100
        porcentaje = porcentaje*-1
        porcentaje = np.round(porcentaje, decimals=0)
        return prediccion_final,porcentaje,output_warp