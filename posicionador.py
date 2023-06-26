import cv2
import numpy as np
import pandas as pd
import knn
class posicionador:
    def __init__(self,test,k):
        self.k=k # hiperparametro
        self.train_W=pd.read_csv("posicionador/train_W.csv",header=None) #Cargar ponderantes de entrenamiento
        self.train_W=np.array(self.train_W)
        self.test=test # Elementos de prueba 64X64
        self.coordenadas_tag=pd.read_csv("posicionador/coordenadas_posicionador.csv",header=None) #Cargar coordenadas del posicionador
        self.coordenadas_tag = np.array(self.coordenadas_tag)
        self.meanface= pd.read_csv("posicionador/meanface_posicionador.csv",header=None) #Cargar cara media
        self.meanface=np.array(self.meanface)
        self.eigenfaces= pd.read_csv("posicionador/eigenfaces.csv",header=None) #Cargar eigenfaces
        self.eigenfaces=np.array(self.eigenfaces)
#%% Proyección de los elementos de prueba en los de entrenamiento    
    def eigenfaces_matriz_ponderantes(self):   #Devuelve los ponderantes necesarios para reconstruir la cara(s) en other_sample
        meanface = self.meanface 
        eigenfaces = self.eigenfaces        
        face_sample =  np.zeros([self.test.shape[0],1]) 
        Q = np.zeros([eigenfaces.shape[1],self.test.shape[1]]) 
        for i in range (0,self.test.shape[1]):
            face_sample[:,0] = self.test[:,i]# Toma de cara muestra del conjunto de datos
            face_sample[:,0] = face_sample[:,0] - meanface[:,0] # Restamos meanface
            Q_aux= np.matmul(eigenfaces.T,face_sample) #calculamos vector de pesos
            Q[:,i]= Q_aux[:,0]
        return Q  # Matriz de ponderantes del conjunto de prueba
#%%Regresion
    def regresion(self,width=64, height=64):
        self.test_W = self.eigenfaces_matriz_ponderantes()
        self.proyeccion_coordenadas= knn.knn(self.train_W,self.test_W,self.coordenadas_tag,self.k)#Objeto tipo KNN
        self.ponderaciones_knn=self.proyeccion_coordenadas.weighted() #Pesos KNN
        self.distancias= self.proyeccion_coordenadas.distancia_euc() # Distancias euclidianas
        aux = np.zeros([2,int(self.ponderaciones_knn.shape[0])])
        #print("aux",aux.shape)
        sum_aux=np.zeros([1,self.k]) #suma de etiquetas con ponderaciones
        coordenadas_regresion = np.zeros([self.test_W.shape[1],8])#Coordenadas Calculadas 
        #print("test_W",self.test_W.shape)
        for sample in range(0,self.test_W.shape[1]):
            aux[0,:]=self.ponderaciones_knn[:,sample] ##Ponderaciones de todo el conjunto 
            aux[1,:]=self.coordenadas_tag[0,:] ## Indices
            aux=aux[:,aux[0].argsort()[::-1]]#ordenamiento, colocar el mas pesado primero
            sum_aux=np.zeros([1,self.k]) #suma de etiquetas con ponderaciones
            sum_w=np.zeros([1,self.k])#Suma de ponderaciones
            
            for i in range(0,self.k):
                sum_w[0,i]=aux[0,i]# Suma de las ponderaciones 
            for j in range (0,8):
                for i in range(0,self.k):
                   # print(aux.T[i,1])
                    #print(self.coordenadas_tag[j+1,int(aux.T[i,1])])
                    sum_aux[0,i]=aux[0,i]*self.coordenadas_tag[j+1,int(aux.T[i,1])]/sum_w.sum()#Ponderación * valor de etiqueta / suma de ponderaciones
                    #print(sum_aux)
                coordenadas_regresion[sample,j]= sum_aux.sum() 
        coordenadas_regresion=coordenadas_regresion*(width/64)                                       
        return coordenadas_regresion
    
    def cordenadas_warp(self,coordenadas_regresion1,width, height):
        coordenadas_regresion=coordenadas_regresion1*(width/64) #Escalamiento de coordenadas
        #%% Coordenadas del rectangulo
        if coordenadas_regresion[0,0]==coordenadas_regresion[2,0]: # Pendiente igual a 0
            #Recta 1 - 3
            p1x = coordenadas_regresion[4,0]
            p1y = coordenadas_regresion[1,0]
            
            #Recta 1 - 4
            p2x = coordenadas_regresion[6,0]
            p2y = coordenadas_regresion[1,0]
 
            #Recta 2 - 3
            p3x = coordenadas_regresion[4,0]
            p3y = coordenadas_regresion[3,0]
            
            #Recta 2 - 4
            p4x = coordenadas_regresion[6,0]
            p4y = coordenadas_regresion[3,0]
            
            
           
        if coordenadas_regresion[0,0]!=coordenadas_regresion[2,0]: # Pendiente diferente de 0
            m1= (coordenadas_regresion[1,0]-coordenadas_regresion[3,0])/(coordenadas_regresion[0,0]-coordenadas_regresion[2,0]) #Pendiente Vertical
            m2= (coordenadas_regresion[5,0]-coordenadas_regresion[7,0])/(coordenadas_regresion[4,0]-coordenadas_regresion[6,0]) #Pendiente Vertical
            #%% Rectas
            recta1= np.array([m2, (m2*(-1*coordenadas_regresion[0,0]))+coordenadas_regresion[1,0]]) #Recta superior
            recta2= np.array([m2, (m2*(-1*coordenadas_regresion[2,0]))+coordenadas_regresion[3,0]]) #Recta inferior
            recta3= np.array([m1, (m1*(-1*coordenadas_regresion[4,0]))+coordenadas_regresion[5,0]]) #Recta izquierda
            recta4= np.array([m1, (m1*(-1*coordenadas_regresion[6,0]))+coordenadas_regresion[7,0]]) #Recta derecha
            #Recta 1 - 3
            p1x = (recta1[1]-recta3[1])/(recta3[0]-recta1[0])
            p1y = (recta1[0]*p1x)+ recta1[1]
            #Recta 1 - 4
            p2x = (recta1[1]-recta4[1])/(recta4[0]-recta1[0])
            p2y = (recta1[0]*p2x)+ recta1[1]
            #Recta 2 - 3
            p3x = (recta2[1]-recta3[1])/(recta3[0]-recta2[0])
            p3y = (recta2[0]*p3x)+ recta2[1]
            #Recta 2 - 4
            p4x = (recta2[1]-recta4[1])/(recta4[0]-recta2[0])
            p4y = (recta2[0]*p2x)+ recta2[1]        
        #Warp
        pts1 = np.float32([[p1x,p1y],[p2x,p2y],[p3x,p3y],[p4x,p4y]])
        return pts1
        
    def warp_image(self,image,pts1,width, height):
        pts1 = pts1
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        output_warp = cv2.warpPerspective(image,matrix,(width,height),flags=cv2.INTER_CUBIC) 
        return output_warp
        
   #      INTER_NEAREST = 0, 
   # INTER_LINEAR = 1, 
   # INTER_CUBIC = 2, 
   # INTER_AREA = 3, 
   # INTER_LANCZOS4 = 4, 
