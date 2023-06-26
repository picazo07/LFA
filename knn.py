import numpy as np
import math
#%%
##Funciones
class knn:
    def __init__(self,train,test,tag_train,k):
        self.train=train
        self.test=test
        self.tag_train=tag_train
        self.k=k
    
    def distancia_euc(self):  ### Distancia Euclidiana, Devuelve una matriz train X test 
        self.distance=np.zeros([self.train.shape[1],self.test.shape[1]])
        for j in range(0,self.test.shape[1]):
            dis_aux=np.zeros(self.train.shape)
            sum_aux=np.zeros(self.train.shape[1])
            for i in range(0,self.train.shape[1]):
                dis_aux[:,i]=self.train[:,i]-self.test[:,j]
            dis_aux=dis_aux*dis_aux
            for i in range(0,self.train.shape[1]):
                sum_aux[i]= dis_aux[:,i].sum()  
            sum_aux=np.sqrt(sum_aux)
            self.distance[:,j]=sum_aux  
        return self.distance
    
    
    def weighted(self): ##Ponderación, entrada train X test  salida train X test
      distance= self.distancia_euc()
      zeros = np.array(np.where(distance==0))
      self.weight = np.zeros(distance.shape)
      z=0
      for j in range(0,distance.shape[1]):
         if distance[:,j].min()==0:
             self.weight[:,j]=0
             self.weight[zeros[0,z],j]=1
             if z < 3:        
                 z=z+1          
         else:        
              for i in range(0,distance.shape[0]): 
                self. weight[i,j] = math.exp((distance[i,j]**2)/(-2*(distance[:,j].min()**2)))  
      return self.weight
    
    def w_knn (self, tag,test_p,w):## tag a buscar, elemento de prueba, matriz de ponderaciones
        clase = np.array(np.where(self.tag_train==tag)) #Filtrar la clase en el vector de etiquetas
        sum_aux=np.zeros(clase.shape)##Suma para un punto
        nn=np.zeros([1,self.k])#vecinos mas cercanos
        weight=w
        k=self.k
        for i in range(0,clase.shape[1]):
            row=int(clase[:,i])
            sum_aux[:,i]= weight[row,test_p] #Suma de las ponderaciones// seleccion de prueba       
        sum_aux=sum_aux[:,sum_aux[0].argsort()[::-1]]#ordenamiento, colocar el mas pesado primero
        for i in range(0,k):
               nn[0,i]= sum_aux[0,i]  #Suma de los vecinos    
        return nn.sum()
    
    def clasificador3c(self,i1, i2, i3):
        if i2<i1>i3: return 1
        elif i1<i2>i3: return 2
        elif i1<i3>i2: return 3
        else: return None 
        
    def knn_clasificacion3c(self):
        'metodo para la clasificacion de 3 clases'
        clasi=np.zeros(self.test.shape[1])
        for i in range (0,self.test.shape[1]):
            i1= self.w_knn( 1, i) ### tag/// k //// test
            i2= self.w_knn( 2, i) ### tag/// k //// test
            i3= self.w_knn( 3, i) ### tag/// k //// test
            clasi[i]=self.clasificador3c(i1, i2, i3)
        return clasi

    def clasificador2c(self,i1, i2, tag1, tag2):
        if i1>i2: return tag1
        elif i2>i1: return tag2
        else: return None 
    
    def knn_clasificacion2c(self, tag1, tag2, w):
        clasi=np.zeros(self.test.shape[1])
        for i in range (0,self.test.shape[1]):
            i1= self.w_knn( tag1, i,w) ### tag/// k //// test
            i2= self.w_knn( tag2, i,w) ### tag/// k //// test
            clasi[i]=self.clasificador2c(i1, i2, tag1, tag2)
        return clasi

    def regresion_img(self,tags):
        ponderacion= self.weighted()
        clasi=np.zeros(self.test.shape[1]) #Vector de Clasificación
        aux=np.zeros([2,int(self.train.shape[1])]) #-matriz para ponderacion y etiquetas 
        for img in range(0,self.test.shape[1]):
            aux[0,:]=ponderacion[:,img] #Ponderacion
            for i in range(0,int(self.train.shape[1]/9)):
                for j in range(0,len(tags)):
                    aux[1,(i*9)+j]=tags[j] #Etiquetas 
            aux=aux[:,aux[0].argsort()[::-1]]#ordenamiento 
            sum_aux=np.zeros([1,self.k]) #suma de etiquetas con ponderaciones
            sum_w=np.zeros([1,self.k])#Suma de ponderaciones
            uno= np.array(np.where(aux[0,:]==1)) #Buscar uno en la ponderación 
            if uno.size>0:  #Si hay un 1   
                clasi[img]=aux[1,0]
            else:
                for i in range(0,self.k):
                    sum_w[0,i]=aux[0,i] #Suma de ponderaciones       
                for i in range(0,self.k):
                    sum_aux[0,i]=aux[0,i]*aux[1,i]/sum_w.sum() #suma de etiquetas con ponderaciones
            clasi[img]=sum_aux.sum()
        
        return clasi
    
    def regresion(self):
        ponderacion= self.weighted()
        clasi=np.zeros(self.test.shape[1]) #Vector de Clasificación
        aux=np.zeros([2,int(self.train.shape[1])]) #-matriz para ponderacion, sujetos y etiquetas 
        for j in range(0,self.test.shape[1]):     
            aux[0,:]=ponderacion[:,j] #Ponderacion, fusion de columna de etiquetas con sus indces
            aux[1,:]=self.tag_train #Etiquetas
            aux=aux[:,aux[0].argsort()[::-1]]#ordenamiento 
            sum_aux=np.zeros([1,self.k]) #suma de etiquetas con ponderaciones
            sum_w=np.zeros([1,self.k])#Suma de ponderaciones
            uno= np.array(np.where(aux[0,:]==1)) #Buscar uno en la ponderación 
            if uno.size>0:  #Si hay un 1                
               clasi[j]=aux[1,0]
            else:
                for i in range(0,self.k):
                    sum_w[0,i]=aux[0,i]       
                for i in range(0,self.k):
                    sum_aux[0,i]=aux[0,i]*aux[1,i]/sum_w.sum()        
                clasi[j]=sum_aux.sum()
        return clasi