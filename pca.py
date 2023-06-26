import numpy as np
#%% Inicializacion 
class pca:
    def __init__(self,sample,por_rec):
        self.sample=sample
        self.por_rec=por_rec


    #%% Meanface
    def meanface_M(self):
        self.meanface = np.zeros([self.sample.shape[0],1])
        for i in range(0,int(self.sample.shape[0])):
            self.meanface[i]= self.sample[i,:].mean() 
        return self.meanface
     #%% Eigenfaces
    def eigenfaces(self): #Devuelve eigenfaces normalizadas
        meanface = self.meanface_M()
        #%% Matriz a partir de la resta de la cara - meanface
        A = np.zeros(self.sample.shape)
        for i in range(0,int(self.sample.shape[1])):
            A[:,i] = self.sample[:,i]-meanface[:,0]
        #%%
        #Matriz de Covarianza
        C= np.matmul(A.T,A) #MxM
        #%% eigenvalores y eigenvectores
        eigenvalor,eigenvector = np.linalg.eig(C) # eigenvectores v
        # print(eigenvector)
        # print(eigenvector.shape)
        eigenvalor = np.real(eigenvalor)
       
        eigenvector = np.real(eigenvector)
        ind = eigenvalor.argsort()
        indice = ind [::-1]
        eigenvalor_sort= eigenvalor[indice]#Ordenar Eigenvalores,colocar el mas grande primero
        eigenvector_sort = eigenvector[:,indice] 
        # print(eigenvector)
        # print("indice:",indice)
        #%%Porcentaje de Representación y eigenfaces 
        aux =  0
        index = 0 #indice para conocer los eigenvalores necesarios para la reconstrucción 
        while aux <= eigenvalor_sort.sum()*self.por_rec:
            aux = aux + eigenvalor_sort[index]
            index = index + 1 #indices fuera del ciclo son excluidos
        #print(index)  
        eigenvector_u = np.zeros([int(self.sample.shape[1]),index])#matriz de eigenvectores a partir del total de eigenvalores
        for i in range(0,index):
            uno= np.array(np.where(eigenvalor==eigenvalor_sort[i])) #Buscar el indice original del eigenvector
           # print(eigenvector)
            eigenvector_u[:,i] = eigenvector_sort [:,int(uno[0,0])]
        eigenvector_u = np.matmul(A,eigenvector_u) #Eigenvectores 936 X 98; ((largoXancho) X #de eigenvalores
        u_norm=np.zeros(eigenvector_u.shape) #almacenar eigenvectores normalizados 
        for i in range (0,index):
              norm = np.linalg.norm(eigenvector_u[:,i])
              u_norm[:,i] = eigenvector_u[:,i]/norm  # normalized matrix
        return u_norm
    #%% Vector de ponderaciones
    def eigenfaces_matriz_ponderantes(self,meanface, eigenfaces ,other_sample):   #Devuelve los ponderantes necesarios para reconstruir la cara(s) en other_sample
       # eigenfaces = self.eigenfaces()
       # meanface = self.meanface_M()
        meanface = meanface 
        eigenfaces = eigenfaces         
        face_sample =  np.zeros([other_sample.shape[0],1]) 
        Q = np.zeros([eigenfaces.shape[1],other_sample.shape[1]]) 
        for i in range (0,other_sample.shape[1]):
            face_sample[:,0] = other_sample[:,i]# Toma de cara muestra del conjunto de datos
            face_sample[:,0] = face_sample[:,0] - meanface[:,0] # Restamos meanface
            Q_aux= np.matmul(eigenfaces.T,face_sample) #calculamos vector de pesos
            Q[:,i]= Q_aux[:,0]
        return Q  # Matriz de ponderantes
    
    def eigenfaces_reconstruccion(self,meanface,eigenfaces, other_sample): #a partir de la matriz de ponderantes se realiza la recontrucción
        meanface = meanface
        eigenfaces = eigenfaces
 #      eigenfaces = self.eigenfaces()
        P = self.eigenfaces_matriz_ponderantes(meanface, eigenfaces, other_sample)#ponderantes
        rebuild= np.zeros(meanface.shape)
        rebuild_samples= np.zeros(other_sample.shape)
        for i in range (0, other_sample.shape[1]):
            rebuild[:,0] = np.matmul(eigenfaces,P[:,i]) + meanface[:,0] ### Reconstrucción de cara 
            rebuild_samples[:,i]=rebuild[:,0] #Almacenar la reconstruccion 
        return rebuild_samples
    
    
    
    
    
    
    
    
    
    
    
    