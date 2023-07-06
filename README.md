# Lung Finder Algorithm (LFA)
# What is it?
The LFA (Lung Finder Algorithm) is a new technique proposed for the normalization of the region of interest (the lungs) in chest radiographs. Through this process, the radiographic images from a database can be represented by new standardized images that are similar in rotation, scale, and contrast, i.e., they are normalized. These images can be used for classification in projects that utilize machine learning and/or deep learning, without the need for long computing times.

# How does it work?
Ya que cada pixel en una imagen se sonvierte en una dimension, se utiliza el metodo de Eigenfaces como método de reducción de dimensiones para asi evitar "la Maldición de la dimensionalidad". Las eigenfaces estan basadas en el PCA y sirven para poder representar un conjunto de datos con una menor cantidad de dimensiones. Para el LFA, las eigenfaces fueron aplicadas en los valores de gris de los pixeles. El K-NN ponderado regresión se utiliza para encontrar las k imagenes mas parecidas del entrenamiento a la imagen de prueba para poder interpolar las coordenadas de la ROI de manera automatica. Finalmente se utiliza una operación de deformación para extraer la ROI a una nueva imagén estandarizada. 


![ALP](https://github.com/picazo07/LFA/assets/99782864/4e6d3823-af53-472d-a1f7-026cee0f6bd2)


# COVID-19 - Normal Classifier


# Preparing the code

# Using the LFA


# Using the Classifier
