# Lung Finder Algorithm (LFA)
# What is it?
The LFA (Lung Finder Algorithm) is a new technique proposed for the normalization of the region of interest (the lungs) in chest radiographs. Through this process, the radiographic images from a database can be represented by new standardized images that are similar in rotation, scale, and contrast, i.e., they are normalized. These images can be used for classification in projects that utilize machine learning and/or deep learning, without the need for long computing times.

# How does it work?
Given that each pixel in an image becomes a dimension, the Eigenfaces method is used as a dimensionality reduction technique to overcome the "Curse of Dimensionality". Eigenfaces rely on Principal Component Analysis (PCA) and are employed to represent a dataset with a lower number of dimensions. In the LFA (Lung Finder Algorithm), Eigenfaces are applied to the grayscale pixel values. Weighted K-NN regression is utilized to find the k most similar training images to the test image, enabling automatic interpolation of the coordinates of the Region of Interest (ROI). Finally, a warpimg operation is employed to extract the ROI and generate a new standardized image.

The following block diagram illustrates the sequence employed in the LFA. The algorithm consists of training and testing stages, which will be detailed further.
![Imagen1](https://github.com/picazo07/LFA/assets/99782864/c806d76a-91a6-48a1-9ea9-15ba14e15965)


# Training stage



# Test stage 


# COVID-19 - Normal Classifier


# Preparing the code
Here you will find all the necessary code for users who want to try the LFA (Lung Finder Algorithm) and the COVID-19 - Normal classifier. Due to file size limitations for uploading, the entire code is compressed into 18 parts. Firstly, download all the 18 compressed parts that are available in this repository. After extracting the folder, you should have all the files as shown in the following image. The entire code has been developed in the Python language, so it is crucial that the respective folders for the classifier or the LFA are not altered for proper usage.

![image](https://github.com/picazo07/LFA/assets/99782864/ebc56eac-cc78-4284-b1db-986f93ea3630)



# Using the LFA


# Using the Classifier
