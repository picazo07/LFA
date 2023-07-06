# Lung Finder Algorithm (LFA)
# What is it?
The LFA (Lung Finder Algorithm) is a new technique proposed for the normalization of the region of interest (the lungs) in chest radiographs. Through this process, the radiographic images from a database can be represented by new standardized images that are similar in rotation, scale, and contrast, i.e., they are normalized. These images can be used for classification in projects that utilize machine learning and/or deep learning, without the need for long computing times.

# How does it work?
Given that each pixel in an image becomes a dimension, the Eigenfaces method is used as a dimensionality reduction technique to overcome the "Curse of Dimensionality". Eigenfaces rely on Principal Component Analysis (PCA) and are employed to represent a dataset with a lower number of dimensions. In the LFA (Lung Finder Algorithm), Eigenfaces are applied to the grayscale pixel values. Weighted K-NN regression is utilized to find the k most similar training images to the test image, enabling automatic interpolation of the coordinates of the Region of Interest (ROI). Finally, a warpimg operation is employed to extract the ROI and generate a new standardized image.

The following block diagram illustrates the sequence employed in the LFA. The algorithm consists of training and testing stages, which will be detailed further.
![Imagen1](https://github.com/picazo07/LFA/assets/99782864/c806d76a-91a6-48a1-9ea9-15ba14e15965)


# Training stage
The LFA (Lung Finder Algorithm) utilizes the coordinates of the training images as their own features. For this purpose, a manual labeling of 400 randomly selected images was performed from the three classes (viral pneumonia, COVID-19, and normal) of the "COVID-19 Radiography Database" available on Kaggle. The manual labeling consists of placing points or landmarks on the image. The first landmark is located at the midpoint of the spine at the lung level, while the second one is also on the spine but at the lower part. The next two points represent the width of each lung and are placed at the height of the midpoint of the first two points, ensuring that the two lines representing the lung width and length are perpendicular to each other.

To increase the diversity of the dataset and achieve a more balanced distribution, a data augmentation technique was employed. This involved creating artificial images with variations in translation and rotation. The translation ranged from -5 to 5 pixels, and the rotation from -10 to 10 degrees, as recommended in different references. This resulted in a total of 4400 augmented examples that were used for training. Additionally, these images underwent contrast enhancement using histogram equalization. To reduce computational time, the images were resized to 64x64 pixels, and the coordinates of the ROI were subsequently scaled to the desired size by the user.

Once the images were prepared, the eigenfaces process was applied to reduce the dimensionality of the training set. The image below shows the arrangement of points Q and  the images created during the data augmentation process.



![fig4](https://github.com/picazo07/LFA/assets/99782864/33dcf47a-61f3-4b77-9c73-732a822ccb85)

# Test stage 


# COVID-19 - Normal Classifier


# Preparing the code
Here you will find all the necessary code for users who want to try the LFA (Lung Finder Algorithm) and the COVID-19 - Normal classifier. Due to file size limitations for uploading, the entire code is compressed into 18 parts. Firstly, download all the 18 compressed parts that are available in this repository. After extracting the folder, you should have all the files as shown in the following image. The entire code has been developed in the Python language, so it is crucial that the respective folders for the classifier or the LFA are not altered for proper usage.

![image](https://github.com/picazo07/LFA/assets/99782864/ebc56eac-cc78-4284-b1db-986f93ea3630)



# Using the LFA


# Using the Classifier
