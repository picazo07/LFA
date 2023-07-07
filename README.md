# Lung Finder Algorithm (LFA)
# What is it?
The LFA (Lung Finder Algorithm) is a new technique proposed for the normalization of the region of interest (the lungs) in chest radiographs. Through this process, the radiographic images from a database can be represented by new standardized images that are similar in rotation, scale, and contrast, i.e., they are normalized. These images can be used for classification in projects that utilize machine learning and/or deep learning, without the need for long computing times.


# Preparing the code
Here you will find all the necessary code for users who want to try the LFA (Lung Finder Algorithm) and the COVID-19 - Normal classifier. Due to file size limitations for uploading, the entire code is compressed into 18 parts. Firstly, download all the 18 compressed parts that are available in this repository. After extracting the folder, you should have all the files as shown in the following image. The entire code has been developed in the Python language, so it is crucial that the respective folders for the classifier or the LFA are not altered for proper usage. Also in the Main file you could watch
how to call the LFA or the MLP.


![image](https://github.com/picazo07/LFA/assets/99782864/53514d55-b942-4591-ba3f-18f4e108a252)



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
Any user can utilize the LFA in their projects related to chest radiographs. The comparison of the test image is automatically performed at a resolution of 64x64 pixels. By finding the k most similar neighbors to the test image, interpolation of the coordinate values is conducted. These coordinates are temporary since a transformation is applied to obtain the corner coordinates of the ROI for performing the Warping operation and generating a new standardized image at 256x256 pixels using bicubic interpolation. The final resolution can be adjusted by the user, although this resolution has shown good results in various classifiers.

To evaluate the performance of the LFA, an additional 100 images were manually labeled to measure the error between the calculated and real coordinates. Through this procedure, the best value of k was found to be 5, with an MSE (Mean Squared Error) of 3.56 pixels and a standard deviation of 1.75 for the 4 points that describe the lung size.

Below are two example radiographs with their extracted ROIs. The red points represent the temporary coordinates, while the blue points depict the final coordinates used in the Warping process.

![fig5](https://github.com/picazo07/LFA/assets/99782864/064c6eae-da7e-4868-a596-0c1d0b2b6095)


# COVID-19 - Normal Classifier
We have made available a classifier that distinguishes between the COVID-19 and normal classes. This classifier utilizes the LFA as a preprocessing step for the images. It employs eigenfaces to further reduce the dimensionality of the images and utilizes Fisher's Ratio (FR) or linear discriminant analysis for feature selection and weighting. Feature weighting is a process we propose to normalize the features and assign higher importance to those with greater discriminatory power. The combination of feature normalization and enhancement through selection and weighting enables an MLP to achieve an impressive accuracy of 97%, which is competitive with state-of-the-art convolutional neural networks commonly used for these tasks. The MLP's topology is relatively simple, consisting of 600 neurons in the input layer, 4 hidden layers with 120 neurons each, and a single neuron in the output layer.

