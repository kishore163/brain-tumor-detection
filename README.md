# brain-tumor-detection

Link to the collab notebook: https://colab.research.google.com/drive/1ygQu45nRHHbbEgosxS1IKP_rYTER5uCj?usp=sharing

## About Dataset
Dataset consists of 512 images of MRI scans of various people who have and do not have brain tumor

## 1. Preprocessing the data
Vaious operations are performed on the image so that all the images are universally blended which helps in better training of the images.
Images are:
#### resized to 128, 128,3
#### converted to a numpy array
#### one hot encoding performed(to convert categorical variables)

## 2. Split the images into Train and test parts
The images are splited into 2 parts. In this project the train part consists of 80% of the total images which are used to train the CNN model. THe rest 20% of the images are used to test how well the model is being trained.
This makes sure that the model is neither overfitted or underfitted the images

## 3.Buiding the model using CNN

I have used 3 layered neural network in this project

1st layer:
the kernel size is (2,2) and the activation function used is relu

2nd layer:
In 2nd layer also the kernel size is (2,2) and relu is used as the activation function

3rd layer:
Here we flatten our array to a single dimension and again apply relu function

Fully connected layer:
In binary/multivariate class classification this layer holds the probability of data belonging to a particular class

## 4. Evaluating the model
The model is evaluated to find out its accuracy

## 5.Predicting the output
When the pixel values of the image is sent in the form of a numpy array,it predicts whether a person has brain tumor or not
