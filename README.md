# ml-project-cifar10-cnn

## INTRODUCTION
The CIFAR-10 data set consists of 60000 32×32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Recognizing photos from the CIFAR-10 collection is one of the most common and challenging problems in the today’s world of machine learning. I will describe the analysis behind the CNN model proposed, being able to achieve over 80% of accuracy.

## SYSTEM ARCHITECTURE
The image recognition/classification problem is a very demanding task, consuming plenty of resources from the system, depending on the complexity and elements involved on the CNN model design.
For my purpose, considering the demands and time constrains to test and deliver the proposed model, I used the cloud-based platform known as FloydHub (https://www.floydhub.com). 
FloydHub is a platform used for the creation of intelligent deep learning models. It is equipped with tools that enable users to create, run, and deploy models at a faster rate.
Part of the benefit of using FloydHub is the possibility of lease a GPU base system that will obviously take advantage of the potential of libraries like Keras of Caffe which implement the models based on CUDA.
In my specific case, the system definition was done using the following GPU based settings:
* Tesla K80 · 12 GB Memory
* 61 GB RAM · 100 GB SSD

## MODEL DESCRIPTION
The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The tech report Learning Multiple Layers of Features from Tiny Images [1], on its Chapter 3, describes the dataset and the methodology followed when collecting it in much greater detail.
In the other hand, Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research, but in order to do that, there is a critical condition to be sorted out, finding the right dataset to test and build predictive models.
That is a key aspect and a common problem in Machine Learning. Fortunately, the Keras.Datasets module already includes methods to load and fetch popular reference datasets. CIFAR-10 is one of the modules already included, considered as small images classification dataset.
As mentioned before, the dataset contains in total 60000 32×32 color images in 10 classes, having 50000 training images and 10000 test images.
During the Dataset Load Process, I was able to split the data accordingly in testing and training, finding the following definition based on the dataset collection:
    • x_train and x_test: uint8 array of RGB image data with shape (num_samples, 32, 32, 3).
    • y_train and y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples, 1).

It can be observed too that the pixel values are in the range of 0 to 255 for each of the red, green and blue channels.

Now, with the datasets already loaded on our module, we have to consider what kind of treatment the data needs in order to be able to manipulate it. With that in mind, it was considered appropriate to transform the label class (Y axis), converting the class vector (integers) to binary class matrix with the to_categorical method defined on Keras.

Besides that, it’s good practice to work with normalized data. Because the input values are well understood, we can easily normalize to the range 0 to 1 by dividing each value by the maximum observation which is 255.

Note, the data is loaded as integers, so we must cast it to floating point values in order to perform the division.

Now, the core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers that will be used to create.
To start with, I defined a simple CNN model. I considered the use of a model with one convolutional layers followed by max pooling and a flattening out of the network to fully connected layers to make predictions.
I used Droupout as regularization technique, implementing it in at two points on our model, the first one right before flattening our data out with 25% and the second one right before the output layer with 50%.
As activation function, it was considered ReLU on every layer, except for the output one, using softmax for multi-dimentional evaluation in our CNN, considering an epoch of a 100 and a batch size of 32.
A logarithmic loss function is used with the stochastic gradient descent optimization algorithm configured with a large momentum and weight decay start with a learning rate of 0.1. We are using "categorical_crossentropy" for loss with metric "accuracy"
The above model worked with ~70% of accuracy. After some iterations it was noticed that there were 5 factors that in this case helped out to the optimization of the model.
    • Changing the Learning Rate (lr) on the Loss Function (from 0.1 to 0.01).
    • Including regularization L2 on each Convolutional Layer.
    • Adding more Convolutional Layers.
    • Increasing epoch (from 100 to 150)
    • Increasing the batch size (from 32 to 64).

Something important to consider on this experiment, is that the balance between Network Complexity vs Accuracy/Error_Loss is important, given the fact that increasing the number of epochs and the batch size allow us to improve in accuracy, creating a more complex model, demanding more resources from the system and taking longer time.
With the above given model, the time spent were around 1.5 hrs with the architecture described before, achieving ~83% of Accuracy.
An increment on epochs is considerably helping to improve the accuracy, but generates a system that will take days to be completed under present circumstances.

In conclusion, the proper analysis of the data, the understanding of the task and moving forward with fast iterations attempting to improve accuracy is what we should be considering appropriate when implementing a model. 

## REFERENCES:
[1] Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton, April 8 2009, https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf  
