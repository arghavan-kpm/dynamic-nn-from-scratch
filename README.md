# Implementing a Dynamic Neural Network from Scratch

This repository is an object-oriented implementation of a neural network using `only python`, **without** taking advantage of any mathematical library even `numpy`. You may ask when there are lots of useful and effecient libraries out there to do complex operations like matrix multiplication very fast, why someone might need to implement all of them from scratch? The answer is that as long as you do not know how different operations are computed exactly, you cannot think about weaknesses in existing approaches or inventing a new baseline.


## Background
Here, neurons, leyers, and our network are different objects. In their corresponding python classes, their unique functions are implemented and they work together in a heirarchical manner. Before strating with setup process, let's take a look at a brief explanation about concepts of a neural network. If you are familiar with the topics, you can simply skip to [Execution Section](https://github.com/arghavan-kpm//dynamic-nn-from-scratch#Execution). You can find more about neural networks at this tutorial (<http://cs231n.github.io/neural-networks-1/>).

![](https://github.com/arghavan-kpm/dynamic-nn-from-scratch/figures/mlp.PNG)

* In a simple image classification problem, we have more than one categories and we want the network to compute the probability of each input image being in each category. Then we select a category for each image that the network reports the highest probability for. 
* In the image above, we have a 2-layer Neural Network (one hidden layer of 4 neurons and one output layer with 2 neurons), and three inputs. You may or may not count input layer as one of the network layers.
* Final output of network is a linear or non-linear combination of all neuron outputs. _firing rate_ of the neuron is determining whether its output can take apart in the final output or not. We model the _firing rate_ of the neuron with an **activation function**. Commonly used activation functions are **Sigmoid**, **Tanh**, and **ReLU** non-linearities and you can see how they squash real numbers to different ranges in images below. From left, they are sigmoid, tanh, and reLU non-linearities, respectively.

![](https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/sigmoid.png) ![](https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/tanh.png) ![](https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/ReLU.png)

* 

This Dynamin net can have multiple layers and number of neurons in each layer can be changed. Implemented activation functions are 1)Linear and 2)Sigmoid. You can have 1)Drop out or 2)L2 Norm for regularization and 1)Gradient Descent or 2)Stochastic Gradient Descent for loss function.
Dataset should be placed in a directory called "DataSet" in the main directory.
For loading weights and biases you can use "Net_bgd.net" in the main directory.

