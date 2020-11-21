# Implementing a Dynamic Neural Network from Scratch

This repository is an object-oriented implementation of a neural network using `only python`, **without** taking advantage of any mathematical library even `numpy`. You may ask when there are lots of useful and effecient libraries out there to do complex operations like matrix multiplication very fast, why someone might need to implement all of them from scratch? The answer is that as long as you do not know how different operations are computed exactly, you cannot think about weaknesses in existing approaches or inventing a new baseline.


## Background
Before strating with setup process, let's take a look at a brief explanation about concepts of a neural network. If you are familiar with the topics, you can simply skip to [Execution Section](https://github.com/arghavan-kpm/dynamic-nn-from-scratch#Execution). You can find more about neural networks at this tutorial (<http://cs231n.github.io/neural-networks-1/>).

<p align="center">
	<img src="https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/mlp.PNG">
</p>

* In a simple image classification problem, we have more than one categories and we want the network to compute the probability of each input image being in each category. Then we select a category for each image that the network reports the highest probability for. Each selection has a cost. If the network selects the correct category, this cost will be small.
* In the image above, we have a 2-layer neural network (one hidden layer of 4 neurons and one output layer with 2 neurons), and three inputs. You may or may not count input layer as one of the network layers.
* Final output of the network is a linear or non-linear combination of all neuron outputs. _Firing rate_ of the neuron is determining whether its output can take apart in the final output or not. We model the _firing rate_ of the neuron with an **activation function**. Commonly used activation functions are **Sigmoid**, **Tanh**, and **ReLU** non-linearities. In images below from left, you can see how sigmoid and tanh non-linearities squash real numbers to different ranges, respectively.

![](https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/sigmoid.PNG) ![](https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/tanh.PNG)

* A neural network should be _trained_ to find the best _parameters_ that help the model select correct answers and minimize the _cost_ (loss) on our data. These parameters are **weights** and **biases** of the neural network and the cost is computed by a **loss function**. 
* You can train or update parameters of your neural network in different ways. One of them is **Batch Gradient Descent** (_GD_) and the other is **Stochastic Gradient Descent** (_SGD_). In GD, you need to calculate the gradient of the loss function over **all samples** in your training data. In contrast, you only need the gradient of the loss function for one sample in SGD. As you can imagine, GD takes longer to complete, but its path to minima is less noisier than that of SGD.
* A regularization term can control overfitting of your neural network. Different types of that is **L2 regularization**, **dropout**, and **input noise**.

## Implementation
Here, neurons, leyers, and our network are different objects. In their corresponding python classes, their unique functions are implemented and they work together in a heirarchical manner. 

You can specify number of layers and neurons in each of them for your customized network as arguments of the code. For example, `[784, 100, 10]` represents a 2-layer network with `784` inputs, `100` neurons in its `first` layer, and `10` neurons in its `second` or last layer. 

In each layer, you can choose an activation function and a regularizer. Differnt choices of activation function in this implementation are `sigmoid` and `linear`. Also, `dropout` and `L2 norm` are implemented as differnt choices of regularizers that you can use at any layer you want. For example, `['sigmoid','linear']` and `['dropOut','nothing']` means first layer has `sigmoid` activation function and `dropout` regularizer, but second layer has neither activation function nor regularizer.

At last, you can decide the amount of learning rate and type of the gradient descent to update parameters of your network by choosing `GD` or `SGD`.

Training data is images of classes `A to J` of `noMNIST` training dataset which are in folder `ROOT/data/` by default.

Initial weights of a 2-layer neural network are in `ROOT/weights/w.net`. For a differnt size of a network you can change `w.net` accordingly. 

Because outputs of a multi-class classification network are probabilities, the activation function for the output layer will be **softmax** and **cross-entropy loss** (also known as negative log likelihood) is used as loss function.

_**Note:** There is not a seperate test data to evaluate the model and it only reports training accuracy._


## Execution
To do training a neural network with 2 layers with the setting described above for 2 epochs, run:
```
python main.py --num_epochs 2 --optimizer GD --layers 784,100,10 --active_funcs linear,sigmoid --regularizers dropOut,nothing --lr 0.7
```

Sample execution output:
<p align="center">
	<img src="https://github.com/arghavan-kpm/dynamic-nn-from-scratch/raw/master/figures/output.PNG">
</p>
