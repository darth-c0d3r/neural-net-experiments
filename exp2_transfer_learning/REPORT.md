## TRANSFER LEARNING
#### Introduction
The aim of this simple experiment is to implement transfer learning by first training a network to classify a set of objects A and then keep parts of this network fixed and retrain the rest of the network to predict the same for a disjoint set of objects B. My aim is to get a basic idea of how much of the network is required to be retrained and with how much lesser data can we get a similar performance.

#### Dataset:
The dataset used is the CIFAR10 dataset. It has 10 classes. As "Set A", I will
use any 5 classes and for "Set B", I will use the remaining 5 classes. I use the images with labels 0-4 as "left dataset" and the images with labels 5-9 as "right datset".

#### Neural Network Model:
The best models for CIFAR10 report accuracies of greater than 90%. But since the aim of this experiment is not achieving higher accuracies, I have used a simple neural network that gives an accuracy of about 74% on the complete dataset. This model gives around 80% accuracy on the left dataset and 85% on the right dataset. We will use the model trained on the right dataset to check the performances on left dataset.

The model I have used has 5 layers: 3 conv layers, 1 fully connected hidden layer and 1 fully connected output layer.

#### Observations

Model Retrained | Dataset Size | Accuracy
--------------- | ------------ | -----------
0 (no layer) | 100% | 0.069200
1 (output layer) | 100% | 0.696000
2 (fc layers) | 100% | 0.761200
3 (1 conv) | 100% | 0.780800
4 (2 conv) | 100% | 0.787800
5 (retrained all) | 100% | 0.794800
0 (no layer) | 75% | 0.069200
1 (output layer) | 75% | 0.649600
2 (fc layers) | 75% | 0.759000
3 (1 conv) | 75% | 0.768800
4 (2 conv) | 75% | 0.781000
5 (retrained all) | 75% | 0.787000
0 (no layer) | 50% | 0.069200
1 (output layer) | 50% | 0.647400
2 (fc layers) | 50% | 0.751400
3 (1 conv) | 50% | 0.755200
4 (2 conv) | 50% | 0.763800
5 (retrained all) | 50% | 0.764400
0 (no layer) | 25% | 0.069200
1 (output layer) | 25% | 0.646000
2 (fc layers) | 25% | 0.733200
3 (1 conv) | 25% | 0.735200
4 (2 conv) | 25% | 0.740800
5 (retrained all) | 25% | 0.728800
0 (no layer) | 10% | 0.069200
1 (output layer) | 10% | 0.635800
2 (fc layers) | 10% | 0.708200
3 (1 conv) | 10% | 0.706200
4 (2 conv) | 10% | 0.700600
5 (retrained all) | 10% | 0.695200


#### Interpretation
If we just go crazy and use the model trained on the left database and see the results it gives for the right database and if everything goes well, the
accuracy should be less 20% (nothing better than random). Okay, we get an accuracy of around 6.5%, which is a little less than what I expected but then again, hey, it's not the data the model was made for.Okay, so now let's get started with the actual experimentation.

#### Results
So as expected, as we keep decreasing the size of the dataset, the maxima of
accuracy keeps shifting towards the model in which more layers are kept frozen. It is not until we cut the data to one-fourth that we observe a decrease in accuracy with increasing the amount of model we are retraining.

Also, the more data we use, the better accuracy we get for a given model which means that more data almost always beats other optimizations and has more increasing returns.

REFERENCES:
Transfer Learning Refresher: https://towardsdatascience.com/transfer-learning-946518f95666

<!-- plot 'one_hot.dat' with linespoints , 'binary_encoding.dat' with linespoints
plot 'data100.dat' with linespoints , 'data75.dat' with linespoints , 'data50.dat' with linespoints , 'data25.dat' with linespoints , 'data10.dat' with linespoints -->