## Neural Network Experiments
These are some simple(!) Neural Network Experiments that I started doing in December 2018.
These are done only for learning purposes and it means that sometimes simplicity is preferred over accuracy/performance.
There might be better ways to do some things for which I might have chosen simple ways to make frequent observations easier.
I will keep adding to these when I do more work. For details, go to the folder of the corresponding experiment.
As of now, following are the contents:

#### 1. Binary Encoding vs. One Hot Encoding

Almost everywhere in literature, you will see that the output is encoded as a one-hot vector, i.e. the number of dimensions of the output is equal to the total number of classes, with the value corresponding to the correct class being 1 and rest all being 0. However, can we also use binary-encoding which only requires log(n) dimensions for target representation? How does it compare to the one-hot encoding?

#### 2. Transfer Learning

If you have a very small dataset, then, instead of training a model from scratch, you should consider using a pre-trained model, which has been trained on a similar task and retraining only some of its layers depending on the size of the dataset you have. If you have a very small dataset, training only some part of the network should give you better results as compared to retraining the complete network.

#### 3. Generating Adversarial Examples

In this experiment, I try to generate adversarial examples for a given convolutional network (white-box attack) for classifying MNIST digits. I try to generate both targeted and non-targeted images. Non-targeted means to simply generate an image that the network will predict as a particular number. Targeted image means an image that looks like a particular number to humans but is classified incorrectly by the network as some other number. I will also try to test the generated images on another network (black-box attack), although the expectations of success in this case are very low.