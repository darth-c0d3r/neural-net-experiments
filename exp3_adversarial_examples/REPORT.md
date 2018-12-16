## ADVERSARIAL EXAMPLES

#### Aim
The aim of this experiment is to learn and see how to generate adversarial examples for a given neural network (white-box attacks) and see how the examples generated on one network work on other networks (black-box attacks). Specifically, following is the breakdown of the task:
- Generate examples that cause network to mis-classify
- Generate examples that cause network to mis-classify and also look like real images
- Try the above 2 cases on another network (black-box attack)

#### Neural Network Models
We will use two networks, one will be a fully connected network and
the other will be a convolutional network with a fully connected output layer.

The fully connected network has the following layers: 128, 64, 32, 10(output)

**Accuracy at last epoch: 96.6600%**

The convolutional network has the following layers: 16, 32, 64, 10(output, fc)

**Accuracy at last epoch: 99.0700%**

Also, it must be noted that the output is coming from a sigmoid layer (instead of a softmax layer), so the output is not to be interpreted as probability but simply as a score.

### Non-Targeted Generation

#### Output on the original Dataset

For the conv net, the average scores for each class on eval set are:

<img src="results/orig0.png" alt="Average Scores for Original 0" width="450"/>
<img src="results/orig1.png" alt="Average Scores for Original 1" width="450"/>
<img src="results/orig2.png" alt="Average Scores for Original 2" width="450"/>
<img src="results/orig3.png" alt="Average Scores for Original 3" width="450"/>
<img src="results/orig4.png" alt="Average Scores for Original 4" width="450"/>
<img src="results/orig5.png" alt="Average Scores for Original 5" width="450"/>
<img src="results/orig6.png" alt="Average Scores for Original 6" width="450"/>
<img src="results/orig7.png" alt="Average Scores for Original 7" width="450"/>
<img src="results/orig8.png" alt="Average Scores for Original 8" width="450"/>
<img src="results/orig9.png" alt="Average Scores for Original 9" width="450"/>

What we can see from above graphs is that the score for correct class is always 1.

#### Ouput on thr generated images (White-Box)
Now the scores for the adversarial examples created by our network for the convnet (white-box)

<p float="left">
	<img src="results/fake0.png" alt="White-Box score for fake 0" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/0.jpg" alt="Fake 0" width="200" style="margin-bottom:50"/>
</p>

<img src="results/fake0.png" alt="White-Box score for fake 0" width="450"/>
<img src="results/fake1.png" alt="White-Box score for fake 1" width="450"/>
<img src="results/fake2.png" alt="White-Box score for fake 2" width="450"/>
<img src="results/fake3.png" alt="White-Box score for fake 3" width="450"/>
<img src="results/fake4.png" alt="White-Box score for fake 4" width="450"/>
<img src="results/fake5.png" alt="White-Box score for fake 5" width="450"/>
<img src="results/fake6.png" alt="White-Box score for fake 6" width="450"/>
<img src="results/fake7.png" alt="White-Box score for fake 7" width="450"/>
<img src="results/fake8.png" alt="White-Box score for fake 8" width="450"/>
<img src="results/fake9.png" alt="White-Box score for fake 9" width="450"/> -->

As we can see above, we have created images which successfully fool the network.

-------------------------------------------------------------------------------------------------------------------
Now the scores for the adversarial examples created by our network for the fc net (black-box)

0 [0.05877913 0.01591519 0.6059366  0.96270746 0.00231174 0.41980547
 0.01317467 0.1945288  0.3232576  0.10295683]
1 [6.8823050e-04 7.4350407e-07 4.4432890e-02 9.9984002e-01 1.1229281e-06
 9.8979771e-01 3.9122930e-05 6.4705720e-04 2.2827242e-01 1.2227631e-01]
2 [2.0495703e-05 6.0962734e-06 4.4688600e-01 9.9995613e-01 1.1776432e-07
 6.0836697e-01 1.4579266e-06 5.4606283e-03 1.1540348e-01 6.1128223e-03]
3 [1.6466048e-06 5.5332030e-06 7.1068817e-01 9.9998891e-01 1.2473334e-08
 2.0184138e-01 1.0894225e-07 9.7925793e-03 8.0448076e-02 1.2571333e-03]
4 [0.27252436 0.05956387 0.64972407 0.8594419  0.01274843 0.34597367
 0.07669801 0.374638   0.39253458 0.18789747]
5 [1.5180642e-02 1.0732048e-02 7.2851646e-01 9.8425204e-01 5.5801729e-04
 2.4834366e-01 3.9108223e-03 1.9213837e-01 3.1017029e-01 4.9440853e-02]
6 [0.20657197 0.12122905 0.78177416 0.8382791  0.02140305 0.21784441
 0.10101975 0.4316194  0.3834854  0.10665417]
7 [0.22910915 0.00218364 0.14038554 0.91541445 0.00553121 0.923186
 0.05785524 0.02372449 0.48182273 0.3675663 ]
8 [3.8741629e-03 2.2189273e-04 1.7859027e-01 9.9698764e-01 1.3061552e-04
 8.6280555e-01 6.1732507e-04 1.3194426e-02 3.8624200e-01 1.5628694e-01]
9 [0.03142137 0.00484307 0.3420504  0.9744076  0.00176277 0.5722772
 0.00494128 0.1300866  0.43006426 0.24039063]

As we can see from the above values, the black box attack didn't quite
work as expected (or were the expectations wrong?). If the black box attack
would've been successful, the fc net should have also classified the images wrongly
with outputs same as the convnet. The fc net always seems to give the output 3
which is most likely due to high bias learned during training corresponding
to 3.
Note that this is not a failure. In fact much research has gone in to make
black-box attacks work and such simple method is not expected to work in the first place.


===================================================================================================================
TARGETED GENERATION

we try to create images that look like 7 but are classified incorrectly by the network
images for 0,3,4,5 were succesfully created with lambda = 10. 
For rest of the targets, I have to reduce the lambda as the image is still being classified as 7.

Scores on the original image:
[2.4641e-04, 8.0803e-02, 8.3755e-01, 7.8569e-01, 6.2991e-07, 7.2013e-05, 2.0091e-13, 1.0000e+00, 2.7920e-02, 6.2292e-01]

Score on the image generated for all targets from above image:
0 [7.2646111e-01 4.8042592e-04 1.1623392e-01 1.0053798e-01 2.8093141e-03
 9.6997702e-03 8.6043979e-04 2.3279868e-01 1.7643560e-01 1.0823961e-01]
1 [7.3669031e-03 7.8452015e-01 8.5757077e-02 1.1645330e-01 7.8954414e-02
 1.6596096e-02 7.7292643e-05 2.0760222e-01 7.5230591e-02 1.0197759e-01]
2 [4.4798151e-02 1.3822880e-02 8.6160123e-01 8.9415260e-02 2.1558885e-02
 3.7237494e-03 3.7044185e-04 1.8570249e-01 1.5446174e-01 6.1126530e-02]
3 [7.95960706e-03 9.09306016e-03 7.58571774e-02 9.11802053e-01
 1.84914060e-02 4.77740020e-02 1.10946246e-04 1.84093744e-01
 1.13543093e-01 1.00613005e-01]
4 [1.3895081e-03 6.6926025e-02 5.7420570e-02 1.0205145e-01 8.4118372e-01
 2.3779195e-02 1.5891697e-05 2.7619311e-01 1.3727874e-01 1.5672967e-01]
5 [1.3338533e-02 3.3119333e-03 4.6915703e-02 1.8901421e-01 1.1967196e-02
 7.8503180e-01 3.5351457e-04 2.4668963e-01 1.6767070e-01 9.6295662e-02]
6 [4.2330451e-02 6.8059680e-04 6.2951393e-02 3.9305586e-02 2.8087718e-03
 9.7114705e-03 7.1007788e-01 1.7636974e-01 1.2193961e-01 9.3278095e-02]
7 [2.91054370e-03 7.22325072e-02 1.43713146e-01 1.04558125e-01
 1.02701859e-04 4.26046783e-03 8.12030088e-09 9.99994993e-01
 1.05506994e-01 1.52133659e-01]
8 [2.8638620e-02 7.0145815e-03 9.1755942e-02 1.0417175e-01 8.5038180e-03
 2.0063791e-02 3.3786651e-04 2.2828068e-01 9.1529310e-01 6.3613720e-02]
9 [2.92265625e-03 5.65171614e-03 4.55132127e-02 1.08002715e-01
 2.19995044e-02 2.96304226e-02 1.91185958e-04 2.32240379e-01
 9.35069919e-02 9.19098377e-01]

As we can see that by making just very small changes to an image which was originally
classified as 7 with high probability, we can change the output classification.
Similarly, we can generate adversarial examples for other images as well.

References:
https://ml.berkeley.edu/blog/2018/01/10/adversarial-examples/
https://pytorch.org/tutorials/beginner/fgsm_tutorial.html