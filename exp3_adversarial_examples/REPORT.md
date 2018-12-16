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

#### Performance of the original Dataset

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


0 [0.99827651 0.00107352 0.19915122 0.00230315 0.02037038 0.03150682 0.12332517 0.00636423 0.04749502 0.11014814]

1 [0.01130811 0.99803437 0.08741341 0.00664729 0.43187734 0.00341541 0.04080755 0.27348676 0.2808405  0.02168577]

2 [3.45153704e-02 2.04207668e-01 9.99606821e-01 4.37718560e-01 1.64276470e-02 3.87474374e-04 4.28899294e-03 4.68803005e-01 4.02274479e-01 3.63151769e-03]

3 [3.93725270e-03 2.71125065e-03 1.12034295e-01 9.98407567e-01 6.51221011e-04 3.09294434e-01 4.59893711e-06 5.02456464e-02 3.36546376e-01 1.97216732e-01]

4 [0.0088929  0.05708734 0.03471481 0.00372155 0.99945771 0.00245556 0.02462172 0.35727438 0.17640862 0.30915913]

5 [0.01478805 0.00691398 0.00215056 0.53942348 0.00227824 0.99811810 0.0480153  0.0031307  0.57343663 0.36211180]

6 [3.28724185e-01 1.26901875e-02 1.91120917e-02 1.36320247e-03 2.58083709e-01 2.84733839e-01 9.95972034e-01 4.29429949e-04 2.84621600e-01 2.23365698e-03]

7 [6.38533036e-03 2.98667008e-01 5.13740084e-01 4.31665474e-01 5.56895785e-02 8.90392150e-03 3.74745462e-07 9.99360495e-01 1.30370445e-01 5.61783650e-01]

8 [0.06039144 0.00193241 0.27119904 0.31153516 0.01770592 0.14298959 0.05586447 0.01831253 0.99795999 0.11259448]

9 [8.61236427e-03 7.16922607e-03 8.80291929e-03 2.06367847e-01 5.48089553e-01 6.43413048e-02 2.90688731e-04 3.26558079e-01 4.62640403e-01 9.97027403e-01]

-------------------------------------------------------------------------------------------------------------------
Now the scores for the adversarial examples created by our network for the convnet (white-box)

0 [9.78767395e-01 2.54253700e-05 1.06392726e-02 1.13833789e-02
 7.01421965e-03 1.38713885e-02 1.11260470e-02 8.27130023e-03
 1.54854199e-02 6.35021226e-03]
1 [3.6884553e-04 9.9453521e-01 4.6057538e-03 5.2676369e-03 5.2410839e-03
 5.3107343e-03 3.9703799e-03 4.1922471e-03 1.9866396e-03 2.0578378e-04]
2 [2.7055218e-04 1.2034425e-03 9.9651849e-01 4.4983584e-03 4.1027670e-03
 5.6306231e-03 4.9696132e-03 5.4429984e-03 8.4642032e-03 1.9811242e-04]
3 [6.6704638e-07 3.2353622e-03 2.2751486e-03 9.9936742e-01 2.4805854e-03
 4.0029921e-03 1.6945103e-03 3.2501472e-03 3.8639340e-03 3.8488770e-03]
4 [3.1547120e-04 3.9747325e-03 5.2608522e-03 3.9176936e-03 9.9640852e-01
 7.0172455e-03 4.4298857e-03 5.7286872e-03 1.0696190e-02 5.7397657e-03]
5 [0.00200592 0.00210634 0.00841839 0.01131465 0.00534858 0.9970367
 0.00677321 0.00631799 0.0129451  0.00696439]
6 [3.2831859e-03 1.9883022e-03 5.8907368e-03 7.3424443e-03 3.5572133e-03
 5.3591868e-03 9.9993527e-01 3.8616627e-03 6.4407815e-03 1.1023811e-04]
7 [0.00317204 0.0038118  0.00998867 0.00722532 0.00517269 0.0079506
 0.00501313 0.9962559  0.0064078  0.0044176 ]
8 [4.5874498e-05 1.6314649e-03 4.9649864e-03 3.7558458e-03 2.7556666e-03
 3.6993870e-03 2.0904655e-03 3.5752403e-03 9.9708325e-01 2.9426618e-03]
9 [2.2136266e-03 6.6364545e-04 4.0064822e-03 4.1779606e-03 3.5769751e-03
 4.5019966e-03 3.3065761e-03 3.4115920e-03 5.0764880e-03 9.9619687e-01]

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