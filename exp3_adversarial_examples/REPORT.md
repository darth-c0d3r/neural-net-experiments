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

#### Ouput on the generated images (White-Box)
Now the scores for the adversarial examples created by our network for the convnet (white-box)

<p float="left">
	<img src="results/fake0.png" alt="White-Box score for fake 0" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/0.jpg" alt="Fake 0" width="200"/>
</p>
<p float="left">
	<img src="results/fake1.png" alt="White-Box score for fake 1" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/1.jpg" alt="Fake 1" width="200"/>
</p>
<p float="left">
	<img src="results/fake2.png" alt="White-Box score for fake 2" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/2.jpg" alt="Fake 2" width="200"/>
</p>
<p float="left">
	<img src="results/fake3.png" alt="White-Box score for fake 3" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/3.jpg" alt="Fake 3" width="200"/>
</p>
<p float="left">
	<img src="results/fake4.png" alt="White-Box score for fake 4" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/4.jpg" alt="Fake 4" width="200"/>
</p>
<p float="left">
	<img src="results/fake5.png" alt="White-Box score for fake 5" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/5.jpg" alt="Fake 5" width="200"/>
</p>
<p float="left">
	<img src="results/fake6.png" alt="White-Box score for fake 6" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/6.jpg" alt="Fake 6" width="200"/>
</p>
<p float="left">
	<img src="results/fake7.png" alt="White-Box score for fake 7" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/7.jpg" alt="Fake 7" width="200"/>
</p>
<p float="left">
	<img src="results/fake8.png" alt="White-Box score for fake 8" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/8.jpg" alt="Fake 8" width="200"/>
</p>
<p float="left">
	<img src="results/fake9.png" alt="White-Box score for fake 9" width="450"/>
	<img src="adversarial_outputs_conv/non_targeted/9.jpg" alt="Fake 9" width="200"/>
</p>

As we can see above, we have created images which successfully fool the network. In fact, we see
that the scores of other (non intended) classes is even lower, on an average than it is for real
images. So, in a way, the network is even more confident about it's results on the adversarial images.


#### Ouput on the generated images (Black-Box)

We can see from the values (not shown here, check the results folder), the black box attack didn't quite
work as expected (or were the expectations wrong?). If the black box attack
would've been successful, the fc net would have also classified the images wrongly
with outputs same as the convnet. The fc net always seems to give the output 3
which is most likely due to high bias learned during training corresponding
to 3.
Note that this is not a failure. In fact much research has gone in to make
black-box attacks work and such simple method is not expected to work in the first place.

### Targeted Generation

The aim here is to create images that look like 7 but are classified incorrectly by the network
images for 0,3,4,5 were succesfully created with lambda = 10. 
For rest of the targets, I had to reduce the lambda as the image is still being classified as 7.
Here, lambda is the hyperparameter which controls how much the generated image looks like the desired image.
Also, please note that 7 is just an example and we can use any image corresponding to any number to run the attack.

#### Output on the original image

<p float="left">
	<img src="results/abs_orig7.png" alt="Score for Original 7" width="450"/>
	<img src="adversarial_outputs_conv/orig7.png" alt="Original 7" width="200"/>
</p>

#### Output on the generated Images (White-Box)

Score on the image generated for all targets from above image:

<p float="left">
	<img src="results/fake70.png" alt="White-Box score for fake 0" width="450"/>
	<img src="adversarial_outputs_conv/targeted/7_to_0.jpg" alt="Targeted Fake 0" width="200"/>
</p>
<p float="left">
	<img src="results/fake71.png" alt="White-Box score for fake 1" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_1.jpg" alt="Targeted Fake 1" width="200"/>
</p>
<p float="left">
	<img src="results/fake72.png" alt="White-Box score for fake 2" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_2.jpg" alt="Targeted Fake 2" width="200"/>
</p>
<p float="left">
	<img src="results/fake73.png" alt="White-Box score for fake 3" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_3.jpg" alt="Targeted Fake 3" width="200"/>
</p>
<p float="left">
	<img src="results/fake74.png" alt="White-Box score for fake 4" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_4.jpg" alt="Targeted Fake 4" width="200"/>
</p>
<p float="left">
	<img src="results/fake75.png" alt="White-Box score for fake 5" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_5.jpg" alt="Targeted Fake 5" width="200"/>
</p>
<p float="left">
	<img src="results/fake76.png" alt="White-Box score for fake 6" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_6.jpg" alt="Targeted Fake 6" width="200"/>
</p>
<p float="left">
	<img src="results/fake77.png" alt="White-Box score for fake 7" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_7.jpg" alt="Targeted Fake 7" width="200"/>
</p>
<p float="left">
	<img src="results/fake78.png" alt="White-Box score for fake 8" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_8.jpg" alt="Targeted Fake 8" width="200"/>
</p>
<p float="left">
	<img src="results/fake79.png" alt="White-Box score for fake 9" width="450"/>
	<img src="adversarial_outputs_conv/targeted/07_to_9.jpg" alt="Targeted Fake 9" width="200"/>
</p>

As we can see that by making just very small changes to an image which was originally
classified as 7 with high probability, we can change the output classification.
Similarly, we can generate adversarial examples for other images as well.

#### References:
https://ml.berkeley.edu/blog/2018/01/10/adversarial-examples/

https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

<!-- set style data histogram
set xtics (0,1,2,3,4,5,6,7,8,9)
set xrange [0:10]
set boxwidth 1
set style fill solid border -1
set title "Avg. scores for original 0"
plot 'scores'
 -->
