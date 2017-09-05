**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictrue/20170906001505.png "Image Visualization"
[image1a]: ./pictrue/20170906001538.png "Train Count Visualization"

[image2]: ./pictrue/20170906001720.png "Grayscaling"
[image2train]: ./pictrue/20170906001754.png "Train Count Visualization"

[image2a]: ./pictrue/20170906001832.png "Normalized"

[image3]: ./pictrue/20170906020226.png "Traffic Sign"

[image4]: ./pictrue/20170906002113.png "My Sign K softmax prediction"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### Here is a link to my [project code](https://github.com/zhixiongruan/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It pulls in a random set of eight images and there labels.

![alt text][image1]

I also detail the train dataset structure by plotting the occurrence of each image class to get an idea of how the data is distributed.

![alt text][image1a]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the excess information only adds extra confusion into the learning process and in the technical paper it outlined several steps they used to achieve 99.7%.

Here is an example of traffic sign images after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the presence of singular sample data can cause the training time to increase and may cause it to fail. It is preferable to normalize the training before it is in the presence of the singular sample data.

I decided to generate additional data because several classes in the data have far fewer samples than others the model will tend to be biased toward those classes with more samples. 

To add more data to the the data set, I used the following techniques such as opencv translates affine and rotation

Here is an example of augmented images:

![alt text][image2a]

I increased the train dataset size to 89860 and also merged and then remade another validation dataset.  Now no image class in the train set has less then 1000 images.

![alt text][image2train]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet for the most part that was given, but I did add an additional convolution without a max pooling layer after it like in the udacity lesson.  I used the AdamOptimizer with a learning rate of 0.00097.  The epochs used was 27 while the batch size was 156.  Other important parameters I learned were important was the number and distribution of additional data generated.  I played around with various different distributions of image class counts and it had a dramatic effect on the training set accuracy.  It didn't really have much of an effect on the test set accuracy, or real world image accuracy.  Even just using the default settings from the Udacity lesson leading up to this point I was able to get 94% accuracy with virtually no changes on the test set. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9％
* validation set accuracy of 99.3%
* test set accuracy of 94.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 * I used a very similar architecture to the paper offered by the instructors.  I used it because they got such a good score the answer was given through it.
* What were some problems with the initial architecture?
 * The first issue was lack of data for some images and the last was lack of knowledge of all the parameters.  After I fixed those issues the LeNet model given worked pretty well with the defaults.  I still couldn't break 98% very easily until I added another convolution.  After that it was much faster at reaching higher accuracy scores.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
 * Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned.  For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals.  The batch size I increased only slightly since starting once I increased the dataset size.  The learning rate I think could of been left at .001 which is as I am told a normal starting point, but I just wanted to try something different so .00097 was used.  I think it mattered little.  The dropout probability mattered a lot early on, but after awhile I set it to 50% and just left it.  The biggest thing that effected my accuracy was the data images generated with random modifications.  This would turn my accuracy from 1-10 epochs from 40% to 60% max to 70% to 90% within the first few evaluations. Increasing the dataset in the correct places really improved the max accuracy as well.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 * I think I could go over this project for another week and keep on learning.  I think this is a good question and I could still learn more about that.  I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]


#### 2. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the results of the prediction:

![alt text][image4]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


