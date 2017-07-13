# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Center Lane"
[image3]: ./examples/recovery_1.jpg "Recovery Image"
[image4]: ./examples/recovery_2.jpg "Recovery Image"
[image5]: ./examples/recovery_3.jpg "Recovery Image"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

(code lines 95-110)
My model consists of a convolution neural network with three layers with filter size 5x5 and a 2x2 stride followed by two convolutional layers with 3x3 filter size and no stride. Filter depths range between 24 and 64.

The convolutional layes are followed by three fully connected layers with linear activations and a linear single-neuron output layer. Adam optimizer is used optimizing a Mean-Squared Error cost function.

The data is normalized in the model using a Keras lambda layer (code line 96) and the images are cropped using a Keras Cropping2D layer (code line 97).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 74-75). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The following hyper-parameters were tuned.

The number of Epochs were tuned based on the validation loss improvement and based on whether training from scratch or finetuning and existing model with additional data.

Although the model used an Adam optimizer, the learning rate was tuned manually in certain cases (see Solution Design Approach below)

The steering correction angle for right and left camera images was also tuned. Initial value of 0.1 was not successful in keeping the car in the center of the lane.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ("recovery laps"), driving counter-clockwise, using the left and right camera images with a steering angle correction.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use transfer learning by building on the basis of a successful architecture and finetuning it using the collected data.

I used a convolution neural network model based on to the research developed by the Nvidia team (research paper pubilshed 26 April 2016). I thought this model might be appropriate because of the successful results as documented by the paper.

In order to gauge how well the model was working, I split my image and steering angle data into a training and a validation set.

I used the following procedure for training. I initially collected around 8000 data points by driving smoothly in the center of the lane of track one. I trained for five epochs, as training beyond five epochs did not show any improvement in the validation loss. This model produced some positive results however the car was falling off the track at the first left turn.

Once, I had a first model showing some learning progress, I built on that by collecting more data (driving in counter-clockwise direction, recovery laps) and finetuning the initial model. After I had a model, which managed to drive successfully around the first turns, I collected more data and build on that, by carefully limiting the number of epochs, when the validation loss stopped improving to avoid overfitting.

When finetuning a model with a dataset larger than the dataset used to train the original model, I also lowered the starting learning rate for the Adam optimizer, in order to avoid the original model weights being substantially modified by the new data. I used the same logic when finetuning a "smooth driving" model with data from a recovery lap. I also used a smaller number of training epochs in these cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 95-110) consisted of a convolution neural network with the following layers and layer sizes:

Three convolutional layers with filter size 5x5 and a 2x2 stride followed by two convolutional layers with 3x3 filter size and no stride. Filter depths range between 24 and 64.

The convolutional layes are followed by three fully connected layers with linear activations and a linear single-neuron output layer. Adam optimizer is used optimizing a Mean Squared Error cost function.

The data is normalized in the model using a Keras lambda layer (code line 96) and the images are cropped using a Keras Cropping2D layer (code line 97).

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the lane. These images show what a recovery looks like starting from the right lane line back towards the center of the lane:

![alt text][image3]
![alt text][image4]
![alt text][image5]



To augment the data set, I also flipped images and angles in order to double the amount of available training data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also used left and right camera images with steering angle correction of intially 0.1, which I then tuned to 0.25. This produced better results in terms of keeping the car in the center of the lane, however it had a small negative side effect of a less smooth driving.

After the collection and augmentation process, I had approx. 46,000 data points. I then preprocessed this data by mean normalizing the images. As track one has many straights, which produce an unbalanced data set with steering angles close to zero dominating, I excluded steering angles in the range (-0.7, 0.7) in order to balance the dataset.

I also cropped the images, leaving only the relevant section of the road in order to speed up the learing process.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 3 and 5 depending on the type of training and the amount of data used. I used an Adam optimizer, manually setting the starting learning rate in certain cases as described above.



