#Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/feature_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bounded_box.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Histogram of Oriented Gradients (HOG)


#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first 3 code cells of the IPython notebook called project5.  

I read in all the `vehicle` and `non-vehicle` images and randomly selected one of each category and showed it below.  

![][image1]


I created the features with the following methods:
1. Spatial binning of color. The images from training data are resized to 16x16 and converted into a vector
1. Create histograms of color. Color of an image can help us to distinguish between a car and non-car
2. Histogram of Oriented Gradient (HOG) - I used scikit-image hog() method to detect a car by looking at its edges. HOG computes the gradietns from blocks of cells and then create a histogram from these gradient values

I extracted and combined the feature vectors into one for each image and then normalized the features. Here's an example of a car image and its features. 

![][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried different combination of parameters and trained a model to get its accuracy to compare and finally I settled with the values below because it gave me over 98% accuracy. 

For Hog: 

- orient = 8
- pix_per_cell = 8
- cell_per_block = 2
- hog_channel = 'All'

For spatial binning and create histograms of color:

- spatial = 16
- histbin = 32
- color_space = 'YCrCb'


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG and Colour features combined. First I normalized the training data and then randomly shuffled and split them into 80% training 20% test sets. This code can be found in code cell #4 of the notebook. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this is in cell #5 of the notebook. I restricted the search space to lower half of the image to capture the road instead of the sky and I used two different scales. Each one has different x_start_stop and y_start_stop and the overlap is set to .75. For each of the sliding window search:
- extract features for that window
- scale extracted features 
- feed extracted features to classifier
- predict if the window contains a car. If yes then I add that to the window list and return this list at the end


###### windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 700], xy_window=(64, 64), xy_overlap=(0.75, 0.75))
###### windows += slide_window(image, x_start_stop=[50, None], y_start_stop=[400, 700], xy_window=(96, 96), xy_overlap=(0.75, 0.75))
                       

Below is an image with the windows drawn.

![][image3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I played around with diffrerent parameters settings to extract the features. I used YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. When doing sliding window search, I restricted it to the lower half of the picture so it only focused on the road vs. the sky.

##### Stats: 
- Using spatial binning of: 16 and 32 histogram bins
- Feature vector length: 5568
- Total training images: 14208
- Total test images: 3552
- 11.08 Seconds to train SVC...
- Test Accuracy of SVC =  0.984

Here are some example images:

![][image4]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_solution.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

By using the sliding window search I was able to recorded the windows where the classifier detected a car. And since I had many overlapped windows, I used a heatmap, thresholded it and called scipy.ndimage.measurements.label to identify the vehicle positions. The goal is to combine multiple overlapped boxes into a single box that is a car. The code is located in cell #7 in the notebook. 

Here are some examples of showing the original pictures, heatmap applied, and the bounding boxes.

![][image5]



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent a lot of times playing with the window settings together with the heatmap threshold in order to eliminate most of the false positive. 

Currently the pipeline works relatively well. However, it can be improved more by making the window search/feature engineering to run more efficient. Also, I will work further eliminating the false positive and negative by keeping track of where the car was from the previous frame and used it to predict if it should be there in the current frame. 

There are many potential point of failures with the current pipeline if the video contains the followings:
- Non car objects (ie: pedestrians)
- The road shows 2 ways street with cars going the opposite direction
- The frame is pixelated
