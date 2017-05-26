## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hsv]: ./output_images/hsv.png
[hog_car]: ./output_images/hog_car.png
[hog_nocar]: ./output_images/hog_nocar.png
[image1]: ./output_images/windows.png
[image2]: ./output_images/heatmaps.png
[image3]: ./output_images/stable.png

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


The basic image processing functions (color conversions, histogram, HOG) are defined in `Common.py`. `get_hog_features` is used to create the HOG features both from the training code (`Train.img_features()`) and the car detection code (`FindCars.find_cars_scaled()`). The invocation is similar in both cases, either only one channel is used for the HOG or all three. What is different is the input size, for the training the 64x64 samples are passed in, while the detection logic processes the whole image once as the windows are overlapping.


#### 2. Explain how you settled on your final choice of HOG parameters.

The parameters for the features preprocessing I tuned manually once the training pipeline was ready. I looked at how the SVM accuracy improved as I increased the size of the features, ie. started with one channel HOG, all channel HOG, added spatial features, increased spatial_size, added hist features, increased number of bins.

The final parameters for the features is set in the `FeatureConfig` class:

```python
class FeaturesConfig:

    def __init__(self):
        self.color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

        self.hog_feat = True  # HOG features on or off
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

        self.spatial_feat = True  # Spatial features on or off
        self.spatial_size = (32, 32)  # Spatial binning dimensions

        self.hist_feat = True  # Histogram features on or off
        self.hist_bins = 32  # Number of histogram bins

        self.y_start = 400
        self.y_stop = 650
```

This is the HSV color transform:

![alt text][hsv]

This how the non car hogs look like:

![alt text][hog_nocar]

This how the car hogs look like:

![alt text][hog_car]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training is implemented in the `Train` class. The `train()` method first prepares the non/car image features according to the configuration, then I use `StandardScaler` for normalization. I used a 20% test size for `train_test_split()`.

I used the `LinearSVC` classifier. For tuning the C parameter I used `GridSearchCV`, but it seemed the default C=1 was the best according to the crossvalidated grid search.

I could get high accuracies, close to 100%, but this was most likely the result of the training set leaking into test set with the randomly shuffled video sequences. I did not address this as the performane of the rest of the pipeline on the video was the most important goal.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In `FindCars` I used the method that did the HOG processing only once for the whole image and then scanned through the whole HOG features with the window. I left the overlap set at `cells_per_step = 2` which created a lot of overlapping hits for actual cars which is good for a strong signal at the heatmap stage.

The code supports multiple scales for the window size, but in the end I used only one scale: 1.25, because the results looked good on the test images, actual cars had lots of window hits and non cars not that much.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I processed a few samples of 2-3 seconds of the whole video to see which settings give good results at the more difficult sections.

The raw window search looked like this:

![alt text][image1]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./processed_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I implemented this logic in `StableCars`. Here I store the bounding boxes from the last N frames. On every frame I add all the boxes from the N last frames to the heatmap, but the older frames are added with an exponential decay (`alpha`), eg. the current bounding boxes have weight 1, the previous frame 0.9, before that 0.81, etc. I then apply the thresholding over this heatmap that is built from the last N frames.

This way the exponential decaying is tracking the real hot spots in time, the location information is not lost from frame to frame.

The processing is run from `Main` notebook, these were the parameters I used for the video:


```python
find = FindCars(svc['cfg'], svc['svc'], svc['scaler'], scales=[1.25])
stable = StableCars(find, threshold=20, last_n=10, alpha=0.9)
clip = VideoFileClip("project_video.mp4")
processed = clip.fl_image(stable.stablize)
```

This is how this looks like for 15 frames:

![alt text][image2]

I then use the `label()` function to create the final bounding boxes over the heatmap:

![alt text][image3]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There was a lot of image processing functionality to integrate in this project and a lot of attention was needed at the details. For example in the beginning when I put together the pipeline the training seemed succesful, but the car finding logic found cars all over the place. It turned out I concatenated the hog/spatial/histogram features in different order during the training and processing, but since the dimensions matched, there was no error, just poor results.

There are a lot parameters in the pipeline that could be tuned to improve performance. The biggest challange was finding a good stabilizing algorithm to filter out false cars, but not the real ones. A lo t of experemintation was needed to find a decay rate, frame lookback and threshold to achieve a stable looking result.

Since the result could only be manually checked and video processing was slow I could not explore too much of the parameter space, there could be much better combinations of the features/stabilizing parameters.

And even if I found even better parameters, these would perform better on this 1 minute recording, but on a different road with different cars under different lighting conditions they may not perform at all. The robustness needs to be tested with many different recordings.