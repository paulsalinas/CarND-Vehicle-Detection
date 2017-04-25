# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## The Project
Identify and highlight videos in a video.

The main files of the project and what they were used for:
* [search_and_classify.py](search_and_classify.py) - extract the features based on certain parameters and then classify using SVM.
* [utils.py](utils.py) - miscellaneous funcs that I use throughout the project that do things like: extract features, sliding window search, etc.
* [pipeline.py](pipelie.py) - the main pipeline function that takes an image, searches for the cars, and returns an image with bounding boxes around the cars drawn
* [movie.py](movie.py) - running the pipeline against the video and outputting the result of this project

## Histogram of Oriented Gradients (HOG)
The parameters for feature extraction are in [search_and_classify.py](search_and_classify.py) lines 23-33. I've also outlined the parameters here:

| Param         | Value         |
| ------------- |:-------------:|
| color_space |YCrCb |
| orient |9 |
| pix_per_cell |8 |
| cell_per_block|2 |

I found that through trial and error, that these parameters gave the best results and gave the least false posives.

## Sliding Window Search

## Video Implementation

## Discussion
