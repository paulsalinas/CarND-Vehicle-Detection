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

The HOG features plus the spatial and color features were classified using SVM in [search_and_classify.py](search_and_classify.py) lines 80-87. Prior to classification data was sclaed to zero mean and unit variance:

```python
X_scaler = StandardScaler().fit(X)
```

## Sliding Window Search
The sliding window search can be found in [utils.py](utils.py) in the `find_cars` function.

I found only incremental improvement when implementing multiple sliding window sizes. For the sake of time, I stuck with only one scale. I'll discuss this result more in the end of this report.

Through trial and error, I found the step size 2 in the sliding window search was sufficient in capturing the features.

Here is an image demonstrating the search:
![sliding windows](./visualizations/test_1_find_cars.png "sliding window search")

### Classifier
The classifier in the sliding window search included the HOG features of all 3 channels in YCrCb. Further, the spatial and color features were added to improve the performance of the classifier.

## Video Implementation
The resulting video is [vehicle_detection.mp4](vehicle_detection.mp4).

To eliminate false positives, I limited the search to a certain boundary of the image where the cars are more likely to show up (not in the air or around the hood). I also implemented a 'heat map' to eliminate false positives. The heat map eliminated possibilities that didn't have many bounding boxes.

Here's a visualization of how the heat map helped filter a false positive in this case:

![sliding windows search with false positive](./visualizations/test_5_find_cars.png "false positive")

the resulting heat map:

![heat map](./visualizations/test_5_heat_map.png "heat map")

the result of using the heat map to filter false positives:

![result from heat map](./visualizations/test_5_car_positions.png "without false positives")


## Discussion
Currently my solution is a bit janky when drawing the bounding boxes. To address the smoothness of the vehicle detection, I would experiment with a more stateful pipeline that maintains the previous state of vehicle detection. Presumably, once cars are found, the previous position of the car can help aid in detecting the vehicle in the next frame. This includes moving and resizing the bounding box only incrementally which would mediate the janky behavior that I'm seeing right now in my video.

Secondly, I'm still noticing some false positives around some shadows. One approach is to augment the non-vehicle data and add shadows, and to use that additional data in the classifier.

Thirdly, I would experiment more with different sliding windows sizes and restricting the search area based on these sizes. I think when done carefully, this might help make the search algorithm more efficient by searching less windows if I understand it correctly.
