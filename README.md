# Project 5: Vehicle Detection and Tracking

The goal of this project is to detect cars and track those vehicles throughout the video. Non-vehicle and vehicle images were trained using a neural network. Advanced Lane Detection and Vehicle Detection/Tracking projects are combined here. Since a custom neural network is used feature selections are done by the model. 

- Test Video Result:
  ![Test Video]("videos/test_video_out.gif")
 
- [Project Video]("https://www.youtube.com/watch?v=Qn0w2xHP8U0")

## Table of Contents ##
- [Project codes](#codes)
- [Data Information](#data)
- [Model](#model)
- [Pipeline](#pipeline)
- [Discussion](#discussion)

## Project codes <a name="codes"></a>
- vehicle_detection_tracking.py -- vehicle and non-vehicle data training also tracking vehicles.  
- video_generation.py -- processing video frames 
- laneheper.py,LaneDetector.py, Line.py -- helper functions for lane detection
- ImageUtils.py -- image processing functions

## Data Information <a name="data"></a>
 **(64x64x3)** vehicle and non-vehicle images are used for training and 10% of data is allocated for testing. Here is data summary:

	Training Data Size = 15984

	Test Data Size = 1776

	Shape of image = (64, 64, 3)

## Model <a name="model"></a>

First images are normalized followed by two 2D Convolution layers then max_pooling. Right after 50% of data is dropped between dense layers. Model image is shown below. 

![Model Information]("images/model_info.png")

## Pipeline 

Setting epoch=20 and batch_size=32, test accuracy is ~0.985 with 0.026 loss. For the last two epoch results are given below. 

![Test data result]("images/test_result.png")
  
test1.jpeg is used for heatmap and detecting the cars in the image.  

![Heat Map]("images/heatmap.png")

For video generation Advanced Lane Detection code and this project are combined for both detecting the lane lines also for car detection/tracking. 

## Discussion <a name="discussion"></a>

One of the challenges in the project was restricting heatmap to an area where cars will be available. When i didn't restricted trees and road signs were also detected. Also, the cars coming from other side of the road were detected. I noticed that size of the box keeps changing during the video process that needs to be improved.
