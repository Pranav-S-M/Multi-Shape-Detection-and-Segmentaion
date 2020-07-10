# Multi-Shape-Detection-and-Segmentaion

This is an exercise in object detection and image segmentation using CNNs (implemented with keras using tensorflow backend). 

## Training Data
The dataset consists of five different shapes (Rectangle, Circle, Triangle, Pentagon and Hexagon) and the bounding boxes and image masks for all the images containing a combination of these shapes (maximum of three objects per image). The overlap between these objects are controlled by calculating the IoU scores for the individual bounding boxes and the default is set at 5% - meaning the bounding boxes in the default setup will not overlap more than 5%. 

The training data can be created using the [Data Generator](https://github.com/Pranav-S-M/Multi-Shape-Detection-and-Segmentaion/blob/master/Data_Generator.py).

## Outputs
In this [notebook](https://github.com/Pranav-S-M/Multi-Shape-Detection-and-Segmentaion/blob/master/multi-object-detection-and-segmentation.ipynb), a simplified version of the YOLO architecture is utilized to achieve object detection while a U-net is used inorder to achieve image segmentation.

Sample output of detection

![Detection output](https://user-images.githubusercontent.com/49246680/87068060-aaec7d00-c232-11ea-9d3d-0e9d20bfb4ca.png)

Sample output of segmentation

![Segmentation output](https://user-images.githubusercontent.com/49246680/87068110-bb9cf300-c232-11ea-8d14-c9c5e314fbbf.png)

