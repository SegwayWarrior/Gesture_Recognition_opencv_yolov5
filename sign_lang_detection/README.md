# Sign Language Detection
<img src="sl_run.gif">

## Overview
This project uses the latest version of Yolo (You only look once) YOLOv5, which can be found here https://github.com/ultralytics/yolov5.  As they state themselves "This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets..." With this software, I was able to use the egohands dataset to create a simple hand detector.  Then I modified the detect.py code provided to save images with bounding boxes to different classes.  Since my sign language dataset has all images of myself, you will probably not see the same results.  If you'd like, you can use create_dataset.py with exp14_hand_best.pt to make your own hands dataset, and combine them with my dataset to make a more robust sign language detector.

## Details
- The yolo5 directory is a copy of the yolov5 link above.  If you'd rather keep yolo up to date, you can fork their repo and simply add create_dataset.py to the main directory.
- The modified dataset, with some of my own images included, can be found here:
- create_dataset.py is a modified version of detect.py.  The parts that are modified are separated by lines of hashes.  The arguments shown below are in addition to the ones from detect.py.
- "--create-ds" tells the program to create a dataset.  If this is not called, the program runs the same as detect.py.
- "--gest-class CLASS" declares which class to save to.  The class directory must already be created.  If you are not saving the classes as integers, as in my case, you'll need a dictionary to convert each string to an integer, starting at 0.
- "--label-dest" used with --create-ds to define the destination directory for the labels.  Make sure the directory has the class directory defined beforehand.  
- "--img-dest" same as --label-dest but for the images.
- "--bb-img-dest" same as --img-dest, but the images include the bounding boxes.  This is useful for double checking the accuracy of the hand detector.
- "--pic-count COUNT" declares the number of images to save to the class. The default value is 400.
- To get the weights for the hand detector, download from here and save it in the main yolo5 directory: https://drive.google.com/file/d/18KoHuoSnHEpGNnnYcFLbC6pbZtvELr5X/view?usp=sharing
- The weights for the sign language detector: https://drive.google.com/file/d/12gRRFeFacBzISP9PKhzkXlU5kGIbNt5v/view?usp=sharing
- The modified dataset used for building the hand detector: https://drive.google.com/file/d/1gpTev25ZnIrOv04ZHm3A-HnomQslxue3/view?usp=sharing
- The personal dataset used for sign language detection: https://drive.google.com/file/d/11Dr8n-bcBUgAh_C5mNOkMT5nLcKxw4zN/view?usp=sharing
- The original EgoHands dataset: http://vision.soic.indiana.edu/projects/egohands/.
