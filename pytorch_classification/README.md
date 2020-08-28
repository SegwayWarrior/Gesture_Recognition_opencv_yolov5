# Pytorch classification
<img src="classification.gif">

## Overview
This project was created using classic  classification techniques with machine learning in pytorch.  Six classes from an American Sign Language dataset were used to recognize the numbers 0 - 5.  Notice image classification is different from object detection in that the part being classified must be the primary object in the image, which is why we need a region of interest (roi) for classifying from a camera.  

## Details
- For information on how this Convolutional Neural Network (CNN) was constructed, check out the Machine Learning Basics section in the main page's README.
- To test the model on your default camera, use "--cam-detect".
- The code to organize the original ASL dataset is included, but should not be required since the updated dataset is included in gesture_numbers_ds.  To split data, use "--data-split" and "--split-root PATH/ROOT PATH/DESTINATION".
- The weights used are included as "gesture_numbers_CNNModel_epoch25_personal.pt", which is the default weights file.  If you'd like to change the weights, use "--load-weights PATH/WEIGHTS.pt"
- To train your own model, use "--train PATH/DATA", where the DATA directory is structured like gesture_numbers_ds. This means having a train and test directory, with both having a directory for each class.
- When training, use "--save-weights PATH/WEIGHTS.pt".  If "--save-weights" is not defined before running, the weights will be saved to default.pt.
- To change the number of epochs use "--epochs EPOCHS" where the default is 25.
- To test on a single image, use "--test-image PATH/IMAGE".
- Based on lighting and other factors, changing the brightness and contrast of the roi image may be useful.  To do so use, "--contrast CONTRAST" and "--brightness BRIGHTNESS" where the default values are both 1, meaning no change to original roi.
- To switch the roi to the other side of the screen, use "--switch-side".
- To show the roi, use "--show-roi"
- The original ASL datset can be found here: https://www.kaggle.com/ardamavi/sign-language-digits-dataset
