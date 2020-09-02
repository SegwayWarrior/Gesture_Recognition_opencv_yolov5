# Hand Gesture Recognition with OpenCV, Pytorch, and YOLOv5


![Sign Language Detection](all_three.gif)

##   


## Overview


This project is a collection of three separate methods in computer vision to detect hand gestures. You can learn more about each project, as well as how to run them, in their individual README's: [Sign Language Detection](https://github.com/SegwayWarrior/Gesture_Recognition_opencv_yolov5/tree/master/sign_lang_detection), [Pytorch Classification](https://github.com/SegwayWarrior/Gesture_Recognition_opencv_yolov5/tree/master/pytorch_classification), and [Basic Thresholding](https://github.com/SegwayWarrior/Gesture_Recognition_opencv_yolov5/tree/master/basic_thresholding). The first uses object detection with YOLOv5 to recognize all the letters of the alphabet in American Sign Language (J and Z edited). The second uses machine learning principles in pytorch to create a CNN for image classification.  The third uses basic image thresholding with openCV to find contours of fingers after applying a mask. Below are instructions on setting up a virtual environment that will be able to run all three projects. There's also a summary on my thoughts about data collection, as well as insights, improvements, and ideas for machine learning in general.

## Setting Up a Virtual Environment
- Create a new environment with python 3.8.  If you'd like, you can do this with venv "python3.8 -m venv /path/venv" and enter the environment. "source /path/venv/bin/activate"
- Download the correct version of pytorch 1.6 for your computer: https://pytorch.org/
- Install dependencies " pip3 install -r requirements.txt"

## Datasets
- Having a large, diverse dataset is important to creating a good neural network.  Even if it appears that your model is improving with more epochs, it could be overfitting to the data it's given, meaning the model won't work in different situations.  You see this problem with detecting different skin tones because the training data wasn't diverse enough.
- This makes the problem of data collection a tough one.  Depending on class complexities, your network could require hundreds if not thousands of images from each class.  This means that anything we can do to automate the process will make machine learning a more usable tool.
- If your classifications are a subset of a separate class (ASL gestures are a subset of hands), then you can create a dataset by using a detector of the parent class. Another example of this would be using a dog detector to create a dataset to classify different dog breeds.
- If we know certain properties ahead of time, we can automate the dataset creation process in other ways.  For instance, I could have created a program to create the sign language classes with the bounding boxes saved as a roi. This way, I'd simply have to make the class gesture in the box shown on the screen when prompted to.  I also could have changed the roi during collection to diversify my data.

## Insights, Improvements, and Ideas
- As my Computer Vision professor, Ying Wu, often said, "Computer vision needs to think more like a human."  And indeed, there have been problems I noticed that a human would not have issues with.
- For the thresholding and classification projects, having backgrounds other than a blank wall would cause issues.  Even if the thresholding project averages the background, any slight movements can throw it off if it isn't relatively blank.  This problem might have been avoided in the classification project if the data provided had more diverse backgrounds to strengthen the network. Because of the data provided from EgoHands, the hand detector with yolo did not have this problem.  
- The hand detector did have issues with falsely classifying other parts of my body as a hand.  This would often happen with my arms, as well as my face when it was close to the camera.  This particular problem may have been avoided with more diverse data, but when comparing this to a human, it would be obvious an arm and a face isn't a hand because we already have classified those parts as such.  
- This brings up an important difference between computer vision models and humans, humans classify everything.  People don't see the world in a vacuum, meaning that if we're looking for hands, were not going to avoid seeing everything else.  Also, by classifying other objects, such as arms, we can get a good idea of where a hand should be.
- Humans have many different sub classes for things, which helps us confirm our predictions. For instance, there is this [article](https://www.wired.com/2015/01/simple-pictures-state-art-ai-still-cant-recognize/) about models being confused by abstract art, being certain that they are objects like school busses and insects.   Humans would not have this issue because we have subclasses for our internal school bus class, like windows, wheels, frame ext.  Without confirming these subclasses, we would lower our confidence in our original prediction.  Our current technology wouldn't have enough resources to classify all subclasses for every class at all times, but like humans, neural networks could change focus depending on the task at hand.
- Sub classes could help improve hand detecting as well.  If we could detect individual fingers, we could create a skeleton structure like the one shown [here](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html), which is the latest in hand tracking technology.  It creates a skeleton-like structure on hands by combining a cropped image from a palm tracker with a gesture recognizer for a discrete set of gestures.  This means that sub classing fingers could be an alternative solution to this problem.
- Sub classes could add another layer of clarification directly to the neural network.  Many neurons represent individual parts of a trained class, and when these neurons light up, that class is identified.  If we create a sub-class layer right before the output layer, we could tell the network to make a classification if it first can find it's subclasses. Of course this is just a thought, but as an enthusiastic machine learning novice, I'm excited to continue learning and experimenting with the most exciting technology of our time.


## Special Thanks
- A big thanks to my advisor, Professor Matthew Elwin.  Without his support, this project would not have been possible.
