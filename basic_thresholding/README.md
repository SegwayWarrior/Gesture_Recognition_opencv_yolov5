# Basic Thresholding
<img src="finger_count.gif" />

## Overview
This project uses openCV to create a threshold image of the hand (top right image), then applies a circular mask to count the contours of extended fingers (bottom right image).

## Details
- An example run: "python thresholding.py --show-threshold --threshold 8"
- The default region of interest (roi) is on the left side of the screen.  If you'd like to move the roi to the other side, use "--switch-side" .
- First the background of the roi must be averaged over 60 frames to differentiate the hand when it's present.  Press 'b' at any time to set the background, making sure your hand is clear.
- Different lighting and other factors can change the tolerance needed on the threshold.  To change this, use "--threshold VALUE" where the default value is 7.
- This project assumes that the user's hand is in the center of the roi, which means certain parts of the hand will be in certain places.  With this we can exclude any contours on the bottom of circular masked image since we know those are from the wrist.  We can also assume that the thumb and pinky will be at least a certain distance away from the center in the X direction.
- Once we know which digits are thumb and pinky, we can define gestures based on if either of them are present and the number of digits detected in total.  Other gestures are "thumb, pinky, pinky and thumb, L, horns, three thumb, and rock out!"
- If you'd like to change which hand you're using, aka changing the sides of thumb and pinky, use "--switch-hand"
- To see the threshold and circular mask images, use "--show-threshold"
