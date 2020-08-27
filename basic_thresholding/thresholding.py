#!/usr/bin/env python
# coding: utf-8

# Hand Gestures using openCV thresholds and masks

import cv2
import numpy as np
from sklearn.metrics import pairwise
import argparse

def segment_hand(frame, threshold):
    # Difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Use a threshold to find the hand in diff
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    return thresholded

def background_avg(img, background_weight):  # Average background together
    # For first time, create the background from a copy of the frame.
    global background
    if background is None:
        background = img.copy().astype("float")
        return None

    # Accumulate weights and update background
    cv2.accumulateWeighted(img, background, background_weight)

def count_fingers(thresholded): # Count fingers based on threshold and circular mask
    gesture = 'None'  # Gesture variables
    thumb = False
    pinky = False
    fing_conts = []  # List of finger contours

    cX = (roi_right - roi_left) // 2  # Center of roi
    cY = (roi_bottom - roi_top) // 2

    max_distance = (roi_right - roi_left) // 2  # Max distince of circular mask
    radius = int(0.75 * max_distance)  # Create a circle around the center

    # Draw roi and use bit-wise AND with threshold to find fingertip contours
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (((roi_right - roi_left) // 2), ((roi_bottom - roi_top) // 2)), radius, 255, 3)
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0  # Finger count starts at 0
    # fing_conts = []  # List of finger contours
    dist_check = True


    for cnt in contours: # Count fingers
        (x, y, w, h) = cv2.boundingRect(cnt)  # Bounding box of countour

        for fing_cnt in fing_conts: # Don't count contours if they are close together
            (x_f, y_f, w_f, h_f) = cv2.boundingRect(fing_cnt)
            fing_cntX = x_f + w_f/2
            fing_cntY = y_f - h_f/2
            cnt_distance = pairwise.euclidean_distances([(cX, cY)], [(fing_cntX, fing_cntY)])[0]
            dist_check = cnt_distance.max() > 0.07*roi_bottom and cnt_distance.max() > 0.07*roi_right

        out_of_wrist = ((cY + (cY * 0.4)) > (y + h))  # Check for wrist on bottom of roi
        if  out_of_wrist and dist_check:  # add to contour count
            fing_conts.append(cnt)
            count += 1
            if count > 5:
                count = 5

        # Add conditions to recognize thumb and pinky based on distance from center
        if not opt.switch_hand:  # switch sides for thumb and pinky if need be
            if not thumb:
                thumb = ((cX + (cX * 0.6)) < (x + w/2)) and out_of_wrist
            if not pinky:
                pinky = ((cX - (cX * 0.4)) > (x + w/2)) and out_of_wrist
        else:
            if not thumb:
                thumb = ((cX - (cX* 0.6)) > (x + w/2)) and out_of_wrist
            if not pinky:
                pinky = ((cX + (cX * 0.4)) < (x + w/2)) and out_of_wrist

        # Additional gestures from thumb and pinky recognition
        if count == 1 and thumb:                    gesture = 'thumb'
        elif count == 1 and pinky:                  gesture = 'pinky'
        elif count == 2 and thumb and pinky:        gesture = 'pinky and thumb'
        elif count == 2 and thumb and not pinky:    gesture ='L'
        elif count == 2 and not thumb and pinky:     gesture = 'horns'
        elif count == 3 and thumb and not pinky:    gesture = 'three_thumb'
        elif count == 3 and pinky and thumb:        gesture = 'rock out!'
        else: gesture = str(count)

        if opt.show_threshold:
            cv2.imshow('circ_roi',circular_roi)
    return count, gesture

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--switch-hand', action='store_true', help='switch side for pinky and thumb')
    parser.add_argument('--switch-side', action='store_true', help='switch side for roi')
    parser.add_argument('--show-threshold', action='store_true', help='show threshold, with and without circular mask')
    parser.add_argument('--threshold', type=int, default=7, help='threshold variable')
    opt = parser.parse_args()
    print(opt)

      # Backtround image that will be averaged together
    background = None
    background_collect = 0  # Trigger and count to start collecting background
    background_cnt = 0
    background_weight = 0.5  # Start with a halfway point between 0 and 1 of accumulated weight

    roi_top = 100  # Region of interest in pixels, where top left is (0,0)
    roi_bottom = 300
    roi_left = 10
    roi_right = 210

    if opt.switch_side:
        roi_left = 400
        roi_right = 600

    cam = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # horizontal flip
        frame_copy = frame.copy()

        # press b to reset background
        cv2.putText(frame_copy, 'press b to', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame_copy, 'set background', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]  # Grab the ROI from the frame
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Apply grayscale and blur to ROI
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # every time 'b' is pressed, create a new background with 60 frames
        if background_collect%2 == 1 and background_cnt < 60:
            background_cnt += 1
            background_avg(gray, background_weight)
            # if num_frames <= 59:
            cv2.putText(frame_copy, "Averaging Background,", (0, 30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(frame_copy, "  please remove hand.", (0, 70), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow("Finger Count",frame_copy)

        elif background_cnt == 60:
                background_collect += 1
                background_cnt = 0
                background_weight = 0.5 # reset weight when finished

        elif background_collect > 1:  # Dont continue without at least first background collected
            thresholded = segment_hand(gray, opt.threshold) # get thresholded hand image

            if thresholded is not None:
                kernel = np.ones((3,3),np.uint8)  # extrapolate the hand to fill dark spots within
                thresholded = cv2.dilate(thresholded,kernel,iterations = 1)
                fingers, gest = count_fingers(thresholded) # count the fingers

                # Display count and create a circle around the center
                cv2.putText(frame_copy, str(fingers), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                cv2.putText(frame_copy, gest, (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                max_distance = (roi_right - roi_left) // 2
                radius = int(0.75 * max_distance)
                cv2.circle(frame_copy, (((roi_left - roi_right) // 2)+roi_right,((roi_bottom - roi_top) // 2)+roi_top),radius,150,3)

                if opt.show_threshold:
                    cv2.imshow("Thresholded", thresholded)  # display the thresholded image

        # Draw ROI Rectangle on frame copy
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (255,255,0), 2)
        num_frames += 1
        cv2.imshow("Finger Count", frame_copy)

        pressedKey = cv2.waitKey(1)
        if pressedKey == ord('q'):  # q to quit
            break
        elif pressedKey == ord('b'):  # b to create background
            background_collect += 1

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()
