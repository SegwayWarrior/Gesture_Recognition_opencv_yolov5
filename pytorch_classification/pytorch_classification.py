#!/usr/bin/env python
# coding: utf-8

# Create a custom CNN for hand gesture classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from PIL import Image, ImageEnhance
from IPython.display import display
from shutil import copy
import cv2
import PIL
import argparse
import warnings

# Organize sign_lang data set for training
def sl_data_split(split_root , dest_root):
    for letter in  os.listdir(os.path.join(split_root)):
        for cnt,img in enumerate(os.listdir(os.path.join(split_root,letter))):
            cont = False
            if letter =='0': new_gest = '0'; cont = True
            elif letter =='1': new_gest = '1'; cont = True
            elif letter =='2': new_gest = '2'; cont = True
            elif letter == '6': new_gest = '3'; cont = True
            elif letter =='4': new_gest = '4'; cont = True
            elif letter =='5': new_gest = '5'; cont = True

            if cont == True:
                img_str = letter + '-' + img
                if cnt/len(os.listdir(os.path.join(split_root,letter))) < 0.75:  # split up for training and testing
                    if not os.path.exists(os.path.join(dest_root,'train',new_gest,img_str)):
                        copy(os.path.join(split_root,letter,img), os.path.join(dest_root,'train',new_gest,img_str))
                else:
                    if not os.path.exists(os.path.join(dest_root,'test',new_gest,img_str)):
                        copy(os.path.join(split_root,letter,img), os.path.join(dest_root,'test',new_gest,img_str))

# Use DataLoader to easily shuffle and load images
def data_loader(root):
    train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)
    global train_loader
    global test_loader
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=True, pin_memory=True, num_workers=8)



# Define the CNN model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) # 3 input layers for rgb
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6) # 6 catagories

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


def train_CNN(epochs, save_root):
    start_time = time.time()
    max_trn_batch = 800  # Limits very large datasets
    max_tst_batch = 300
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        for b, (X_train, y_train) in enumerate(train_loader): # Run the training batches
            # Limit the number of batches
            if b == max_trn_batch: break
            b+=1
            if torch.cuda.is_available(): # Turn tensors to cuda
                X_train = X_train.to('cuda', non_blocking=True)
                y_train = y_train.to('cuda', non_blocking=True)

            y_pred = CNNmodel(X_train)  # Apply the model
            loss = criterion(y_pred, y_train)

            predicted = torch.max(y_pred.data, 1)[1] # Tally the number of correct predictions
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            optimizer.zero_grad()  # Update parameters
            loss.backward()
            optimizer.step()

            if b%50 == 0:  # Print interim results
                print(f'epoch: {i:2}  batch: {b:4} loss: {loss.item():10.8f} accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

        train_losses.append(loss)
        train_correct.append(trn_corr)

        with torch.no_grad():  # Run the testing batches
            for b, (X_test, y_test) in enumerate(test_loader):
                if b == max_tst_batch: break  # Limit the number of batches

                if torch.cuda.is_available(): # Turn tensors to cuda
                    X_test = X_test.to('cuda', non_blocking=True)
                    y_test = y_test.to('cuda', non_blocking=True)

                y_val = CNNmodel(X_test)  # Apply the model
                loss = criterion(y_val, y_test)

                predicted = torch.max(y_val.data, 1)[1]  # Tally the number of correct predictions
                tst_corr += (predicted == y_test).sum()

        test_losses.append(loss)
        test_correct.append(tst_corr)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

    # Plot Results
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.title('Loss at the end of each epoch')
    plt.legend();

    plt.plot([t/9.3 for t in train_correct], label='training accuracy')
    plt.plot([t/3. for t in test_correct], label='validation accuracy')
    plt.title('Accuracy at the end of each epoch')
    plt.legend();

    torch.save(CNNmodel.state_dict(), save_root)

def load_weights(weights_root):
    if torch.cuda.is_available():
        CNNmodel.load_state_dict(torch.load(weights_root))
    else:
        CNNmodel.load_state_dict(torch.load(weights_root, map_location=torch.device('cpu')))
    CNNmodel.eval


# Test an image
def test_image(test_img_root):
    im2 = Image.open(test_img_root).convert('RGB')
    enhancer = ImageEnhance.Contrast(im2)  # change contrast for better results
    im2 = enhancer.enhance(.5)
    im2 = test_transform(im2)
    print(im2.shape)
    plt.imshow(np.transpose(im2.numpy(), (1, 2, 0)))

    if torch.cuda.is_available():
        im2 = im2.cuda()  # CNN Model Prediction
    CNNmodel.eval()
    with torch.no_grad():
        preds = CNNmodel(im2.view(1,3,224,224))
        print(preds)
        new_pred = preds.argmax()
    print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')


# Detect with camera
def cam_detect():
    cam = cv2.VideoCapture(0)
    CNNmodel.eval()

    roi_top = 100
    roi_bottom = 300
    roi_left = 10
    roi_right = 210

    if opt.switch_side:
        roi_left = 400
        roi_right = 600

    # Intialize a frame count
    num_frames = 0

    # keep looping, until interrupted
    while True:
        # get the current frame
        ret, frame = cam.read()
        height = frame.shape[0]
        width = frame.shape[1]
        frame = cv2.flip(frame, 1)  # flip the frame so that it is not the mirror view

        # create roi image
        frame_copy = frame.copy()
        roi_copy = frame.copy()
        roi_image = PIL.Image.fromarray(roi_copy, "RGB")
        enhancer = ImageEnhance.Contrast(roi_image)
        roi_image = enhancer.enhance(opt.contrast)
        enhancer = ImageEnhance.Brightness(roi_image)
        roi_image = enhancer.enhance(opt.brightness)
        roi_np = np.array(roi_image)
        roi_np = roi_np[:, :, ::-1].copy()

        roi = roi_np[roi_top:roi_bottom, roi_left:roi_right]
        if opt.show_roi:
            cv2.imshow("roi", roi)
        roi2 = PIL.Image.fromarray(roi, "RGB")

        # Draw ROI Rectangle on frame copy
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
        num_frames += 1

        roi2 = test_transform(roi2) # Convert for CNN
        if torch.cuda.is_available():
            roi2 = roi2.cuda()

        with torch.no_grad():
            preds = CNNmodel(roi2.view(1,3,224,224))
            new_pred = preds.argmax()

        # Display count
        cv2.putText(frame_copy, str(new_pred.item()), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Display the frame with segmented hand
        cv2.imshow("Finger Count", frame_copy)

        if cv2.waitKey(1) == ord('q'):  # q to quit
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # Filter  warnings

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-weights', nargs='+', type=str, default='gesture_numbers_CNNModel_ep25_personal.pt', help='model.pt path(s)')
    parser.add_argument('--data-split', action='store_true', help='use to split data')
    parser.add_argument('--split-root', nargs='+', type=str, default='', help='2 entries (data_root data_dest)')
    parser.add_argument('--train-data',  type=str, default='gesture_numbers_dk', help='triggers training and declares data location')
    parser.add_argument('--save-weights', type=str, default='default.pt', help='save results to *.pt')
    parser.add_argument('--test-image', type=str, default='', help='test the CNN on single image')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--contrast', type=int, default=2.5, help='edit roi for better cnn recognition')
    parser.add_argument('--brightness', type=int, default=.6, help='edit roi for better cnn recognition')
    parser.add_argument('--cam-detect', action='store_true', help='use model on default camera')
    parser.add_argument('--switch-side', action='store_true', help='switch side for roi')
    parser.add_argument('--show-roi', action='store_true', help='show roi after transformation')

    opt = parser.parse_args()
    print(opt)

    # Define image transforms
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.RandomRotation(10),      # up to 10 degrees of random rotation
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],     # standard normalization values
                                 [0.229, 0.224, 0.225]) ])

    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    inv_normalize = transforms.Normalize( # to reverse normalize to see image
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225] )

    # Create the model
    torch.manual_seed(31)
    CNNmodel = ConvolutionalNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
    if torch.cuda.is_available():
        CNNmodel = CNNmodel.cuda()
    CNNmodel

    if opt.data_split: # optional split data
         sl_data_split(opt.split_root[0], opt.split_root[1])

    if opt.train_data != '': # optional training
        data_loader(opt.train_data)
        train_CNN(opt.epochs, opt.save_weights)

    if opt.train_data == '': # load weights if no training
        load_weights(opt.load_weights)

    if opt.test_image != '':  # test single image
        test_image(opt.test_image)

    if opt.cam_detect:  # use model on camera
        cam_detect()
