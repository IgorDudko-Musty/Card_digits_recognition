# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:37:01 2025

@author: iGOR
"""

import numpy as np
import cv2
import imutils
from imutils.contours import sort_contours
import os

class Image_Proc():
    
    def __init__(self,
                 images_path=r'../data_for_test/cc01.png'):
        # self.images_list = []
        # for image in os.listdir(images_path):
        #     path = os.path.join(images_path, image)
        #     self.images_list.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        
        self.image = cv2.imread(images_path, cv2.IMREAD_GRAYSCALE)
        self.num_roi = [(40, 210), (640, 290)]
        
    
    # def __getitem__(self, index):
    #     return self.images_list[index]
    
    # def __len__(self):
    #     return len(self.images_list)
    
    def digit_extract(self):
        digits_arr = []
        image = self.card_transform(self.image)
        
        top_left_y = self.num_roi[0][1]
        bottom_right_y = self.num_roi[1][1]
        top_left_x = self.num_roi[0][0]
        bottom_right_x = self.num_roi[1][0]
        
        roi = image[top_left_y:bottom_right_y, 
                    top_left_x:bottom_right_x]
        _, th = cv2.threshold(roi, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th.copy(), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, 
        #                   key=cv2.contourArea, 
        #                   reverse=True)
        contours = sort_contours(contours, 
                                 method="left-to-right")[0]
        for i in range(len(contours)):
            c = contours[i]
            # compute the bounding box for the rectangle
            (x, y, w, h) = cv2.boundingRect(c) 
            if contours[i][:,:,1].min() < 2:
                continue
            if contours[i][:,:,1].max() > roi.shape[0] - 2:
                continue
            if w > h:
                digits_arr.append('Bad contour')
                continue
            if w >= 5 and h >= 25 and cv2.contourArea(c) < 1000:
                roi_num = roi[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_num, 
                                         (32,32), 
                                         interpolation=cv2.INTER_AREA)
                digits_arr.append(roi_resized / 255.)
        return digits_arr
                
        
    
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        (tl, tr, br, bl) = pts
        rect = pts

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [690, 0],
            [690, 430],
            [0, 430]], dtype 
            = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (690, 430))
        return warped

    def card_transform(self, image):
        orig_height, orig_width = image.shape[:2]
        ratio = image.shape[0] / 500.0

        orig = image.copy()
        image = imutils.resize(image, height = 500)
        orig_height, orig_width = image.shape[:2]
        original_Area = orig_height * orig_width
        
        # convert the image to grayscale, blur it, and find edges
        # in the image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_bordered = cv2.copyMakeBorder(image, 
                                            50, 50, 50, 50, 
                                            cv2.BORDER_CONSTANT, 
                                            None, 
                                            value=255)
        _, th = cv2.threshold(image_bordered, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.erode(th, np.ones((75, 75), np.uint8))
        th = cv2.dilate(th, np.ones((75, 75), np.uint8))
       
        # show the original image and the edge detected image

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        contours, _  = cv2.findContours(th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse = True)
        
        dots = np.zeros((4,2), dtype=np.float32)
        s = contours[1].sum(axis=2)
        
        dots[0] = contours[1][np.argmin(s)]
        dots[2] = contours[1][np.argmax(s)]
        
        diff = np.diff(contours[1], axis = 2)
        dots[1] = contours[1][np.argmin(diff)]
        dots[3] = contours[1][np.argmax(diff)]
        
        # peri = cv2.arcLength(contours[1], True)
        # approx = cv2.approxPolyDP(contours[1], 0.02 * peri, True)
        
        warped = self.four_point_transform(image_bordered, dots)# * ratio)
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = cv2.resize(warped, 
                            (640,403), 
                            interpolation=cv2.INTER_AREA)
        return warped    

