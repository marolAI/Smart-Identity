#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:44:46 2018

@author: maro
"""


import cv2
import numpy as np
import imutils
from skimage import exposure

im_width = 500
im_height = 324

def compute_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.medianBlur(gray,3)
    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    edged = cv2.Canny(dilation, 20, 200)
    return edged

def sharpen_edge(image):
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
    sharpened = cv2.filter2D(image, -1, kernel_sharpen)
    return sharpened

def get_angle(image):
    edged = compute_edges(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    angle = cv2.minAreaRect(c)[-1]
    rect = cv2.minAreaRect(c)
    p=np.array(rect[1])
    #print(p[0], p[1])
    if p[0] < p[1]:
        act_angle=rect[-1]+180
    else:
        act_angle=rect[-1]+90

    if act_angle < 90:
        angle = 90 - act_angle
    else:
        angle=act_angle- 90
    return angle

def rotate_image(mat, angle):
  # angle in degrees
  height, width = mat.shape[:2]
  image_center = (width/2, height/2)
  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])
  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)
  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]
  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

def find_contour(image):
    # load the query image, compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)
    edged = compute_edges(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    pts = None
    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c,False)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # now that we have our screen contour, we need to determine the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right, and bottom-left order
    try:
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point has the smallest sum whereas the bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points -- the top-right will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # multiply the rectangle by the original ratio
        rect *= ratio
        # now that we have our rectangle of points, let's compute the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # take the maximum of the width and height values to reach our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        # construct our destination points which will be used to map the screen to a top-down, "birds eye" view
        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
        # calculate the perspective transform matrix and warp the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
        # convert the warped image to grayscale and then adjust the intensity of the pixels to have minimum and maximum
        # values of 0 and 255, respectively
        warp = exposure.rescale_intensity(warp, out_range=(0, 255))
        warp = cv2.resize(warp, (im_width, im_height), interpolation=cv2.INTER_NEAREST)
        return warp
    except AttributeError as e:
        print(e, ".Please check your background to solve the problem!")


#image = cv2.imread("../../../uploads/IMG_21.jpg")
##image = cv2.resize(image, (341, 512))
#image = imutils.resize(image, height=500)
#image = sharpen_edge(image)
#rotated = rotate_image(image, get_angle(image))
#screen = find_contour(rotated)
#cv2.imshow("Original", image)
#cv2.imshow("Rotated", rotated)
#cv2.imshow("Screen", screen)
#cv2.waitKey(0)
#cv2.destroyAllWindows()