#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 04:18:26 2018

@author: maro
"""

# load the necessary packages
import os
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def get_imlist(path):
    """
    Return a list of filenames for
    all jpg, jpeg, png images in directory
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') \
            or f.endswith('.jpeg') or f.endswith('.png')]

def sharpen_edge(image):
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
    sharpened = cv2.filter2D(image, -1, kernel_sharpen)
    return sharpened

def enhance(image, alpha = 2.25, beta = -0.0):
    """
    remove noise from the input image
    """
    img = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 121, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 3, 7, 21)
    enhanced_im  = np.array(denoised, np.uint8)
    thresh_1, bw_im = cv2.threshold(enhanced_im, 128, 255, cv2.THRESH_BINARY)
    img_bin = 255 * (bw_im > 128).astype(np.uint8)
    new_im = alpha * img_bin + beta
    new_im = np.clip(new_im, 0, 255).astype(np.uint8)
    new_im = cv2.fastNlMeansDenoising(new_im, None, 31, 7, 21)
    return new_im

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    #matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg, h



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def variance_of_laplacian(image):
	"""
    compute the Laplacian of the image and then return the focus
	measure, which is simply the variance of the Laplacian
    """
	return cv2.Laplacian(image, cv2.CV_64F).var()