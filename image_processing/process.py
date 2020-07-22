#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 04:41:27 2018

@author: maro
"""

from __future__ import print_function

import os
import cv2
import numpy as np
import imutils
from PIL import Image
import pytesseract
import urllib.request
from image_preprocessing.imtools import  enhance, get_imlist, alignImages, sharpen_edge
from image_preprocessing.preprocess import find_contour, rotate_image, get_angle
from image_classification.idClassification import id_not_id

kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                           [-1,2,2,2,-1],
                           [-1,2,8,2,-1],
                           [-1,2,2,2,-1],
                           [-1,-1,-1,-1,-1]]) / 8.0

kernel = np.ones((5,5),np.uint8)

def process_image(url=None,path=None):
    """
    process the image by predicting first if it is an ID or Not,
    then if so detect and extract text before parsing them.
    """
    if url != None:
        image = url_to_image(url)
    elif path != None:
        image = cv2.imread(path)
    else:
        return "Wrong Wrong Wrong, What are you doing ??? "

    while True:
        # preprocessing the input image
        image = sharpen_edge(image)
        rotated = rotate_image(image, get_angle(image))
        warpped = find_contour(rotated)
        # Read reference image
#        cv2.imshow("warp", warpped)
#        cv2.waitKey(0)
        cv2.imwrite("data/demo/warpped.jpg", warpped)
#        refFilename = "data/tmp/tmp.jpg"
#        print("[INFO] Reading reference image : ", refFilename)
#        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
#        print("[INFO] Aligning images ...")
#        # Registered image will be resotred in imReg.
#        # The estimated homography will be stored in h.
#        if warpped.all() == None:
#            print("Error in matching the images.\
#                  Possible reason: 'The input image is not taken in the dark background.'")
#            return "Error in matching the images.\
#                  Possible reason: 'The input image is not taken in the dark background.'"
#            break
#        else:
#            imReg, h = alignImages(warpped, imReference)
#            # Write aligned image to disk.
#            outFilename = "data/demo/aligned.jpg"
#            print("[INFO] Saving aligned image : ", outFilename);
#            cv2.imwrite(outFilename, imReg)
#            # Print estimated homography
#            #print("Estimated homography : \n",  h)


        # use the function 'get_imlist'  to call the images' path
        imlist = get_imlist('data/demo')
        for im in imlist:
            print(('[INFO] Detecting either {:s} is an ID or not...'.format(im)))
            img = cv2.imread(im)
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 129, 7, 21)
#            cv2.imshow("Im", img)
#            cv2.waitKey(0)
            label, proba = id_not_id(img, model="image_classification/id_not_id.model")
            if label == 'Not ID':
                #os.remove(im)
                #print('This is not an ID, please enter a new one!')
                return "This is not an ID, please enter a new one!"
                break
            else:
                print('This is an ID')
                print(('[INFO] Processing {:s}...'.format(im)))
                # retrieving the filename with extension in the input  images
                filename_w_ext = os.path.basename(im)
                # splitting the filename and the extension
                filename, file_extension = os.path.splitext(filename_w_ext)
                file_path = 'text-detection-ctpn/model/'
                img_path = 'data/demo'
                txt_file = 'res_temp.txt'
                img_file = filename_w_ext
                txt_full_path = os.path.join(file_path, txt_file)
                img_full_path = os.path.join(img_path, img_file)

                print('[INFO] Loading {:s}...'.format(img_full_path))
                # read the image
                image = cv2.imread(img_full_path)
                image = cv2.fastNlMeansDenoisingColored(image, None, 3, 129, 7, 21)
                image = cv2.filter2D(image, -1, kernel_sharpen)
#                image = 255 * (image > 150).astype(np.uint8)
#                cv2.imshow("Img", image)
#                cv2.waitKey(0)
                #os.remove(im)
                print('[INFO] Loading {:s}...'.format(txt_full_path))
                # read txt file
                data = np.loadtxt(txt_full_path, delimiter=',')

                strs = []
#                strs_2 = ["numero CIN:", "Nom:", "Prenom:", "Date de naissance:", "Sexe:",
    #                      "Date d'expiration:","Adresse du domicile:"]

                for i in range(len(data)):
#                    if i % 2 == 0:
                    roi = image[int(data[i][1]):int(data[i][3]), int(data[i][0]):int(data[i][2])]
                    roi = cv2.resize(roi, None, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.fastNlMeansDenoising(roi_gray, None, 3, 7, 21)
                    roi_gray = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)
                    roi_gray = cv2.filter2D(roi_gray, -1, kernel_sharpen)
                    roi_gray = 255 * (roi_gray> 128).astype(np.uint8)
#                    denoised = enhance(im)
#                    cv2.imshow("den1",  roi_gray)
#                    cv2.waitKey(0)
                    filename = "{}.jpg".format(os.getpid())
                    cv2.imwrite(filename, roi_gray)
                    config = ("-l eng --oem 1 --psm 7")
                    text = pytesseract.image_to_string(Image.open(filename),  config=config)
                    os.remove(filename)
                    strs.append(text)
            return strs

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

##a = process_image("https://www.pyimagesearch.com/wp-content/uploads/2017/06/tesseract_header.jpg")
#a = process_image(path="data/uploads/IMG_31.jpg")
#print(a)