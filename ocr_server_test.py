#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 03:51:22 2018

@author: maro
"""

import os
#import logging
#import json
from logging import Formatter, FileHandler
from flask import Flask, request, session, render_template, redirect, make_response, url_for
from werkzeug.utils import secure_filename
from image_processing.process import process_image

import sys

sys.setrecursionlimit(10000) # 10000 is an example, try with different values

basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

_VERSION = 1  # API version

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def index():
#	return redirect(url_for('upload_file'))
    return render_template('about.html')

result = []

@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path = "data/uploads/{}".format(filename)
            print("file was uploaded in {}".format(path))
            rec_string = process_image(path=path)
            for str in rec_string:
                result.append(str)
            return redirect(url_for('results'))
        

print(result)
@app.route('/result')
def results():
    return render_template('result.html', result=result)



if __name__ == '__main__':
	#app.debug = True
	app.run(host="0.0.0.0",port = int(8080))
