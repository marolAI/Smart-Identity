import cv2
import numpy as np
import urllib.request
import streamlit as st

from backend.classifier import classify
from backend.utils import sharpen_edge


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def process():
    st.subheader("Classify New Image")
    selection = st.radio("Either use url or drop an image", ["url", "path"])
    if selection == "url":
        st.markdown('*Need an image to test?*')
        st.code("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2aNMlz30mQn0D843poj582LXZdiuJ8FZzqW45Y0csmw&s", language="python")
        url = st.text_input("Link to an image")
        process_button = st.button("Process")
        
        if process_button:
            image = url_to_image(url)
            
            image = sharpen_edge(image)
            
            res = classify(image, model="backend/models/id_not_id.model")
            c1, _, c2 = st.columns([2, 0.5, 2])
            c1.image(url, use_column_width=True)
            c2.subheader("Classification Result")
            c2.write("The image is classified as **{}**.".format(res))
    elif selection == "path":
        path = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        
        if path:
            image = cv2.imdecode(np.fromstring(path.read(), np.uint8), 1)
            
            image = sharpen_edge(image)
            
            res = classify(image, model="backend/models/id_not_id.model")
            c1, _, c2 = st.columns([2, 0.5, 2])
            c1.image(path, use_column_width=True)
            c2.subheader("Classification Result")
            c2.write("The image is classified as **{}**.".format(res))
    else:
        st.error(f"{selection} not valid.")
    
    
    
    
    