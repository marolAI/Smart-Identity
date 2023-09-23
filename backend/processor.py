import cv2
import numpy as np
import urllib.request
import streamlit as st

from backend.classifier import classify
from backend.utils import sharpen_edge


def add_space(num_spaces=1):
    for _ in range(num_spaces):
        st.write("\n")


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def process():
    image_with_info = []
    with st.sidebar:
        st.subheader("About")
        st.markdown(
            """This is a simple web app that classifies an image as either an ECOWAS ID Card or not."""
        )
        
        add_space()
        
        st.subheader("Classify New Image")
        st.markdown(
            """In order to classify a new image upload an image or provide an image URL."""
        )
        selection = st.radio("Choose input method", ["url", "path"])
        
        add_space()
        
        if selection == "url":
            url = st.text_input("Add an image URL here")
            process_button = st.button("Process")
            
            add_space(3)
            
            st.markdown("##### *Need an image to test?*")
            st.code("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2aNMlz30mQn0D843poj582LXZdiuJ8FZzqW45Y0csmw&s", language="python")
            
            if process_button:
                image = url_to_image(url)
                image = sharpen_edge(image)
                image_with_info.append((image, url))
                
        elif selection == "path":
            path = st.file_uploader("Import an image file", type=["jpg", "png", "jpeg"])
            if path:
                image = cv2.imdecode(np.fromstring(path.read(), np.uint8), 1)
                image = sharpen_edge(image)
                image_with_info.append((image, path))
                
        else:
            st.error(f"{selection} not valid.")
    
    col1, _, col2 = st.columns([2, 0.5, 2])
    
    with col1:
        imageLocation = st.empty()
        imageLocation.markdown(
            '<img src="./app/static/smartid_icon.png" height="333" style="border: 1px solid black">',
            unsafe_allow_html=True,
        )
        
    with col2:
        st.markdown("#### Classification Result:")
        if image_with_info:
            try:
                res = classify(image_with_info[0][0], model="backend/id_not_id.model")
                
                if res == "ECOWAS ID Card":
                    st.success("The image is classified as **{}**.".format(res))
                    imageLocation.markdown(
                        '<img src="./app/static/id_icon.png" height="333" style="border: 1px solid black">',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("The image is classified as **{}**.".format(res))
                    imageLocation.markdown(
                        '<img src="./app/static/notid_icon.png" height="333" style="border: 1px solid black">',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error("Error: {}".format(e))
            
            st.image(image_with_info[0][1], use_column_width=True)
    st.write("\n")
    st.markdown("*** *It is important to note that the outcome of the model's classification can still be enhanced.* ***")
    
    
    
    
    
