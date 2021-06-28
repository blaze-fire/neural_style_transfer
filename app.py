import streamlit as st
import streamlit.components.v1 as stc
from os import walk, path
import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    
    im1 = img.save("temp.jpg")
    img = tf.io.read_file("./temp.jpg")
    
    return img

def preprocess_image(img):
    img = tf.image.decode_image(img, channels=3)        # making sure image has 3 channels
    img = tf.image.convert_image_dtype(img, tf.float32) # making sure image has dtype float 32
    img = img[tf.newaxis, :]
    return img




def main():
    st.title("Neural Style Transfer")
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

     

    st.subheader("Upload Images")

    st.write("** Select two images, first being the source image and second one being the style image. **")

    images = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
    
    if st.button("Process"):
        if images is not None:
            # To See Details
            # st.write(type(image_file))
            # st.write(dir(image_file))
            
            #file_details = {"Filename": content_image.name, "FileType": content_image.type, "FileSize": content_image.size}
            
            #st.write(file_details)
            
            #st.image(images[0])
            content_image = load_image(images[0])
            style_image = load_image(images[1])
            content_image = preprocess_image(content_image)
            style_image = preprocess_image(style_image)
            st.write(content_image.dtype)   
            #st.image(np.squeeze(content_image))
            
            #st.image(content_image)
            #st.image(style_image)
            
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            st.image((np.squeeze(stylized_image)))
            
            

if __name__ == '__main__':
    main()
