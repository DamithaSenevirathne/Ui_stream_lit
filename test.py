import streamlit as st
import numpy as np
import time
import os

def _max_width_():
    max_width_str = st.markdown(
        """
    <style>
    .reportview-container .main .block-container{{
        {}
    }}
    </style>    
    """.format("max-width: 2000px;"),
        unsafe_allow_html=True,
    )

_max_width_()

def imageUpdate(latest_image1,K,predicted):

    latest_image1.image([K[0],K[1],K[2],K[3],'images/normal6.jpg'],width=180)        
    time.sleep(2)




page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

    #st.write(df)
if page == "Exploration":
    st.title("Data Exploration")
    #x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=3)
    #y_axis = st.selectbox("Choose a variable for the y-axis", df.columns, index=4)
    #visualize_data(df, x_axis, y_axis)



if page == "Homepage":

    filename = st.text_input('Enter ROS bag file path:')

    st.write('image_prdictions')

    latest_image1 = st.empty()

    st.write('image 1     ,image 2         ,image 3           ,predicted         ,actual, Error 0.23454')

    latest_it = st.empty()

    K = ['normal1.jpg','normal2.jpg','normal3.jpg','normal4.jpg','normal5.jpg','normal6.jpg','normal4.jpg','normal5.jpg','normal6.jpg']

    #imageUpdate(latest_image1,K,'test2.png')

    for i in range(len(K)):

        K[i] = 'images/' + K[i]

    st.write('IMU_prdictions')
    st.write('Auto Encoder Error')
    st.write('prediction : ')

    imageUpdate(latest_image1,K,'images/normal6.jpg')
    imageUpdate(latest_image1,K[1:],'images/normal6.jpg')
    imageUpdate(latest_image1,K[2:],'images/normal6.jpg')
    imageUpdate(latest_image1,K[3:],'images/normal6.jpg')

    


