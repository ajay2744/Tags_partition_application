import cv2
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import numpy as np

st.title("welcome")

file=st.file_uploader("Upload tags image to be split:",type=["jpg","jpeg","png"])

if file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(file.read())
        temp_filepath = temp_file.name
        

    image = cv2.imread(temp_filepath)
    st.write("The uploaded input is:")
    st.image(temp_filepath)
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    adpt_thresh=cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)


    for i in range(4):
        contours,hierarchy=cv2.findContours(adpt_thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

# Exclude the largest contour
        filtered_contours = [c for c in contours if not np.array_equal(c, largest_contour)]


        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box coordinates
            cv2.rectangle(adpt_thresh, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw rectangle

    contours,hierarchy=cv2.findContours(adpt_thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

# Exclude the largest contour
    filtered_contours = [c for c in contours if not np.array_equal(c, largest_contour)]
    filtered_contours=sorted(filtered_contours,key=cv2.contourArea,reverse=True)[0:18]

    #plt.figure(figsize=(20,3))

    st.write("The output after tags split:")

    for ind,contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box coordinates
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #print(ind)
        #plt.subplot(18,1,ind+1)
        #print(image[y:y+h,x:x+w])
        plt.imsave(f"img{ind}.png",image[y:y+h,x:x+w])
        st.image(f"img{ind}.png")

#st.pyplot(plt.gcf())
    

    

