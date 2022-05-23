import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
import warnings
import pandas as pd
import plotly.express as px
warnings.filterwarnings(action='ignore')
model=joblib.load('tumor_detection')

with st.sidebar:
  selected=option_menu(
      menu_title='Main_Menu',
      options=['Home','Upload']
  )
if selected=='Home':
  st.title('Tumor Detector')
  st.subheader("This is an app which detects Brain Tumor")
  st.image(r"/content/drive/MyDrive/ML/homepage.jpg",width=800)
flag=0
classification=0
if selected=='Upload':
  st.title("YOU CAN UPLOAD THE IMAGE HERE")
  st.image(r"/content/drive/MyDrive/ML/upload1.webp")
  image_file = st.file_uploader("Upload Images", type=["jpg"])
  flag=1
  try:
    img = Image.open(image_file)
    st.image(img)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    res = model.predict(x)
    confidence=res[0][classification]*100
    classification = np.where(res == np.amax(res))[1][0]
    data={'Tumor':['Not_present','present'],'confidence':[0,0]}
    df=pd.DataFrame(data)
    if classification==1:
      df.iloc[1,[1]] = confidence
      df.iloc[0,[1]] = 100-confidence
      bar_chart=px.bar()
      st.subheader('No tumor detected')
      st.write('The model prediction along with the confidence is plotted below')
      fig = px.bar(df, x='Tumor', y='confidence')
      st.plotly_chart(fig)
    else:
      df.iloc[1,[1]] = confidence
      df.iloc[0,[1]] = 100-confidence
      st.subheader('Tumor detected')
      st.write('The model prediction along with the confidence is plotted below')
      bar_chart=px.bar()
      fig = px.bar(df, x='Tumor', y='confidence')
      st.plotly_chart(fig)
  except:
    print("error")
 
  

# if selected=='Result' and flag==1:
#   res = model.predict(x)
#   classification = np.where(res == np.amax(res))[1][0]
#   if classification==1:
#     st.write('No tumor detected')
#   else:
#     st.write('Tumor detected')

