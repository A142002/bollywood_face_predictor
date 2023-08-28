import os
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import  Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
#insert img into upload folder
feature_list=pickle.load(open("embedding.pkl",'rb'))
filenames=pickle.load(open("filenames.pkl",'rb'))
def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('uploades',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img=cv2.imread(img_path)
    results=detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    # extract features from inserted img
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list,features):
    similarity = []  # it stores similarity distence of input img to all images in dataset
    for i in range(len(feature_list)):  # run loop 8654 times
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1)))

    # it contain tuple with 2 items index and similarity score
    # sort image based on similarity score
    # and return index of that img by getting index
    # we get index in 0th tuple in 0th index
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]  # it hold index
    return index_pos

st.title("Which Bollywood celebrity are you?")
uploaded_img=st.file_uploader("choose an image")

if uploaded_img is not None:
    #save the image in directory
    if save_uploaded_image(uploaded_img):
        #load the image and display

        display_img=Image.open(uploaded_img)
        #extract features of current image
        features=extract_features(os.path.join('uploades',uploaded_img.name),model,detector)
        
        #recommend
        index_pos=recommend(feature_list,features)
        #display
        col1,col2=st.columns(2)
        with col1:
            st.header("Your uploaded image")
            st.image(display_img,width=300)
        with col2:
            st.header(" ".join(filenames[index_pos].split('\\')[1].split('_')))
            st.image(filenames[index_pos],width=150)


