# load new image -->face detection -->and extract its features
# find the cosine distance of current image with all the 8655 features
# recommend that img
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list=np.array(pickle.load(open("embedding.pkl",'rb')))
filenames=pickle.load(open("filenames.pkl",'rb'))
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector = MTCNN()

#load img and face detection
sample_img=cv2.imread('sample/paa.jpg')
# extract face from img
results=detector.detect_faces(sample_img)
x,y,width,height=results[0]['box']
face = sample_img[y:y+height,x:x+width]


#extract features from inserted img
image=Image.fromarray(face)
image=image.resize((224,224))
face_array=np.asarray(image)
face_array=face_array.astype('float32')
expanded_img=np.expand_dims(face_array,axis=0)
preprocessed_img=preprocess_input(expanded_img)
result=model.predict(preprocessed_img).flatten()


# result is vector of that image that we test so we calculate cosine distence of this vector to all vectors of all images in dataset to get shortest distence vector means most matching
similarity=[] # it stores similarity distence of input img to all images in dataset
for i in range(len(feature_list)): #run loop 8654 times
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1)))

#it contain tuple with 2 items index and similarity score
#sort image based on similarity score
#and return index of that img by getting index
#we get index in 0th tuple in 0th index
index_pos=sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]#it hold index

temp_img=cv2.imread(filenames[index_pos])
cv2.imshow("output",temp_img)
cv2.waitKey(0)

#recommend img


