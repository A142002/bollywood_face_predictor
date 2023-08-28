#import os
#import pickle # to convert binary module
#fetch names of actors
#actors = os.listdir('data')


#filenames = []

#for actor in actors:
 #   for file in os.listdir(os.path.join('data',actor)): # enter into each actors folder
 #       filenames.append(os.path.join('data',actor,file)) # actors each file add into filename
#pickle.dump(filenames,open('filenames.pkl','wb')) #convert into binary this pkl file contain path of all images


# feature eximport pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
filenames = pickle.load(open('filenames.pkl','rb'))
 # model building
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
print(model.summary())

# function to extract feature
def feature_extractor(img_path,model):
    #load image
    img=image.load_img(img_path,target_size=(224,224))
    #convert image into numpy array
    img_array=image.img_to_array(img)
    # covert into batch of image
    expanded_img=np.expand_dims(img_array,axis=0)
    #preprocess img
    preprocessed_img=preprocess_input(expanded_img)
    #predict
    result=model.predict(preprocessed_img).flatten() #we use flatten to get features in 1d array
    return  result  #result contain features

#from each img we get 2048 size feature vectore we append that to feature list
features=[]
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))


#create another file to all extracted features of all images
pickle.dump(features,open('embedding.pkl','wb'))









