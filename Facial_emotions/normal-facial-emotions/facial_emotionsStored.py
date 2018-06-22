'''date: May/15/2018
    Author: Belayneh Mathewos , belaynehm3@gmail.com
    Place: @iCog Labs, Ethiopia
    License: open to modify but, keep it

'''
import cv2
import numpy as np
#import tensorflow as tf
#from PIL import Image
from keras.models import load_model
#from keras import backend as K
from utils.dataset import get_labels

from utils.infer import detect_faces
from utils.infer import draw_text
from utils.infer import draw_bounding_box
from utils.infer import apply_offsets
from utils.infer import load_detection_model
from utils.infer import get_colors
from statistics import mode

import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import cv2
import dlib
import itertools
#from sklearn.svm import SVC
import sys, os, subprocess


def preprocess_input(x_face, v2=True):
    x_face = x_face.astype('float32')
    x_face = x_face / 255.0
    if v2:
        x_face = x_face - 0.5
        x_face = x_face * 2.0
    return x_face

def emotion_detector():

# load the input image, resize it, and convert it to grayscale
    try:
        #img_path = './dataset/surprise/2.jpg'
        img_path = './dataset/76.jpg'
        bgr_image = cv2.imread(img_path)
            
        bgr_image = imutils.resize(bgr_image, width=800)
        
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        
        #faces = detector(gray_image, 0)
        #let's detect multiscale (some images may be closer to camera than others) images
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5);

        #print the number of faces found
        #print('you1')
        #print('Faces found: ', len(faces))

        
        #for rect in faces:
        for (x, y, w, h) in faces:
                        
                #(x1, x2, y1, y2) = apply_offsets(face_coordinates, emotion_offsets)
                #(x, y, w, h) = rect_to_bb(face_coordinates)
                
                #gray_face = gray_image[y1:y2, x1:x2]
                gray_face = gray_image[y:y+h, x:x+w]
                x_off, y_off = emotion_offsets
                #gray_face = gray_image[x-x_off:x+x_off+w, y-y_off:y+y_off+h]
                #gray_face = gray_image[w:h, x:y]

                try:
                    #gray_face = cv2.resize(gray_face, (emotion_target_size), interpolation = cv2.INTER_AREA)
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                    
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                
                emotion_prediction = emotion_classifier.predict(gray_face)

                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)
                #print("emotion_text11: ", emotion_text)
                
                #print("face_coordinates: ", face_coordinates)
                try:
                    emotion_mode = mode(emotion_window)
                    
                except:
                    continue

                
                #print('emotion: ', emotion_text)
                if emotion_text == 'angry':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((255, 0, 0)) #red
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()

                    #draw_bounding_box(rect, rgb_image, color)
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                    
                elif emotion_text == 'disgust':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((255, 0 , 255)) #red
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()

                    #draw_bounding_box(rect, rgb_image, color)
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (255, 0, 255), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
                    
                elif emotion_text == 'fear':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((255, 255, 0)) #green
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
                    
                elif emotion_text == 'happy':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((255, 255, 0))#green
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
                    
                elif emotion_text == 'sad':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((0, 0, 255)) #blue
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                    
                elif emotion_text == 'surprise':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((0, 255, 255))#cyna
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
                    
                elif emotion_text == 'neutral':
                    print('emotion: ', emotion_text)
                    color = emotion_probability * np.asarray((0, 255, 0)) #green
                    #color = get_colors(emotion_probability)
                    color = color.astype(int)
                    color = color.tolist()
                    cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                    #draw_text(rect, rgb_image, reply.emotion,
                    #         color, 0, -45, 1, 1)
                    cv2.putText(bgr_image, emotion_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                     
                
            
        #print("face_coordinates: ", face_cordinates)
        #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Emotions_window', bgr_image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        cv2.waitKey(0)

    except:
        pass

    #cap.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    # parameters for loading data and images
    #emotion_model_path = './models/emotion_recognition_models/emotion_model.hdf5'
    emotion_model_path = './models/emotion_recognition_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('yes')

    # hyper-parameters for bounding boxes shape
    frame_window = 14
    emotion_offsets = (28, 56)

    # loading models
    face_cascade = cv2.CascadeClassifier('./models/face_detection_models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []
    emotion_probability = 0.998

    #cap = cv2.VideoCapture(0) # Webcam source
    try:
        emotion_detector()

    except: pass    
