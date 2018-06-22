'''date: May/10/2018
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

    while cap.isOpened(): # True: cap.isOpened()

        ret, bgr_image = cap.read()

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #print("bgr_image:", bgr_image)
        #faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
    	#		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        faces = detect_faces(face_cascade, gray_image)

        #DataMan_inst=DataManager('fer2013')
        #face1, emotio=DataMan_inst._load_fer2013()
        #print("Emotion: ", emotio)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)

            #emotion_loss = emotion_classifier.evaluate(gray_face)
            #print("evaluated: ", emotion_loss)

            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)
            #print("Emotion text: ", emotion_text)
            #print("emotion_probability: ", emotion_probability)
            #print("emotion_label_arg: ", emotion_label_arg)
 

            try:
                emotion_mode = mode(emotion_window)
                #print("emotion mode: ", emotion_mode)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0)) #red
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1)
            elif emotion_text == 'disgust':
                color = emotion_probability * np.asarray((255, 0, 255)) #magenta
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1)
            elif emotion_text == 'fear':
                color = emotion_probability * np.asarray((0, 0, 0)) #black
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1)
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))#yellow
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1)
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255)) #blue
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1)
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))#cyna
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1)
            elif emotion_text == 'neutral':
                color = emotion_probability * np.asarray((0, 255, 0)) #green
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()
                draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, emotion_mode,
                #          color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_text,
                         color, 0, -45, 1, 1) 
            
            
            #print("face_coordinates: ", face_cordinates)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Emotions_window', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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

    cap = cv2.VideoCapture(0) # Webcam source
    try:
        emotion_detector()

    except: pass    
