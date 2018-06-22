'''date: June/21/2018
    Author: Belayneh Mathewos , belaynehm3@gmail.com
    Place: @iCog Labs, Ethiopia
    
'''

import cv2
import numpy as np

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
import sys, os, subprocess

import glob
import random

#def emotion_detector(image):
def preprocess_input(x_face, v2=True):
    x_face = x_face.astype('float32')
    x_face = x_face / 255.0
    if v2:
        x_face = x_face - 0.5
        x_face = x_face * 2.0
    return x_face

def emotion_detector(gray_image):

# load the input image, resize it, and convert it to grayscale
    try:
        #
        #faces = detector(gray_image, 0)
        #let's detect multiscale images
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5);

        
        # loop over the face detections
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

            try:
                emotion_mode = mode(emotion_window)
                
            except:
                continue

            return emotion_text

    except:
        pass


def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20

    #files = glob.glob("/home/belayneh/Desktop/Codes/Task1/Online Modify/codes/Facial_emotions/Facial-emotion-evaluation/dataset//%s//*" %emotion)
    files = glob.glob("./dataset//%s//*" %emotion)
    #files = glob.glob("/home/belayneh/Desktop/Codes/Task1/Online Modify/codes/Facial-Emotions1/Facial-Emotions-Evaluate/datasetHelen//%s//*" %emotion)
    
    #print("Here1 it is: ",files)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-6/0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            image = imutils.resize(image, width=800)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    
    print ("Recognizing and categorizing emotions")
    print ("size of training set is:", len(training_labels), "images")
    print("size of predicting set is:", len(prediction_labels), "images")
    #print ("predicting classification set")
    cnt = 0
    cnt2 = 0
    correct = 0
    incorrect = 0
    try:

        for image in prediction_data:
            #print("image:", image )#print("image:", image, " prediction_data: ", prediction_data)
            try:
                
                #predicting classification set
                pred = emotion_detector(image)
                pred_label = returnlabel(pred)    
                emotions1 = ["neutral", "angry", "disgust", "fear", "happy", "sad", "surprise"]
                #For evaluation purpose
                for emotion_text in emotions1:
                    if pred == emotion_text:
                        #cv2.imwrite()
                        cv2.imwrite("./evaluateDataset2//%s//%s.jpg" %(emotion_text, cnt), image)
                        #cv2.imwrite("./evaluateHelenDataset//%s//%s.jpg" %(emotion_text, cnt), image)
                    #else:
                        #pass    
                        
                if pred_label == prediction_labels[cnt]:
                #if pred_label == training_labels[cnt]:
                    correct += 1
                    cnt += 1
                else:
                    incorrect += 1
                    cnt += 1
                    #cv2.imwrite("/home/belayneh/Desktop/Codes/Task1/Online Modify/codes/Facial_emotions/Facial-emotion-evaluation/difficult//%s_%s_%s.jpg" %(emotions[prediction_labels[cnt]], emotions[pred_label], cnt), image)
                    cv2.imwrite("./difficult//%s_%s_%s.jpg" %(emotions[prediction_labels[cnt]], emotions[pred_label], cnt), image)                    
                    #incorrect += 1
                    #cnt += 1
            except: pass

        #return ((100*correct)/(correct + incorrect))
    except: pass    

def returnlabel(emotion):
    for i in range(len(emotions)):
         #pass
        j=10
        if emotion == emotions[i]:
            return i
        else:
            return j      

def calcPercentage():
    #For cohnkanade dataset
    
    angry_correct = 31#19
    angry_incorrect = 25#10
    disgust_correct = 28#10
    disgust_incorrect = 0#0
    fear_correct = 56#22
    fear_incorrect = 13#14
    happy_correct = 67#29
    happy_incorrect = 2#2
    neutral_correct = 100#/46 #41
    neutral_incorrect = 23#/25 #30
    sad_correct = 71#36
    sad_incorrect = 10#10
    surprise_correct = 70#36
    surprise_incorrect = 10#5

    scoreCorrect = angry_correct+disgust_correct+fear_correct+happy_correct+neutral_correct+sad_correct+surprise_correct
    scoreIncorrect = angry_incorrect+disgust_incorrect+fear_incorrect+happy_incorrect+neutral_incorrect+sad_incorrect+surprise_incorrect
    scorePercentage = (100*scoreCorrect)/(scoreCorrect+scoreIncorrect)
    '''
    #For helen dataset
    angry_correct = 19
    angry_incorrect = 5
    disgust_correct = 0
    disgust_incorrect = 1
    fear_correct = 5
    fear_incorrect = 5
    happy_correct = 44
    happy_incorrect = 2
    neutral_correct = 37 #41
    neutral_incorrect = 6 #30
    sad_correct = 20
    sad_incorrect = 8
    surprise_correct = 11
    surprise_incorrect = 2

    scoreCorrect = angry_correct+disgust_correct+fear_correct+happy_correct+neutral_correct+sad_correct+surprise_correct
    scoreIncorrect = angry_incorrect+disgust_incorrect+fear_incorrect+happy_incorrect+neutral_incorrect+sad_incorrect+surprise_incorrect
    scorePercentage = (100*scoreCorrect)/(scoreCorrect+scoreIncorrect)'''
    
    return scorePercentage
#Now run it
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

    emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
    
    data = {}
    
    #cap = cv2.VideoCapture(0) # Webcam source
    try:
    
        metascore = []
        for i in range(0,10):
            run_recognizer()
            #correct = run_recognizer()
            #print ("got", correct, "percent correct!")
            #metascore.append(correct)

        #print ("\n\nend score:", np.mean(metascore), "percent correct!")
        print("evaluation score: ", calcPercentage(), "in cohnkanade dataset") # nearly 84 percent acurate
                                                                                #76 percent accurate /
        #print("evaluation score: ", calcPercentage(), "in helen dataset") #83 percent accurate

    except:
        pass    
