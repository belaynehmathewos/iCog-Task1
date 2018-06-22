import asyncio

import grpc

from grpclib.client import Channel
#import generated class
from facial_emotions_pb2 import emoRequest
from facial_emotions_pb2 import face_emotion
from facial_emotions_pb2 import image_frame
from facial_emotions_grpc import Emotion_recognizerStub
#import necessary packages
from Facial_Emotions1.utils.infer import draw_text
from Facial_Emotions1.utils.infer import draw_bounding_box
#from Facial_Emotions1.utils.infer import apply_offsets
#from Facial_Emotions1.utils.infer import detect_faces
from Facial_Emotions1.utils.infer import get_colors

#from Facial_Emotions1.facial_emotion_stored1 import img_processor
#from Facial_Emotions1.facial_emotion_stored1 import preprocess_input

import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

#-------------------------------
import random
import time
from concurrent import futures
import sys
import cv2
import numpy as np
import dlib
#import cv2

from statistics import mode
#-------------------------------

async def mainCode():
    channel = Channel(loop=asyncio.get_event_loop())
    stub = Emotion_recognizerStub(channel)

    # ---------------------------------------------------

    # Exception handling.
    try:
        #
        #for i in range(2):
        # initialize dlib's face detector and then create
        # the facial landmark predictor and the face aligner
        #fa = FaceAligner(predictor, desiredFaceWidth=256)

        # load the input image, resize it, and convert it to grayscale
        img_path = './dataset/76.jpg'
        bgr_image = cv2.imread(img_path)
            
        bgr_image = imutils.resize(bgr_image, width=800)
        
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        
        #faces = detector(gray_image, 0)
        #let's detect multiscale (some images may be closer to camera than others) images
        faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5);

        #print the number of faces found
        print('Faces found: ', len(faces))

        #faceOrig = []
        #faceAligned = []
        #print("rects: ", rects)
        # loop over the face detections
        #for rect in faces:
        for (x, y, w, h) in faces:
            # using facial 
            '''
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(rgb_image[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(rgb_image, gray_image, rect)

            import uuid
            
            f = str(uuid.uuid4())
            cv2.imwrite("./images/foo/" + f + ".jpg", faceAligned)'''



            # ------------------------------------------------------------------------
            # STREAMS
            '''
            print('EMOTION_CLASSIFIER')    
            emotion_labels = [face_emotion(emotion_labels='angry'), face_emotion(emotion_labels='disgust'), face_emotion(emotion_labels='fear'), face_emotion(emotion_labels='happy'), face_emotion(emotion_labels='sad'), face_emotion(emotion_labels='surprise'), face_emotion(emotion_labels='neutral')]
            #msgs = 
            repl = await stub.emotion_classifier(emotion_labels)
            repl = list(repl)[0]
            print(repl.emotion)
            #print(await stub.StreamStreamGreeting(msgs))
            #
            async with stub.emotion_classifier.open() as stream:
                for label in emotion_labels:
                    await stream.send_message(label)
                await stream.end()
                reply = await stream.recv_message()
                #print(reply)
                print(reply.emotion)'''

            #--------------------------------------------------------------------
            #STREAMS
            print('RECEIVING EMOTION STREAM')    
            emotion_labels = [emoRequest(emolabel='angry'), emoRequest(emolabel='disgust'), emoRequest(emolabel='fear'), emoRequest(emolabel='happy'), emoRequest(emolabel='sad'), emoRequest(emolabel='surprise'), emoRequest(emolabel='neutral')]
            #msgs = 
            repl = await stub.StreamStreamEmotion(emotion_labels)
            repl = list(repl)[0]
            print(repl.emotion)
            #print(await stub.StreamStreamGreeting(msgs))
            #
            async with stub.StreamStreamEmotion.open() as stream:
                for label in emotion_labels:
                    await stream.send_message(label)
                await stream.end()
                reply = await stream.recv_message()
                #print(reply)
            #    print(reply.emotion)

                #---------------------------------------------------------------------------------------------------
                #draw a bounding box and emotion text here by using reply.emotion or repl.emotion
                # ... but, it needs an image which is processed in server side should be shown here using cv2 !
                
                color = emotion_probability * np.asarray((255, 0, 0)) #red
                #color = get_colors(emotion_probability)
                color = color.astype(int)
                color = color.tolist()

                #draw_bounding_box(rect, rgb_image, color)
                cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
                #draw_text(rect, rgb_image, reply.emotion,
                #         color, 0, -45, 1, 1)
                cv2.putText(bgr_image, reply.emotion, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                #---------------------------------------------------------------------------------------------------

                
        # display the output images
        #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Emotions_window', bgr_image)
            
        #cv2.imshow("Original", faceOrig)
        #cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)

        
    # Catch any raised errors by grpc.
    except grpc.RpcError as e:
        print("Error raised: " + e.details())

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def startupServer():
    looper = asyncio.get_event_loop()
    looper.run_until_complete(mainCode())

    #asyncio.get_event_loop().run_until_complete(mainCode())
    

if __name__ == '__main__':
    #
    #
    #PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    #predictor = dlib.shape_predictor(PREDICTOR_PATH)
    #detector = dlib.get_frontal_face_detector()
    
    
    # hyper-parameters for bounding boxes shape
    frame_window = 14
    
    # loading models
#    face_cascade = cv2.CascadeClassifier('./models/face_detection_models/haarcascade_frontalface_default.xml')
    #load cascade classifier training file for haarcascade
    haar_face_cascade = cv2.CascadeClassifier('./models/face_detection_models/haarcascade_frontalface_default.xml')
    
    # getting input model shapes for inference
    
    
    emotion_probability = 0.998

    startupServer()
 
    