'''date: June/15/2018
    Author: Belayneh Mathewos , belaynehm3@gmail.com
    Place: @iCog Labs, Ethiopia
    License: open to modify 

'''

import asyncio

import grpc

from grpclib.server import Server
#import generated classes
from facial_emotions_pb2 import emoReply
from facial_emotions_pb2 import Emotion
from facial_emotions_pb2 import Rects
from facial_emotions_grpc import Emotion_recognizerBase
#from facial_emotions1_pb2_grpc import 
#---------------------------------
from concurrent import futures
import time
import sys
import cv2
import dlib
from keras.models import load_model
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

#import necessary file
from Facial_Emotions1.utils.dataset import get_labels
#from Facial_Emotions1.utils.infer import draw_text
#from Facial_Emotions1.utils.infer import draw_bounding_box
#from Facial_Emotions1.utils.infer import apply_offsets
#from Facial_Emotions1.utils.infer import detect_faces
#from Facial_Emotions1.utils.infer import get_colors

#from Facial_Emotions1.facial_emotion_stored1 import img_processor
from Facial_Emotions1.facial_emotion_stored1 import preprocess_input

from imutils import face_utils
import imutils
import numpy as np


#from Facial_Emotions1.utils import infer
#from Facial_Emotions1.utils import dataset

from statistics import mode

#---------------------------------

class Emotion_recognizer(Emotion_recognizerBase):
    #def __init__(self, count):
        #self.count = count
    def __init__(self):
        self.count = 0    
    
    # STREAM
    async def image_processor(self, stream):
    #async def emotion_classifier(self, stream):
        async for request in stream:
            img_data = request.image_data
            await stream.send_message(Rects(rects=img_data))
        #emo_label = ''
        #await stream.send_message(Emotion(emotion=emo_label))
   
    # STREAM
    async def emotion_classifier(self, stream):
        async for request in stream:
            emo_label = request.emotion_labels
            await stream.send_message(Emotion(emotion=emo_label))
        #emo_label = ''
        #await stream.send_message(Emotion(emotion=emo_label))

                    
    # STREAM
    async def StreamStreamEmotion(self, stream):
        # Exception handling.
        try:
            #
            #fa = FaceAligner(predictor, desiredFaceWidth=256)

            # load the input image, resize it, and convert it to grayscale
            img_path = './dataset/76.jpg'
            bgr_image = cv2.imread(img_path)
            
            bgr_image = imutils.resize(bgr_image, width=800)
            
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            #faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
            #       minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            #faces = detect_faces(face_cascade, gray_image)
            #faces = detector(gray_image, 0)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5);

            num_of_faces = len(faces)
            print("num_of_faces: ", num_of_faces)
            
            #for face_coordinates in faces:
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
                
                #print('emotion_probability:', emotion_probability)
                try:
                    emotion_mode = mode(emotion_window)
                    
                except:
                    continue

            
                async for request in stream:
                    label = request.emolabel
                    
                    #await stream.send_message(emoReply(number_of_faces=num_of_faces))
                        
                    if emotion_text == label:
                        print("emotion_text: ", emotion_text)

                        #await stream.send_message(emoReply(number_of_faces=num_of_faces))
                        await stream.send_message(emoReply(emotion=emotion_text))
                        #await stream.send_message(emoReply(number_of_faces=num_of_faces))
                        
            


        # Catch any raised errors by grpc.
        except grpc.RpcError as e:
            print("Error raised: " + e.details())



def mainCode():
    count = 0
    loop = asyncio.get_event_loop()
    #server = Server([Emotion_recognizer(count)], loop=loop)
    server = Server([Emotion_recognizer()], loop=loop)

    host, port = '127.0.0.1', 50051
    loop.run_until_complete(server.start(host, port))
    print('Serving on {}:{}'.format(host, port))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()


if __name__ == '__main__':
    #
    #PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    #predictor = dlib.shape_predictor(PREDICTOR_PATH)
    #detector = dlib.get_frontal_face_detector()
    
    #parameters for loading data and images
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
    print("emotion_target_size: ", emotion_target_size)

    # starting lists for calculating modes
    emotion_window = []
                
    mainCode()
