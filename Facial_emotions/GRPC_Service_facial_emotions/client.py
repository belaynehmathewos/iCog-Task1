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

from imutils import face_utils

#-------------------------------
import random
import time
from concurrent import futures
import sys
import cv2
import numpy as np
import imutils
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
            print('RECEIVING EMOTION STREAMS')    
            emotion_labels = [emoRequest(emolabel='angry'), emoRequest(emolabel='disgust'), emoRequest(emolabel='fear'), emoRequest(emolabel='happy'), emoRequest(emolabel='sad'), emoRequest(emolabel='surprise'), emoRequest(emolabel='neutral')]
            #msgs = 
            repl = await stub.StreamStreamEmotion(emotion_labels)
            repl = list(repl)[0]
            print(repl.emotion)
            #num_of_faces = repl.number_of_faces
            #num_of_faces = int(num_of_faces)
            #print(num_of_faces)
            #for fa in range(num_of_faces):
            #print(await stub.StreamStreamGreeting(msgs))
            #
            async with stub.StreamStreamEmotion.open() as stream:
                for label in emotion_labels:
                    await stream.send_message(label)
                await stream.end()
                reply = await stream.recv_message()
                #print(reply)
                print(reply.emotion)

                #draw a bounding box and emotion text here by using reply.emotion or repl.emotion
                # ... but, it needs an image which is processed in server side should be shown here using cv2 !
                
                #draw_bounding_box(face_coordinates, rgb_image, color)
                #draw_text(face_coordinates, rgb_image, reply.emotion,
                 #        color, 0, -45, 1, 1)
                


        
    # Catch any raised errors by grpc.
    except grpc.RpcError as e:
        print("Error raised: " + e.details())


if __name__ == '__main__':
    #
    looper = asyncio.get_event_loop()
    looper.run_until_complete(mainCode())

    #asyncio.get_event_loop().run_until_complete(mainCode())
    