//

1. Normally emotion recognition
	-emotion recognition from webcam (usage: python facial_emotions1.py):
	-emotion recognition from stored image (usage: python facial_emotionsStored.py)
(README: This is for facial landmarks, face detection, facial emotion recognition using dlib, opencv, keras and cnn.
)

2. grpc service emotion recognition: (usage: python server.py, in other terminal python client.py/imageClient.py):
(README: This grpc service is implemented only for emotions that is recognised from processing input image at the server side then it will be transfered and displayed in client side. But, for the time being the image is not transferring	through the created grpc service.

)

3. emotion recognizer evaluator (usage: python Emotion_evaluator.py):
(README: The recognizer evaluation is done on the cohnkanade dataset. Its emotion recognition accuracy is approaximatelly 84 percent accurate.

)


