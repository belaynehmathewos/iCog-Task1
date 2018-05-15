#!/usr/bin/python
# USAGE
#Correctly to run this code, the dlib's shape_predictor_68_face_landmarks.dat should be placed in the same folder,
#or a slight change should be needed in 'PREDICTOR_PATH'.
#...
#python facial_landmarks_with_annotation.py 

#The code is tested on ubuntu machine

#For Raspberry Pi it will be slightly modified ...

# import the necessary packages
import dlib
import cv2
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import imutils
import time
  
# initialize dlib's face detector and then create
# the facial landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])

# loop over the frames from the video stream
def get_and_anotate_landmarks():

		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-covordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
				#pass
		    
		cv2.putText(frame,'Wow', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

		# show the frame
		cv2.imshow("Face_with_landmarks", frame)
	

if __name__ == '__main__':
	print("[INFO] loading predictor...")
    	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] camera warming up...")
	vs = VideoStream(src=0).start()
	#Try by using vs = captureFromCam(0)
	time.sleep(2.0)

	while True:	
		get_and_anotate_landmarks()

		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	
	cv2.destroyAllWindows()
	vs.stop() 
