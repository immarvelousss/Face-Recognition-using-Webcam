# This file contains code originally created by Caroline Dunn (c) 2021.
# Licensed under the MIT License. See LICENSE file for details.

## -------------------------------- MODIFICATIONS ----------------------------- ##
## - used SMTP library to send out email instead of mailgun						##
## - Notification sent on unknown face detected									##
## - Email sent includes a captured picture and video 							##
## ---------------------------------------------------------------------------- ##

# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import requests
import os
import datetime
import smtplib
import imghdr
import vidhdr
from email.message import EmailMessage

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"

# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# Use this xml file
cascade = "haarcascade_frontalface_default.xml"



## --------------------- FUNCTION FOR SENDING OUT EMAILS --------------------- ##
def send_message():
	email_sender = os.environ.get('EMAIL_USER') ## email address of sender
	email_password = os.environ.get('EMAIL_PASS')  ##gmail app password
	email_receiver = 'receiver@gmail.com'

	# Getting the image to be sent as attachment
	with open ('image.jpg', 'rb') as image: #rb stands for read bytes
		image_data = image.read()
		image_type = imghdr.what(image.name)
		image_name = "image_{}.jpg".format(datetime.datetime.now().replace(microsecond=0))
	
	# Using ffmpeg to convert video file type
	file_datetime = datetime.datetime.now().strftime("%y%m%d_%H-%M-%S")
    	##formatting in YYYYMMDD_HH-MM-SS (HH is hour, MM is minute, SS is seconds )
	video_attachment_name = "output_{}.mp4".format(file_datetime)

	cmd_convert = "ffmpeg -i video.avi -vcodec h264 {}".format(video_attachment_name)
	os.system(cmd_convert) 

	# Getting the video to be sent as attachment
	with open (video_attachment_name, 'rb') as video: 
		video_data = video.read()
		video_type = 'mp4'
		video_name = "video_{}.mp4".format(datetime.datetime.now().replace(microsecond=0))

	# Email content
	message = EmailMessage()
	message['Subject'] = "Unknown Face Detected!"
	message['From'] = email_sender
	message['To'] = email_receiver
	message.set_content("An unkown face has been detected at {}".format(datetime.datetime.now().replace(microsecond=0)))


	# Attaching the image and video to the email
	message.add_attachment(image_data, maintype='image', subtype=image_type, filename = image_name)
	message.add_attachment(video_data, maintype='video', subtype=video_type, filename = video_name)
		
	with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:

		# Email login (need to be there for ssl and tls method)
		smtp.login(email_sender, email_password)

		# Sending the email
		smtp.send_message(message)

	# Cleaning Up - moving the video to video_collection directory
	cmd_move = "move {} video_collection".format(video_attachment_name)
	os.system(cmd_move)


## -------------------------------- MAIN CODE -------------------------------- ##

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# Loop over frames from the video file stream
while True:
	# Grab the frame from the threaded video stream and resize it to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# Convert the input frame from (1) BGR to grayscale (for face detection) 
	# and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Detect faces in the grayscale frame (m adjustable settings for better results)

	# ---------------------------- Settings Options --------------------------- #
	# scaleFactor -> specifies how much the image size is reduced at   			#
	# 				 image scale 												#
	# 	- 1.1 means the image is reduced by 10% at each step					#
	#	- the smaller the value, the higher the accuracy but also slower		#
	# minNeighbours -> higher number would require more overlapping 			#
	# 				   rectangles to confirm a detection						#
	#	- higher value would have higher accuracy								#
	# minSize -> minimum possible object size in the form of (width, height)	#
	#	- objects smaller than the specified size will be ignored 				#
	# 	- helps to reduce noise													#
	# --------------------------------------------------------------------------#

	rects = detector.detectMultiScale(gray, scaleFactor=1.05, 
		minNeighbors=7, minSize=(75, 75),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order but we need 
	# them in (top, right, bottom, left) order, so we need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	# Compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# Loop over the facial embeddings
	for encoding in encodings:
		# Display the image to our screen
		cv2.imshow("Facial Recognition is Running", frame)
		key = cv2.waitKey(1) & 0xFF

		# Attempt to match each face in the input image to our known encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# Check to see if we have found a match
		if True in matches:
			# Find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# Loop over the matched indexes and maintain a count for
			# each recognized face 
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# Determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			print("Known face is detected: ", name)

		# If someone in your dataset is not identified
		else:
			print("Unknown face detected")
			
			#Take a picture to send in the email
			img_name = "image.jpg"
			cv2.imwrite(img_name, frame)
			print('Taking a picture.')


			## ----------- CAPTURING A VIDEO ------------ ##
			
			#---------Specifying Video Properties---------#
			# specify the desired video codec             #
			# we need to specify the FourCC 4-byte code   #
			#   for .avi files --> use DIVX, XVID         #
			#   for .mp4 files --> use MJPG, mp4v         #
			#---------------------------------------------#
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			video_name = 'video.avi' 

			# Specifying the video dimensions and frame rate 
			# (either specify it or use the camera's dimemsions using vs.get)
			width =  640 #int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) 
			height = 480 #int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
			frame_rate = 300 #int(vs.get(cv2.CAP_PROP_FPS)) 
			
			video_dimension = (width, height)

			# Creating a VideoWriter object
			recorded_video = cv2.VideoWriter(video_name, fourcc, frame_rate, video_dimension)

			# Setting the duration of the video
			start_time = time.time()
			capture_duration = 5.0 #the video will be slightly more than 5 seconds as it is based on the frames per second (higher fps, the shorter the vid)

			print('Taking a video')
			while True:
				# Capturing the video
				video_frame = vs.read()
				
				# Writing the frame into the file video.avi
				recorded_video.write(video_frame)

				if ((int(time.time() - start_time) == capture_duration)):
					#stopping the recording
					break
			print('video ended')

			# Release the video file to allow it to be used for the email
			recorded_video.release()

			# Sending the email to notify of an unknown fae detected
			request = send_message()
			print("Email Sent")
		
	# Loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# Draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

	# Display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# If the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# Update the FPS counter
	fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
