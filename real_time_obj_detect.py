"""
	

	FUCK THAT SHIT
	I'M OUT



"""
import numpy as np
import argparse
import time
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize the list of class and generate a set of box colors
f = open("classes.txt", 'r')
CLASSES = [line.strip() for line in f]	
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()
# load serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromTensorflow(args["model"], args["prototxt"]) 

# if a video path was not supplied, grab the reference to the web cam
print("[INFO] starting video stream...")
if not args["video"]:
	vs = cv2.VideoCapture(0)	
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["video"])

start_time = time.time()
num_frames = 0

# loop over frames from the video stream
while True:
	num_frames += 1
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	ret, frame = vs.read()
	# check to see if we have reached the end of the stream
	if frame is None:
		break

	h = frame.shape[0]
	w = frame.shape[1]

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)

	# grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	boxes = []

	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and get the detections
	net.setInput(blob)
	detections = net.forward()
	
	(success, boxes) = trackers.update(frame)
	
	# draw the prediction box on the frame
	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		(startX, startY, endX, endY) = [int(v) for v in box]
		#idx = box[0]
		#confidence = box[1]
		#label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[1], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		#cv2.putText(frame, label, (startX, y),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` 
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, compute the (x, y)-coordinates of
			# the bounding box for the object
			tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			box_new = (startX, startY, endX, endY)
			trackers.add(tracker, frame, box_new)	
			#boxes.append((idx, confidence, startX, startY, endX, endY))


	# create a new object tracker for the bounding box and add it
	# to our multi-object tracker

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are using a webcam, release the pointer
if args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

seconds = time.time() - start_time
fps = num_frames / seconds
print("[INFO] elapsed time: {:.2f}".format(seconds))
print("[INFO] approx. FPS: {:.2f}".format(fps))

# close all windows
cv2.destroyAllWindows()