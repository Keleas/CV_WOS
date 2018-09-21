"""

	THE FIRST VERSION OF DETECT ALGO
	simple detect object without splay window and flowing (aprox) detection

"""
import numpy as np
import argparse
import time
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", required=False,
	help="path to video file")
args = vars(ap.parse_args())

# initialize the list of class and generate a set of box colors
f = open("classes.txt", 'r')
CLASSES = [line.strip() for line in f]	
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromTensorflow(args["model"], args["prototxt"]) 

# initialize the video stream and time timer
print("[INFO] starting video stream...")
if args["video"]:
	vs = cv2.VideoCapture(args["video"])
else:
	vs = cv2.VideoCapture(0)
	time.sleep(2.0)
start_time = time.time()
num_frames = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	ret, frame = vs.read()
	if frame is None:
		break
	num_frames += 1
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	h = frame.shape[0]
	w = frame.shape[1]

	# for noraml view
	"""
	if not args["video"]:
		center = (w / 2, h / 2)
		M = cv2.getRotationMatrix2D(center, 270, 1.0)
		frame = cv2.warpAffine(frame, M, (w, h))
	"""

	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and get the detections
	net.setInput(blob)
	detections = net.forward()


	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` 
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction box on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	#time.sleep(0.1)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
seconds = time.time() - start_time
fps = num_frames / seconds
print("[INFO] elapsed time: {:.2f}".format(seconds))
print("[INFO] approx. FPS: {:.2f}".format(fps))


cv2.destroyAllWindows()
vs.release()