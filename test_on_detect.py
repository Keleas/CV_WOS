"""

	the main problem is EPSILON with we need aproximade objects the same classes
	on pack of frames 
	EPSILON is very suitable about distance and scale's objects 

"""
import numpy as np
import argparse
import time
import cv2


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
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 4))
SIZE_WINDOW = 7
# load serialized model
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromTensorflow(args["model"], args["prototxt"]) 


# initialize the video stream and time timer
print("[INFO] starting video stream...")
if args["video"]:
	vs = cv2.VideoCapture(args["video"])
else:
	vs = cv2.VideoCapture(0)
	time.sleep(1.0)
start_time = time.time()
num_frames = 0

detect_array = [] # array of predictions with size is SIZE_WINDOW

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	ret, frame = vs.read()
	if frame is None:
		break
	num_frames += 1
	frame = cv2.resize(frame, (1440, 900))

	# grab the frame dimensions and convert it to a blob
	h = frame.shape[0]
	w = frame.shape[1]

	# for normal view
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
	detection = net.forward()

	# slice window with size is SIZE_WINDOW
	if len(detect_array) != SIZE_WINDOW:
		detect_array.append(detection)
		continue
	else:
		for i in range(len(detect_array) - 1):
			detect_array[i] = detect_array[i+1]
		detect_array[len(detect_array)-1] = detection

	# loop over slice window to get correcly prediction frame
	# detect all prediction classes
	classes_array = []
	for j in range(len(detect_array) - 1):
		detections = detect_array[j]
		# collect all the general classes of cadres whose confidence is greater than the general 
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (probability) associated with the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` 
			if confidence > args["confidence"]:
				# save index, coords and confidence of detected class
				idx = int(detections[0, 0, i, 1])
				(x, y, z, k) = detections[0, 0, i, 3:7]
				classes_array.append((idx, x, y, z, k, confidence))


	# loop over detection classes
	if classes_array:
		# prepeare to loop over each class 
		classes_array.sort()
		av_coords = np.zeros((1,4)) # avarage coordinate (startX, startY, endX, endY) of the same slass 
		count = 0 # number of elements of the same class
		last_idx = classes_array[0][0]
		len_loop = 0 # size of loop 
		av_confidence = 0 # avarage confidence of the same class
		(last_start_X, last_start_Y, last_end_X, last_end_Y) = classes_array[0][1:5]
		EPSILON = 0.01 # epsilon square to detect same model


		for cur_class in classes_array:
			# take data about current class on classes_array 
			# and get avarage data to the same member of the class
			(start_X, start_Y, end_X, end_Y) = cur_class[1:5] 
			av_coords += (start_X, start_Y, end_X, end_Y) * np.array([w, h, w, h])
			av_confidence += cur_class[5]

			idx = cur_class[0]
			"""
			print(cur_class)
			print("[INFO] {}".format(abs(last_start_X - start_X)))
			print("[INFO] {}".format(abs(last_start_Y - start_Y)))
			print("[INFO] {}".format(abs(last_end_X - end_X)))
			print("[INFO] {}".format(abs(last_end_Y - end_Y)))
			"""
			cur_center_X = (start_X + end_X) / 2.0
			cur_center_Y = (start_Y + end_Y) / 2.0
			last_center_X = (last_start_X + last_end_X) / 2.0
			last_center_Y = (last_start_Y + last_end_Y) / 2.0

			if last_idx == idx and len_loop != len(classes_array) - 1 and count < 5 and abs(cur_center_X - last_center_X) <= EPSILON and abs(cur_center_Y - last_center_Y) <= EPSILON:
				count += 1
				print("[COUNT] {} [IDX] {}".format(count, idx))
			else:
				if count > 2:
					# take geometric mean
					box = av_coords / count
					av_confidence /= count

					# draw the prediction box on the frame
					(startX, startY, endX, endY) = box.astype("int")[0]
					label = "{}: {:.2f}%".format(CLASSES[last_idx], av_confidence * 100)

					cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[last_idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

				av_coords = np.zeros((1,4))
				av_confidence = 0
				count = 1

			(last_start_X, last_start_Y, last_end_X, last_end_Y) = (start_X, start_Y, end_X, end_Y)
			last_idx = idx
			len_loop += 1
				

		# show the output frame
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