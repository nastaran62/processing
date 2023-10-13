import os
import numpy as np
import cv2
from processing.preprocessing.models.yolo_utils import *

current_directory = os.path.dirname(os.path.abspath(__file__))
net = cv2.dnn.readNetFromDarknet('{0}/models/yolov3-face.cfg'.format(current_directory),
                                 '{0}/models/model_weights/yolov3-wider_16000.weights'.format(current_directory))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def normalization(image, width, height):
    img = cv2.resize(image, (width, height))
    normalizedImg = np.zeros((width, height))
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    return normalizedImg

def yolo_face_detection(frames, img_width=64, img_height=64):
    detected_faces_frames = []
    i = 0
    for frame in frames:
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
  
        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        if len(faces) == 0:
            print("No face")
            i += 1
            continue

        # We need the bigest face in the frame. In DEAP dataset some frames has
        # a screen which display stimuli with face
        detected_face = faces[0]
        if len(faces) > 1:
            i = 1
            while i < len(faces):
                if faces[i][2] > detected_face[2]:
                    detected_face = faces[i]
                i += 1

        [x, y, width, height] = detected_face
        X = 0 if x<0 else x
        Y = 0 if y<0 else y
        output = frame[Y:y+height, X:x+width]
        gray_image = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        normalized = normalization(gray_image, img_width, img_height)
        detected_faces_frames.append(normalized)
        i += 1
    return detected_faces_frames