import os
import cv2
from models.yolo_utils import *

current_directory = os.path.dirname(os.path.abspath(__file__))
net = cv2.dnn.readNetFromDarknet('{0}/models/yolov3-face.cfg'.format(current_directory),
                                 '{0}/models/model_weights/yolov3-wider_16000.weights'.format(current_directory))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def test_yolo_face_detection():
    video_path = "/media/nastaran/HDD/projects/f2f_remote/output/p16/webcam/webcam-16-01-13.avi"
    cap = cv2.VideoCapture(0)
    i = 0
    detected_faces = []
    while True:
        has_frame, frame = cap.read()
        # Stop the program if reached end of video
        if not has_frame:
            cv2.waitKey(1000)
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(faces) == 0:
            continue
        
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print(faces)
        [x, y, width, height] = faces[0]
        X = 0 if x<0 else x
        Y = 0 if y<0 else y
        output = frame[Y:y+height, X:x+width]
        gray_image = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        #normalized = normalization(gray_image, img_width, img_height)
        detected_faces.append(gray_image)

        # initialize the set of information we'll displaying on the frame
        #info = [
        #    ('number of faces detected', '{}'.format(len(faces)))
        #]

        #for (i, (txt, val)) in enumerate(info):
            #text = '{}: {}'.format(txt, val)
            #cv2.putText(frame, text, (10, (i * 20) + 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        cv2.imshow("aaaa", gray_image)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')
if __name__ == "__main__":
    test_yolo_face_detection()
