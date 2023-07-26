"""
This is fast api that call to:
* Take frame from client (Client POST)
* preprocess frame and request respond from triton server
* post process respond from triton inference server 
* Respond to Client (GET)
"""
import cv2
import tritonclient.http as httpclient
import time

# from utils.recog_untils import *
#from utils.detect_utils import *

from utils.detect_utils_grpc import *
from utils.recog_utils_grpc import *
from utils.infer_utils import *
from utils.file_utils import * 


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera!")
        exit()

    while True: 
        ret, frame = cap.read()
        if not ret:
            print("Can't read from camera!")
            break
        raw_image = frame
        raw_image = process_frame_for_infer(raw_image)
        cv2.imshow("Camera", raw_image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()