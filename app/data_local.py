"""
Get new class from local
"""

import tritonclient.http as httpclient
import cv2 
import numpy as np
import os

from utils.detect_utils_grpc import *

PATH_TARGET  = "/media/hoangphan/Data/code/acs/face_recog/save/target"
PATH_CLASS = "/media/hoangphan/Data/code/acs/face_recog/save/class.txt"

def create_new_class():
    print("Khởi tạo id và tên người mới (cách ra một dấu cách) VD: <id>_<username>:")
    try:
        text = input().strip()
        id, name = text.split(" ", maxsplit=1)
    except:
        print("Lỗi cú pháp!")

    with open(PATH_CLASS, "r") as file:
        content = file.read()
        if id in content:
            print(content)
            print("ID đã tồn tại!")
            return -1
    with open(PATH_CLASS, "a") as file:
        file.write(text + "\n")
        os.mkdir(os.path.join(PATH_TARGET,str(id)))
        print("Đã khởi tạo: ", id, name)
        return id

if __name__ == "__main__":

    id = create_new_class()
    if id == -1:
        exit()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

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
        raw_h, raw_w,_ = raw_image.shape
        raw_image  =cv2.resize(raw_image,(640, 640))
        list_face_b = detection_on_frame(raw_image)
        list_face = crop_face(raw_image, list_face_b)

        for i in range(len(list_face)):
            face = cv2.resize(list_face[i], (112,112))
            name_face = str(id) + "_" + str(time.time()) + ".jpg"
            print(os.path.join(PATH_TARGET,str(id),name_face))
            cv2.imwrite(os.path.join(PATH_TARGET,str(id),name_face),face)
            print("Đã lưu ",name_face)
        #time when we finish processing for this frame
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        fps = str(fps)
        # putting the FPS count on the frame
        cv2.putText(raw_image, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        for b in list_face_b:
            b = list(map(int, b))
            cv2.rectangle(raw_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12

            # landms
            cv2.circle(raw_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(raw_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(raw_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(raw_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(raw_image, (b[13], b[14]), 1, (255, 0, 0), 4)

        raw_image = cv2.resize(raw_image,(raw_w, raw_h))
        cv2.imshow("Camera", raw_image)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
