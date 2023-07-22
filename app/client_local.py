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

from utils.detect_utils import *
from utils.recog_untils import *

PATH_TO_TARGET = "target" #target folder used to compare, should be local or gg driver, .etc


if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    list_person_img = []
    list_cls = []
    with open('class.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cl,  label= line.strip().split(" ",maxsplit=1)
            list_person_img.append(label)
            list_cls.append(cl)
    
    print(list_person_img)

    feat_lis = create_feat_list(list_cls,PATH_TO_TARGET)

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

        try:
            for face in list_face: 
                face_feat = inference_on_image(face)
                index, score = iden(face_feat, feat_lis)
                if index == -1:
                    text_name = "unknown"
                else:
                    text_name = list_person_img[index]
        except:
            continue

        #time when we finish processing for this frame
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        # putting the FPS count on the frame


        cv2.putText(raw_image, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        for b in list_face_b:
            text = "{} {:.4f}".format(text_name, score)
            # b = list(map(lambda x:int(round(x, 0)), b))
            b = list(map(int, b))
            cv2.rectangle(raw_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(raw_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(raw_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(raw_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(raw_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(raw_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(raw_image, (b[13], b[14]), 1, (255, 0, 0), 4)

        
        #print("face num:", len(list_face_b))
        raw_image = cv2.resize(raw_image,(raw_w, raw_h))
        cv2.imshow("Camera", raw_image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()