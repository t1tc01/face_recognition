import cv2
import base64
import time

from router.camera_multi import Camera
from utils.recog_utils_grpc import *
from utils.detect_utils_grpc import *
#from utils.file_utils_2 import *

from utils.file_utils import *
# from utils.detect_utils import *
# from utils.recog_untils import *


#
PATH_TO_TARGET  = "/media/hoangphan/Data/code/acs/face_recog/save/target"
PATH_TO_CLASS = "/media/hoangphan/Data/code/acs/face_recog/save/class.txt"

#
def create_new_class(id: str, name: str):
    with open(PATH_TO_CLASS, "r") as file:
        content = file.read()
        if id in content:
            print(content)
            print("ID đã tồn tại!")
            return -1
    with open(PATH_TO_CLASS, "a") as file:
        text = id + " " + name
        file.write(text + "\n")
        os.mkdir(os.path.join(PATH_TO_TARGET,str(id)))
        print("Đã khởi tạo: ", id, name)
        return id

#
def get_list_class(path_to_class=PATH_TO_CLASS):
    list_person_img = []
    list_cls = []
    with open(path_to_class, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cl,  label= line.strip().split(" ",maxsplit=1)
            list_person_img.append(label)
            list_cls.append(cl)
    return list_cls, list_person_img

#list_class is ID, list_person_img is name of person

list_cls, list_person_img = get_list_class(PATH_TO_CLASS)

#feat_list is feature of 
# feat_lis = create_feat_list(list_cls,PATH_TO_TARGET)

create_feat_list_file(list_cls,PATH_TO_TARGET)
feat_list =load_feat_list_file()
print(len(feat_list))

def reload_feat_list():
    global feat_list
    global list_cls, list_person_img
    feat_list =load_feat_list_file()
    list_cls, list_person_img = get_list_class(PATH_TO_CLASS)
    print("number of class:", len(feat_list))

#
def process_frame_for_get_data(frame, checkIn=True):
    raw_image = frame
    raw_h, raw_w,_ = raw_image.shape
    raw_image  =cv2.resize(raw_image,(640, 640))
    list_face_b = detection_on_frame(raw_image)
    list_face = crop_face(raw_image, list_face_b)

    for b in list_face_b:
        # b = list(map(lambda x:int(round(x, 0)), b))
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

    create_feat_list_file()
    return raw_image, list_face

#
def process_frame_to_crop(frame, id: str, name_face: str):
    raw_image = frame
    raw_h, raw_w,_ = raw_image.shape
    raw_image  =cv2.resize(raw_image,(640, 640))
    list_face_b = detection_on_frame(raw_image)
    list_face = crop_face(raw_image, list_face_b)

    for i in range(len(list_face)):
        face = cv2.resize(list_face[i], (112,112))
        name_face = str(id) + "_" + str(time.time()) + ".jpg"
        cv2.imwrite(os.path.join(PATH_TO_TARGET,str(id),name_face),face)
        print("Đã lưu ",name_face,"to", os.path.join(PATH_TO_TARGET,str(id)))

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
    create_feat_list_file()
    return raw_image

#
def process_frame_for_infer(frame, checkIn=True):
    raw_image = frame
    raw_h, raw_w,_ = raw_image.shape
    raw_image  =cv2.resize(raw_image,(640, 640))
    list_face_b = detection_on_frame(raw_image)
    list_face = crop_face(raw_image, list_face_b)

    text_name = ""
    score = ""
    id_name = "-"
    try:
        for face in list_face: 
            face_feat = inference_on_image(face)
            index, score = iden(face_feat, feat_list)
            if index == -1:
                text_name = "unknown"
            else:
                text_name = list_person_img[index]
                id_name = list_cls[index]
                print(id_name, text_name)
    except:
        text_name = "unknown"
        print("except!")

    #save check 
    if id_name != "-":
        save_check_to_csv(id_name, checkIn)
        # sumary_day()

    for b in list_face_b:
        text = "{} {}".format(text_name, id_name)
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

    raw_image = cv2.resize(raw_image,(raw_w, raw_h))
    return raw_image

#
def process_frame(frame):
    """
        Input: a frame (np.ndarray)
        Output: a drawed frame (bbox, name, landmark) with same size and type, coordinate is in image 640x640
    """
    result_data = {}
    raw_image = frame
    raw_h, raw_w,_ = raw_image.shape
    raw_image  =cv2.resize(raw_image,(640, 640))
    list_face_b = detection_on_frame(raw_image)
    list_face = crop_face(raw_image, list_face_b)

    text_name = ""
    score = ""
    try:
        for face in list_face: 
            face_feat = inference_on_image(face)
            index, score = iden(face_feat, feat_list)
            if index == -1:
                text_name = "unknown"
            else:
                text_name = list_person_img[index]
    except:
        text_name = "unknown"


    result_data.update({"text_name": text_name, "score":str(score)})
    for b in list_face_b:
        b = list(map(int, b))
        result_data.update({"x1":b[0], "y1":b[1], "x2":b[2], "y2":b[3]}) #bbox corrdinate

        # landms
        result_data.update({"lm1x":b[5], "lm1y":b[6], "lm2x":b[7], "lm2y":b[8], "lm3x":b[9], "lm3y":b[10],
                            "lm4x":b[11], "lm4y":b[12], "lm5x":b[13], "lm5y":b[14]})
    return result_data

#
def dataURL2numpy(data_url):
    """"
    Get data_url base64 of a frame and convert to numpy array
    """
    img_url = data_url.split('image/png;base64,')[1]
    img_bytes = base64.b64decode(img_url)
    im_arr = np.frombuffer(img_bytes, dtype=np.int8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


#
def gen_camera_for_infer(camera, checkin: bool):
    while True:
        frame = camera.get_frame()
        frame = process_frame_for_infer(frame, checkin)
        byte_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')

#
def gen_camera_for_add_class(camera, id, name):
    while True:
        frame = camera.get_frame()
        frame, list_face = process_frame_to_crop(frame, id, name)
        byte_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')