from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pathlib import Path
import cv2
import uvicorn
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from json import JSONDecodeError
from PIL import Image
import base64
from io import BytesIO
import json
import csv

from router.camera_multi import Camera
from utils.detect_utils import *
from utils.recog_untils import *


#
PATH_TO_TARGET  = "target"
PATH_TO_CLASS = "class.txt"

#pydantic to take data
class Item(BaseModel):
    eth_addr: str

app = FastAPI()

templates = Jinja2Templates(directory="templates")

def get_list_class(path_to_class):
    list_person_img = []
    list_cls = []
    with open(path_to_class, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cl,  label= line.strip().split(" ",maxsplit=1)
            list_person_img.append(label)
            list_cls.append(cl)
    return list_cls, list_person_img

#
def process_frame_camera_local_for_infer(frame):
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
            index, score = iden(face_feat, feat_lis)
            if index == -1:
                text_name = "unknown"
            else:
                text_name = list_person_img[index]
    except:
        text_name = "unknown"

    for b in list_face_b:
        text = "{} {}".format(text_name, score)
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
            index, score = iden(face_feat, feat_lis)
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
def gen_camera_for_infer(camera):
    while True:
        frame = camera.get_frame()
        frame = process_frame_camera_local_for_infer(frame)
        byte_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')

def gen_camera_for_add_class(camera):
    while True:
        frame = camera.get_frame()
        frame = process_frame_camera_local_for_infer(frame)
        byte_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')

list_cls, list_person_img = get_list_class(PATH_TO_CLASS)
feat_lis = create_feat_list(list_cls,PATH_TO_TARGET)


"""
Call camera from server
"""
@app.get("/",response_class=HTMLResponse)
async def index(request: Request):
   return templates.TemplateResponse('index.html', {"request": request})

@app.get("/streaming")#, response_class=HTMLResponse)
async def streaming():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return  StreamingResponse(gen_camera_for_infer(Camera()),
                    media_type='multipart/x-mixed-replace; boundary=frame')

"""
Open client's camera 
"""
#Video streaming from a camera that is connectêd with server
@app.get('/client_stream', response_class=HTMLResponse) 
async def client_stream(request: Request):
   return templates.TemplateResponse('client_stream.html', {"request": request})

@app.post("/client_stream")
async def image_data(request: Request):
    content_type = request.headers.get('Content-Type')
    if content_type is None:
        return 'No Content-Type provided.'
    elif content_type == 'application/json':
        try:
            data_json = await request.json()
            data_url = data_json['data']
            img_url = data_url.split('image/png;base64,')[1]
            img_bytes = base64.b64decode(img_url)
            im_arr = np.frombuffer(img_bytes, dtype=np.int8)
            img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            result_data = process_frame(img)
            #print(result_data)
            #json_data = json.dumps(result_data)
            return result_data #JSONResponse(content=json_data)
        except JSONDecodeError:
            return 'Invalid JSON data.'
    else:
        return 'Content-Type not supported.'



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8008)