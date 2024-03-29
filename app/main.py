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


from utils.file_utils import *
from utils.infer_utils import *

#pydantic to take data
class Item(BaseModel):
    eth_addr: str

app = FastAPI()

templates = Jinja2Templates(directory="templates")

"""
Open camera 
"""
@app.get("/",response_class=HTMLResponse)
async def index(request: Request):
   return templates.TemplateResponse('index.html', {"request": request})

@app.get("/streaming_checkin")
async def streaming_checkin():
    return  StreamingResponse(gen_camera_for_infer(Camera(), True),
                    media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/streaming_checkout")
async def streaming_checkout():
    return  StreamingResponse(gen_camera_for_infer(Camera(), False),
                    media_type='multipart/x-mixed-replace; boundary=frame')


"""
Get data 
"""
@app.get("/get_data", response_class=HTMLResponse)
async def get_data(request: Request):
    return templates.TemplateResponse('getdata.html', {"request": request})

@app.get("/streaming_data")
async def streaming_data():
    return  StreamingResponse(gen_camera_for_add_class(Camera()),
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
            img = dataURL2numpy(data_url)
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