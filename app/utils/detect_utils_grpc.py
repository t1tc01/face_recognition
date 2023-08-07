import sys
import numpy as np
import cv2
import time
import torch
from itertools import product as product
from math import ceil

import tritonclient.grpc as grpcclient

#Constant for application
LONG_SIDE = 640

cfg_blaze = {
    'name': 'Blaze',
    # origin anchor
    # 'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
    # kmeans and evolving for 640x640
    'min_sizes': [[8, 11], [14, 19, 26, 38, 64, 149]],
    'steps': [8, 16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1,
    'cls_weight': 6,
    'landm_weight': 0.1,
    'gpu_train': True,
    'batch_size': 256,
    'ngpu': 1,
    'epoch': 200,
    'decay1': 130,
    'decay2': 160,
    'decay3': 175,
    'decay4': 185,
    'image_size': 320,
    'num_classes':2
}

nms_threshold = 0.1
confidence_threshold = 0.3
keep_top_k = 1000
top_k = 2000
vis_thres = 0.3

#Create grpc client
try:
    client = grpcclient.InferenceServerClient(
        url="localhost:8001",
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
    )
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()

#
class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        #output = torch.Tensor(anchors).view(-1, 4)
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            #output.clamp_(max=1, min=0)
            output = np.clip(output, a_min=0, a_max=0)
        return output #output.numpy()

#Preprocess
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 16), np.mod(dh, 16)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


#Post-process
#
def decode_landm(pre, priors, variances):
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)
    return landms

#
def decode(loc, priors, variances):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

#
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

#Pipeline
#
def detection_on_frame(raw_img):
    """
    It's include pre and post process 
    Return list detected face 
    """
    #Pre process
    img = np.float32(raw_img)

    target_size = LONG_SIDE
    

    img = cv2.resize(img, (640,640))
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    #
    img, ratio, (dw, dh) = letterbox(img, (target_size, target_size), color=(104, 117, 123), auto=False, scaleFill=False)
    resize = np.max(ratio)
    im_height, im_width, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    detection_input = grpcclient.InferInput( #Create input to triton server
            "input", img.shape, datatype="FP32"
        )
    detection_input.set_data_from_numpy(img)

    # Query the server
    detection_response = client.infer(
        model_name="blazeface", inputs=[detection_input]
    )

    #Get result from respond
    locs = detection_response.as_numpy("boxes")
    conf = detection_response.as_numpy("scores")
    landms = detection_response.as_numpy("landmark")

    #Post process output
    priorbox = PriorBox(cfg_blaze, image_size=(im_height, im_width))
    priors = priorbox.forward()              
    prior_data = priors
    boxes = decode(locs.squeeze(0), prior_data, cfg_blaze['variance'])
    boxes = boxes * scale / resize #boxes

    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, cfg_blaze['variance'])
    scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    
    landms = landms * scale1 / resize
    
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    face_det_list = []

    for b in dets:
        if b[4] < vis_thres:
            continue
        face_det_list.append(b)
    
    return face_det_list

#
def crop_face(img_raw, list_face_b):
    list_face = []
    for b in list_face_b:
        pt1 = (b[0],b[1])
        pt2 = (b[2], b[3])
        crop_img = img_raw[int(pt1[1]):int(pt2[1]), int(pt1[0]):int(pt2[0])]
        list_face.append(crop_img)
    return list_face

