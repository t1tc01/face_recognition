# camera_multi.py

import cv2
from router.base_camera import BaseCamera


class Camera(BaseCamera):
    def __init__(self):
        super().__init__()

    # over-wride of BaseCamera class frames method
    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()            
            yield img