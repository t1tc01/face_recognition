import sys 

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

from utils.detect_utils_grpc import *
from utils.recog_utils_grpc import *
from utils.infer_utils import *
from utils.file_utils import *
from utils.detect_utils_grpc import *




#
def open_camera_window(window):
    camera_window = tk.Toplevel(window)
    camera = cv2.VideoCapture(0)

    def close_camera_window():
        camera.release()
        camera_window.destroy()

    def show_frame():
        ret, frame = camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = process_frame_for_infer(frame,True)
            current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=current_image)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        window.after(1, show_frame) 

    video_label = tk.Label(camera_window)
    video_label.pack()

    show_frame()

    camera_window.protocol("WM_DELETE_WINDOW", close_camera_window)

#
def on_open_check_in(window):
    messagebox.showinfo(title="Thông báo", message="Mở chế độ check-in!")
    open_camera_window(window)

