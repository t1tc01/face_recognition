import sys 

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

from utils.infer_utils import *

PATH_TO_TARGET  = "/media/hoangphan/Data/code/acs/face_recog/save/target"
list_cls, list_person_img = get_list_class()

#
def open_camera_window(window, id, name_face):
    camera_window = tk.Toplevel(window)
    camera = cv2.VideoCapture(0)

    def close_camera_window():
        camera.release()
        camera_window.destroy()
        list_cls, list_person_img = get_list_class()
        create_feat_list_file(list_cls,PATH_TO_TARGET)
        messagebox.showinfo(title="Attention", message="Saved to feature store")
        reload_feat_list()

    def show_frame():
        ret, frame = camera.read()
        if ret:
            frame = process_frame_to_crop(frame, id, name_face)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        video_label.after(1, show_frame)

    video_label = tk.Label(camera_window)
    video_label.pack()

    show_frame()

    camera_window.after(2000, close_camera_window)

#
def on_add_person(window, id, name):
    messagebox.showinfo(title="Attention", message="look straight at the camera!")
    open_camera_window(window, id, name)
   

#
def open_submit_window(window):

    new_window = tk.Toplevel(window)
    #
    def submit_form():
        user_id = id_entry.get()
        username = name_entry.get()
        print("ID:", user_id)
        print("Username:", username)
        # Thực hiện xử lý dữ liệu nhập

        on_add_person(window, user_id, username)
        create_new_class(user_id, username)

        new_window.destroy()
    
    # Label và Entry cho ID
    id_label = tk.Label(new_window, text="ID:")
    id_label.grid(row=0, column=0, padx=10, pady=10)

    id_entry = tk.Entry(new_window)
    id_entry.grid(row=0, column=1, padx=10, pady=10)

    # Label và Entry cho Tên người dùng
    name_label = tk.Label(new_window, text="Tên người dùng:")
    name_label.grid(row=1, column=0, padx=10, pady=10)

    name_entry = tk.Entry(new_window)
    name_entry.grid(row=1, column=1, padx=10, pady=10)

    # Button để gửi form
    submit_button = tk.Button(new_window, text="Gửi", command=submit_form)
    submit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)


if __name__ == "__main__":
    window = tk.Tk()
    open_camera_button = tk.Button(window, text="Mở camera", command=on_add_person(window,"-2", "21"))
    open_camera_button.pack(pady=10)

    window.mainloop()