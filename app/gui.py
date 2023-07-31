import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

from utils.file_utils import *

from add_class_window import open_submit_window
from check_in_window import on_open_check_in
from check_out_window import on_open_check_out

# Khởi tạo cửa sổ giao diện
window = tk.Tk()
window.title("Check In/Check Out")
window.geometry("800x600")

#
def restart_window():
    # Xóa các thành phần hiện tại của cửa sổ
    for widget in window.winfo_children():
        widget.destroy()
    
    # Tạo lại các thành phần cần thiết
    create_widgets()

#
def create_widgets():
   # Khởi tạo nút "Check In"
    check_in_button = tk.Button(window, text="Check In", command=check_in)
    check_in_button.grid(row=2, column=0, padx=10, pady=10)

    # Khởi tạo nút "Check Out"
    check_out_button = tk.Button(window, text="Check Out", command=check_out)
    check_out_button.grid(row=2, column=1, padx=10, pady=10)

    # Khởi tạo nút "Exit"
    exit_button = tk.Button(window, text="Exit", command=exit_program)
    exit_button.grid(row=2, column=2, padx=10, pady=10)

    # Khởi tạo nút "Aggregate"
    aggregate_button = tk.Button(window, text="Aggregate", command=aggregate)
    aggregate_button.grid(row=2, column=3, padx=10, pady=10)

    #Khởi tạo nút create class
    add_person_button = tk.Button(window, text="Add person", command=add_class)
    add_person_button.grid(row=2, column=4, padx=10, pady=10)


#
def add_class():
    open_submit_window(window)

# 
def check_in():
    on_open_check_in(window)
    

# Hàm xử lý sự kiện khi nhấn nút "Check Out"
def check_out():
  on_open_check_out(window)

# Hàm xử lý sự kiện khi nhấn nút "Exit"
def exit_program():
    window.destroy()

# Hàm xử lý sự kiện khi nhấn nút "Aggregate"
def aggregate():
    sumary_day()

if __name__ == "__main__":

    create_widgets()

    # Chạy giao diện chính
    window.mainloop()