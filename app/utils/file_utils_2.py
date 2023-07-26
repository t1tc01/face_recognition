"""
    This is upgrade of file_utils.py, eliminate check in and check out, summary of the day will 
    take first checkin and last checkin.

    file structure is the same with file_utils.py, but dont have check in and check out anymore
"""
import os
import csv
import time 
import pandas as pd 
from datetime import datetime

PATH_TO_SAVE = "/media/hoangphan/Data/code/acs/face_recog/save"
PATH_TO_CLASS = "/media/hoangphan/Data/code/acs/face_recog/save/class.txt"



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
list_cls, list_person = get_list_class(PATH_TO_CLASS)

def get_first_and_last_checkin_by_ID(id: str):

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    local_time = time.localtime()
    current_day = local_time.tm_mday
    current_month = local_time.tm_mon
    current_year = local_time.tm_year

    year_path = os.path.join(PATH_TO_SAVE, str(current_year))
    month_path = os.path.join(PATH_TO_SAVE,str(current_year) 
                                ,str(current_month) + str(current_year))
    day_path  = os.path.join(PATH_TO_SAVE,str(current_year) 
                                ,str(current_month) + str(current_year), 
                                str(current_day) + str(current_month) + str(current_year)) 

    ckin_session_path = os.path.join(day_path, "checkin.csv")
    
    first_checkin = ""
    last_checkin = ""

    #From id, get classname
    idx = -1
    class_name = "unknown"
    for i in range(len(list_cls)):
        if list_cls[i] == id:
            idx = i
    class_name = list_person[idx]

    if os.path.exists(ckin_session_path):
        fieldnames = ['ID', 'Name' ,'Time check in']
        with open(ckin_session_path, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            data_checkin = [row for row in csv_reader] #list dicts

    for i in range(len(data_checkin)):
        if data_checkin[i]["ID"] == id:
            first_checkin = data_checkin[i]["Time check in"]
            break
    
    for i in range(len(data_checkin)-1,-1,-1):
        if data_checkin[i]["ID"] == id:
            last_checkin = data_checkin[i]["Time check in"]
            break
    data = {"ID": id, "Name": class_name ,"First check in": first_checkin, "Last check in": last_checkin}
    return data


def sumary_day():
    """
    Summary user 
    """    
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    local_time = time.localtime()
    current_day = local_time.tm_mday
    current_month = local_time.tm_mon
    current_year = local_time.tm_year

    day_path  = os.path.join(PATH_TO_SAVE,str(current_year) 
                                ,str(current_month) + str(current_year), 
                                str(current_day) + str(current_month) + str(current_year))

    summary_day_path = os.path.join(day_path,"sumary_day.csv")
    fieldnames = ["ID", "Name", "First check in", "Last check in"]

    sum_data = []
    for i in range(len(list_cls)):
        data = get_first_and_last_checkin_by_ID(list_cls[i])
        sum_data.append(data)
    with open(summary_day_path, "w", newline='') as csv_file: #if have checkin yet, check in
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for data in sum_data:
            writer.writerow(data)
        print("Saved sumary day!")
        csv_file.close()


def save_check_to_csv(id: str):
    """
    Save data to csv file when user checkin
    """ 


    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    local_time = time.localtime()
    current_day = local_time.tm_mday
    current_month = local_time.tm_mon
    current_year = local_time.tm_year

    year_path = os.path.join(PATH_TO_SAVE, str(current_year))
    month_path = os.path.join(PATH_TO_SAVE,str(current_year) 
                                ,str(current_month) + str(current_year))
    day_path  = os.path.join(PATH_TO_SAVE,str(current_year) 
                                ,str(current_month) + str(current_year), 
                                str(current_day) + str(current_month) + str(current_year))

    #Create folder for each day in real time 
    if os.path.exists(year_path) == False:
        os.mkdir(year_path)
    if os.path.exists(month_path) == False:
        os.mkdir(month_path)
    if os.path.exists(day_path) == False:
        os.mkdir(day_path)

    #From id, get classname
    idx = -1
    class_name = "unknown"
    for i in range(len(list_cls)):
        if list_cls[i] == id:
            idx = i
    class_name = list_person[idx]

    #Save to data
    data = {"ID": id, "Name": class_name ,'Time check in': formatted_now}

    #
    fieldnames = ['ID', "Name" ,'Time check in']
    ckin_session_path = os.path.join(day_path, "checkin.csv")

    #save to csv
    with open(ckin_session_path, "a", newline='') as csv_file: #if have checkin yet, check in
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(data)
        print("[{}] {} {} check in!".format(formatted_now, id, class_name))
    csv_file.close()

if __name__ == "__main__":
    sumary_day()
    

    



