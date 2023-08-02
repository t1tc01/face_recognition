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

def sumary_day():
    """
        Get first time check in and check out of each id and summary to csv file (summary file of the day)
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
    checkin_filepath = os.path.join(day_path, "check_in", "checkin.csv")
    checkout_filepath = os.path.join(day_path, "check_out", "checkout.csv")
    sumary_path = os.path.join(day_path, "summary.csv")

    data_checkin = []
    data_checkout = []
    data_sumary = []

    

    if os.path.exists(checkin_filepath):
        fieldnames = ['ID', 'Time check in']
        with open(checkin_filepath, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            data_checkin = [row for row in csv_reader] #list dicts
            
    if os.path.exists(checkout_filepath):
        fieldnames = ['ID', 'Time check out']
        with open(checkout_filepath, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            data_checkout = [row for row in csv_reader]
    
    for i in range(len(data_checkin)):
        for j in range(len(data_checkout)):
            if data_checkin[i]["ID"] == data_checkout[j]["ID"]:

                #From id, get classname
                idx = -1
                class_name = "unknown"
                for k in range(len(list_cls)):
                    if list_cls[k] == data_checkin[i]["ID"]:
                        idx = k
                class_name = list_person[idx]

                #
                sumary = {"ID":data_checkin[i]["ID"], "Name": class_name ,"Time check in":  data_checkin[i]["Time check in"], "Time check out" : data_checkout[j]["Time check out"]}
                data_sumary.append(sumary)

    fieldnames = ["ID", "Name" ,"Time check in", "Time check out"]
    with open(sumary_path, "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for data in data_sumary:
            writer.writerow(data)
    csv_file.close()    

    print("Đã tổng hợp ngày", formatted_now)
    return data_sumary
            

#
def save_check_to_csv(id: str, checkin=True):
    """
        Save checkin and checkout to csv file
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
    checkin_path = os.path.join(day_path, "check_in")
    checkout_path = os.path.join(day_path, "check_out")

    if os.path.exists(year_path) == False:
        os.mkdir(year_path)
    if os.path.exists(month_path) == False:
        os.mkdir(month_path)
    if os.path.exists(day_path) == False:
        os.mkdir(day_path)
    if os.path.exists(checkin_path) == False:
        os.mkdir(checkin_path)
    if os.path.exists(checkout_path) == False:
        os.mkdir(checkout_path)

    
    if checkin:
        data = {"ID": id, 'Time check in': formatted_now}
        fieldnames = ['ID', 'Time check in']
        data_csv = []
        ckin_session_path = os.path.join(checkin_path, "checkin.csv")

        if os.path.exists(ckin_session_path):
            with open(ckin_session_path, 'r') as csv_file: #getdata
                reader = csv.DictReader(csv_file, fieldnames=fieldnames)
                data_csv = [row for row in reader]
            csv_file.close()

        for row in data_csv: #check isChekin or not
            if row["ID"] == data["ID"]:
                # print("ID", id ,"đã check in rồi!")
                return
            
        with open(ckin_session_path, "a", newline='') as csv_file: #if have checkin yet, check in
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(data)
        csv_file.close()
    else:
        data = {"ID": id, 'Time check out': formatted_now}
        fieldnames = ['ID', 'Time check out']
        data_csv = []
        ckout_session_path = os.path.join(checkout_path, "checkout.csv")

        if os.path.exists(ckout_session_path):
            with open(ckout_session_path, 'r') as csv_file: #getdata
                reader = csv.DictReader(csv_file, fieldnames=fieldnames)
                data_csv = [row for row in reader]
            csv_file.close()

        for row in data_csv: #check isChekin or not
            if row["ID"] == data["ID"]:
                # print("ID", id ,"đã check out rồi!")
                return
            
        with open(ckout_session_path, "a", newline='') as csv_file: #if have checkin yet, check in
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(data)
        csv_file.close()

if __name__ == "__main__":
   data = sumary_day()
   print(data)

    