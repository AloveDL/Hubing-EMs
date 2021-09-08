# create date: 2021/09/06
import xlrd
import numpy as np
import os
from utils.LBP import lbp_top, load_video


def save_label():
    workbook = xlrd.open_workbook('data/CASME2-coding-20140508.xlsx')
    sheet = workbook.sheets()[0]
    label_int_map = {
        "happiness": 1,
        "others": 2,
        "disgust": 3,
        "repression": 4,
        "surprise": 5,
        "fear": 6,
        "sadness": 7,
    }
    with open("data/CASME2_label.txt", 'w') as w:
        for i in sheet.col_values(8)[1:]:
            w.write(str(label_int_map[i])+'\n')


def cap_lbp_feature(root):
    lbp_feature = []
    for subdir in os.listdir(root):
        new_path = root+"\\"+subdir
        for video_file in os.listdir(new_path):
            print(video_file)
            video_data = load_video(new_path+"\\"+video_file)
            lbp = lbp_top(video_data['video_tensor'])
            save_lbp_feature(lbp)
    return 1


def save_lbp_feature(data):
    np.savetxt('data/CASME2_lbp_xy_data.txt', data, fmt="%f")


def load_lbp_feature(path):
    return np.loadtxt(path, delimiter=',')


def load_label(path):
    label = []
    with open(path, 'r') as r:
        row_data = r.readline()
        label.append(row_data)
    return label


if __name__ == '__main__':
    root = "C:\\Users\\27635\\Desktop\\公司实习\\data\\CASME2_Compressed video\\CASME2_compressed"
    lbp_feature = cap_lbp_feature(root)

    # save_label()