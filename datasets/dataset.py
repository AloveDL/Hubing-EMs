# create date: 2021/09/06
import xlrd
import os
from utils.LBP import load_video,lbp_sip


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
            w.write(str(label_int_map[i]) + '\n')


def cap_lbp_feature(root):
    lbp_feature = []
    for subdir in os.listdir(root):
        new_path = root + "\\" + subdir
        for video_file in os.listdir(new_path):
            print(video_file)
            video_data = load_video(new_path + "\\" + video_file)
            lbp = lbp_sip(video_data['video_tensor'])
            save_lbp_feature(lbp)
    return 1


def save_lbp_feature(data):
    with open("data/CASME2_lbp_sip_data.txt", 'a') as w:
        for i in data:
            w.write(str(i) + '\n')


def load_lbp_feature(path):
    feature = []
    with open(path, 'r') as r:
        data = r.read().split('\n')
        # print(data)
        data = [float(i) for i in data[:-1]]
        feature.append(data)
    result = []
    length = len(feature[0])
    i = 500

    while i <= length:
        result.append(feature[0][i - 500:i])
        i = i + 500
    return filter_data(result)


def load_label(path):
    label = []
    with open(path, 'r') as r:
        row_data = r.readline()
        while row_data != '':
            label.append(int(row_data[:-1]))
            row_data = r.readline()


    return label

def filter_data(data):
    # data = np.array(data)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if str(data[i][j]) == 'nan':
                data[i][j] = 0.0
    return data
    # print(df)


if __name__ == '__main__':
    root = "C:\\Users\\27635\\Desktop\\公司实习\\data\\CASME2_Compressed video\\CASME2_compressed"
    lbp_feature = cap_lbp_feature(root)
    # print(load_lbp_feature("data/CASME2_lbp_xy_data.txt"))
    # save_label()
    # print(load_label("data/CASME2_label.txt"))

    # filter_data(load_lbp_feature("data/CASME2_lbp_xy_data.txt"))