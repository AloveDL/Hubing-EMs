# create date: 2021/09/06

import cv2
import logging
import numpy as np
import math
from tqdm import tqdm


def load_video(video_filename):
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_tensor = np.zeros((frame_count, width, height), dtype='float')
    video_tensor_shape = video_tensor.shape
    video_frame_index = 0
    # print(video_tensor_shape, frame_count)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video_tensor[video_frame_index] = np.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            video_frame_index += 1
        else:
            break

    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "video_tensor": video_tensor,
        "shape": video_tensor_shape,
    }


def lbp_xyot(data, x_radius=1, y_radius=1, t_radius=1, neighbour_points=None, time_length=1, border_length=1):
    if neighbour_points is None:
        neighbour_points = [8, 8]
    length, width, height = data.shape
    xt_neighbour_points = neighbour_points[0]
    yt_neighbour_points = neighbour_points[1]

    n_dim = 2 ** yt_neighbour_points
    histogram = np.zeros((2, n_dim), float)

    for t in tqdm(range(time_length, length - time_length)):
        for yc in range(border_length, height - border_length):
            for xc in range(border_length, width - border_length):
                center_val = data[t, xc, yc]
                basic_lbp = 0
                FeaBin = 0
                for p in range(0, xt_neighbour_points):
                    X = int(xc + x_radius * math.cos((2 * math.pi * p) / xt_neighbour_points) + 0.5)
                    Z = int(t + t_radius * math.sin((2 * math.pi * p) / xt_neighbour_points) + 0.5)

                    CurrentVal = data[Z, X, yc]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin
                    FeaBin += 1
                histogram[0, basic_lbp] = histogram[0, basic_lbp] + 1
                basic_lbp = 0
                FeaBin = 0
                for p in range(0, yt_neighbour_points):
                    Y = int(yc - y_radius * math.cos((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                    Z = int(t + t_radius * math.sin((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                    CurrentVal = data[Z, xc, Y]
                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin
                    FeaBin += 1
                histogram[1, basic_lbp] = histogram[1, basic_lbp] + 1
    # for j in range(0, 3):
    #     histogram[j, :] = (histogram[j, :]*1.0)/sum(histogram[j, :])

    histogram = histogram.flatten()

    return standardization(histogram)


def lbp_xot(data, x_radius=1, t_radius=1, neighbour_points=None, time_length=1, border_length=1):
    if neighbour_points is None:
        neighbour_points = 8
    length, width, height = data.shape
    xt_neighbour_points = neighbour_points

    n_dim = 2 ** xt_neighbour_points
    histogram = np.zeros((1, n_dim), float)
    for t in tqdm(range(time_length, length - time_length)):
        for yc in range(border_length, height - border_length):
            for xc in range(border_length, width - border_length):
                center_val = data[t, xc, yc]
                basic_lbp = 0
                FeaBin = 0
                for p in range(0, xt_neighbour_points):
                    X = int(xc + x_radius * math.cos((2 * math.pi * p) / xt_neighbour_points) + 0.5)
                    Z = int(t + t_radius * math.sin((2 * math.pi * p) / xt_neighbour_points) + 0.5)

                    CurrentVal = data[Z, X, yc]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin
                    FeaBin += 1

                histogram[1, basic_lbp] = histogram[1, basic_lbp] + 1
    # for j in range(0, 3):
    #     histogram[j, :] = (histogram[j, :]*1.0)/sum(histogram[j, :])
    histogram = histogram.flatten()
    return standardization(histogram)


def lbp_yot(data, y_radius=1, t_radius=1, neighbour_points=None, time_length=1, border_length=1):
    if neighbour_points is None:
        neighbour_points = 8
    length, width, height = data.shape
    yt_neighbour_points = neighbour_points

    n_dim = 2 ** yt_neighbour_points
    histogram = np.zeros((1, n_dim), float)
    for t in tqdm(range(time_length, length - time_length)):
        for yc in range(border_length, height - border_length):
            for xc in range(border_length, width - border_length):
                center_val = data[t, xc, yc]
                basic_lbp = 0
                FeaBin = 0
                for p in range(0, yt_neighbour_points):
                    Y = int(yc - y_radius * math.cos((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                    Z = int(t + t_radius * math.sin((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                    CurrentVal = data[Z, xc, Y]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin
                    FeaBin += 1

                histogram[2, basic_lbp] = histogram[2, basic_lbp] + 1

    # for j in range(0, 3):
    #     histogram[j, :] = (histogram[j, :]*1.0)/sum(histogram[j, :])
    histogram = histogram.flatten()
    return standardization(histogram)


def lbp_top(data, x_radius=1, y_radius=1, t_radius=1, neighbour_points=None, time_length=1, border_length=1):
    if neighbour_points is None:
        neighbour_points = [8, 8, 8]
    length, width, height = data.shape
    xy_neighbour_points = neighbour_points[0]
    xt_neighbour_points = neighbour_points[1]
    yt_neighbour_points = neighbour_points[2]

    n_dim = 2 ** yt_neighbour_points
    histogram = np.zeros((3, n_dim), float)

    for t in tqdm(range(time_length, length - time_length)):
        for yc in range(border_length, height - border_length):
            for xc in range(border_length, width - border_length):
                center_val = data[t, xc, yc]

                basic_lbp = 0
                FeaBin = 0

                for p in range(0, xy_neighbour_points):
                    X = int(xc + x_radius * math.cos((2 * math.pi * p) / xy_neighbour_points) + 0.5)
                    Y = int(yc - y_radius * math.sin((2 * math.pi * p) / xy_neighbour_points) + 0.5)

                    CurrentVal = data[t, X, Y]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin

                    FeaBin += 1
                histogram[0, basic_lbp] = histogram[0, basic_lbp] + 1

                basic_lbp = 0
                FeaBin = 0

                for p in range(0, xt_neighbour_points):
                    X = int(xc + x_radius * math.cos((2 * math.pi * p) / xt_neighbour_points) + 0.5)
                    Z = int(t + t_radius * math.sin((2 * math.pi * p) / xt_neighbour_points) + 0.5)

                    CurrentVal = data[Z, X, yc]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin
                    FeaBin += 1

                histogram[1, basic_lbp] = histogram[1, basic_lbp] + 1

                basic_lbp = 0
                FeaBin = 0

                for p in range(0, yt_neighbour_points):
                    Y = int(yc - y_radius * math.cos((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                    Z = int(t + t_radius * math.sin((2 * math.pi * p) / yt_neighbour_points) + 0.5)

                    CurrentVal = data[Z, xc, Y]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin
                    FeaBin += 1

                histogram[2, basic_lbp] = histogram[2, basic_lbp] + 1
    # for j in range(0, 3):
    #     histogram[j, :] = (histogram[j, :]*1.0)/sum(histogram[j, :])
    histogram = uniform_bit8_lbp(histogram)
    histogram = histogram.flatten()

    return standardization(histogram)


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def lbp_xy(data, x_radius=1, y_radius=1, neighbour_points=None, time_length=1, border_length=1):
    if neighbour_points is None:
        neighbour_points = 8
    length, width, height = data.shape
    xy_neighbour_points = neighbour_points

    n_dim = 2 ** xy_neighbour_points
    histogram = np.zeros((1, n_dim), float)
    for t in tqdm(range(time_length, length - time_length)):
        for yc in range(border_length, height - border_length):
            for xc in range(border_length, width - border_length):
                center_val = data[t, xc, yc]

                basic_lbp = 0
                FeaBin = 0

                for p in range(0, xy_neighbour_points):
                    X = int(xc + x_radius * math.cos((2 * math.pi * p) / xy_neighbour_points) + 0.5)
                    Y = int(yc - y_radius * math.sin((2 * math.pi * p) / xy_neighbour_points) + 0.5)

                    CurrentVal = data[t, X, Y]

                    if CurrentVal >= center_val:
                        basic_lbp += 2 ** FeaBin

                    FeaBin += 1
                histogram[0, basic_lbp] = histogram[0, basic_lbp] + 1

    # for j in range(0, 3):
    #     histogram[j, :] = (histogram[j, :]*1.0)/sum(histogram[j, :])
    histogram = histogram.flatten()
    return standardization(histogram)


def flbp_top(data, x_radius=1, y_radius=1, t_radius=1, neighbour_points=None):
    if neighbour_points is None:
        neighbour_points = [1, 1, 1]
    length, width, height = data.shape
    xy_neighbour_points = neighbour_points[0]
    xt_neighbour_points = neighbour_points[1]
    yt_neighbour_points = neighbour_points[2]

    hist_xy = np.zeros((1, 2 ** xy_neighbour_points), float)
    hist_xt = np.zeros((1, 2 ** xt_neighbour_points), float)
    hist_yt = np.zeros((1, 2 ** yt_neighbour_points), float)

    xy_tensor = np.zeros((height, 2 * t_radius + width * length), float)
    xt_tensor = np.zeros((length, 2 * y_radius + height * width), float)
    yt_tensor = np.zeros((width, 2 * x_radius + length * height), float)

    xy_tensor[:, t_radius:-t_radius] = data.reshape(data.shape[2], -1)
    xt_tensor[:, y_radius:-y_radius] = data.reshape(data.shape[0], -1)
    yt_tensor[:, x_radius:-x_radius] = data.reshape(data.shape[1], -1)
    for xc in tqdm(range(x_radius, xy_tensor.shape[0] - x_radius)):
        for yc in range(y_radius, xy_tensor.shape[1] - y_radius):
            center_val = xy_tensor[xc, yc]
            basic_lbp = 0
            FeaBin = 0
            for p in range(0, xy_neighbour_points):
                X = int(xc + x_radius * math.cos((2 * math.pi * p) / xy_neighbour_points) + 0.5)
                Y = int(yc - y_radius * math.sin((2 * math.pi * p) / xy_neighbour_points) + 0.5)
                CurrentVal = xy_tensor[X, Y]
                if CurrentVal >= center_val:
                    basic_lbp += 2 ** FeaBin
                FeaBin += 1
            hist_xy[0, basic_lbp] += 1
    print(hist_xy, hist_xy.shape)
    for xc in range(x_radius, xt_tensor.shape[0] - x_radius):
        for tc in range(t_radius, xt_tensor.shape[1] - t_radius):
            center_val = xt_tensor[xc, tc]
            basic_lbp = 0
            FeaBin = 0
            for p in range(0, xt_neighbour_points):
                X = int(xc + x_radius * math.cos((2 * math.pi * p) / xt_neighbour_points) + 0.5)
                T = int(tc - t_radius * math.sin((2 * math.pi * p) / xt_neighbour_points) + 0.5)
                CurrentVal = xt_tensor[X, T]
                if CurrentVal >= center_val:
                    basic_lbp += 2 ** FeaBin
                FeaBin += 1
            hist_xt[0, basic_lbp] += 1

    for yc in range(y_radius, yt_tensor.shape[0] - y_radius):
        for tc in range(t_radius, yt_tensor.shape[1] - y_radius):
            center_val = yt_tensor[yc, tc]
            basic_lbp = 0
            FeaBin = 0
            for p in range(0, yt_neighbour_points):
                Y = int(yc + y_radius * math.cos((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                T = int(tc - t_radius * math.sin((2 * math.pi * p) / yt_neighbour_points) + 0.5)
                CurrentVal = yt_tensor[Y, T]
                if CurrentVal >= center_val:
                    basic_lbp += 2 ** FeaBin
                FeaBin += 1
            hist_yt[0, basic_lbp] += 1
    return standardization(np.hstack((np.hstack((hist_xy, hist_xt)), hist_yt))[0])


def uniform_bit8_lbp(histogram):
    map = {
        "0": 0, "1": 1, "3": 3, "4": 4, "6": 5, "7": 6, "8": 7, "12": 8, "14": 9,
        "15": 10, "16": 11, "24": 12, "28": 13, "30": 14, "31": 15, "32": 16,
        "48": 17, "56": 18, "60": 19, "62": 20, "63": 21, "64": 22, "96": 23,
        "112": 24, "120": 25, "124": 26, "126": 27, "127": 28, "128": 29,
        "129": 30, "131": 31, "135": 32, "143": 33, "159": 34, "191": 35,
        "192": 36, "193": 37, "195": 38, "199": 39, "207": 40, "223": 41,
        "224": 42, "225": 43, "227": 44, "231": 45, "239": 46, "240": 47,
        "241": 48, "243": 49, "247": 50, "248": 51, "249": 52, "251": 53,
        "252": 54, "253": 55, "254": 56, "255": 57
    }
    new_histogram = np.zeros((histogram.shape[0], 59), float)
    for index, n_hist_data in enumerate(histogram):
        for i, data in enumerate(n_hist_data):
            if str(i) in map.keys():
                new_histogram[index, map[str(i)]] += data
            else:
                new_histogram[index, -1] += data
    return new_histogram


if __name__ == '__main__':
    print(lbp_top(load_video('video/EP02_01f.avi')['video_tensor']))
