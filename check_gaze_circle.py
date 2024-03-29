# -*- coding: UTF-8 -*-
import cv2
import sys
import time
import imutils

import show_gaze as vs
import matplotlib.pyplot as plt
import ellipse_util as uu
import argparse
import numpy as np
import ShapeDetector as iris
import math


def cal_gaze(ell_A, ell_B, ell_C, ell_D, ell_E, ell_F):
    # print('---', ell_A * aa * aa + ell_B * aa * bb + ell_C * bb * bb + ell_D * aa + ell_E * bb + ell_F)
    Z = np.array([[ell_A, ell_B / 2.0, -ell_D / (2.0)],
                  [ell_B / 2.0, ell_C, -ell_E / (2.0)],
                  [-ell_D / (2.0), -ell_E / (2.0), ell_F]])

    eig_vals, eig_vecs = np.linalg.eig(Z)
    # print(eig_vals, eig_vecs)

    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    # print(eig_vals, eig_vecs)

    g = np.sqrt((eig_vals[1] - eig_vals[2]) / (eig_vals[0] - eig_vals[2]))
    h = np.sqrt((eig_vals[0] - eig_vals[1]) / (eig_vals[0] - eig_vals[2]))

    tt = np.array([h, 0, g]) * np.array([[1, 0, -1], [1, 0, 1], [-1, 0, -1], [-1, 0, 1]])
    # tt = np.array([[h, 0, -g], [h, 0, g], [-h, 0, -g], [-h, 0, g]])

    R = eig_vecs
    # print(tt,tt.T)
    normals = R.dot(tt.T)
    return normals.T


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to the input video")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = frame
    # cv2.imshow('frame',frame)
    # continue
    resized = frame
    ratio = 1

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)

    imgray = cv2.Canny(image, 600, 100, 3)  # Canny边缘检测，参数可更改
    # cv2.imshow("0",imgray)
    imgray = imgray[480:,:]
    ss = image[480:,:]
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,

                                              cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    circle = []
    for cnt in contours:
        if 100 > len(cnt) > 50:
            S1 = cv2.contourArea(cnt)
            ell = cv2.fitEllipse(cnt)
            S2 = math.pi * ell[1][0] * ell[1][1]
            if (S1 / S2) > 0.2:  # 面积比例，可以更改，根据数据集。。。
                ss = cv2.ellipse(ss, ell, (0, 255, 0), 2)
                circle.append(cnt)
    # cv2.imshow("0", ss)

    normals = np.array([[0, 0, 0]])

    contour = []
    # loop over the contours
    for c in circle:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        contour = c[:, 0, :]

        for con in contour:
            cv2.circle(resized, (con[0], (con[1] + 480)), 1, (255, 0, 255), -1)
            # cv2.imshow('outline', image)

        # TODO:undistort
        # contour = contour - [image.shape[0] / 2, image.shape[1] / 2]
        fx = 883.3836
        fy = 879.1104
        cx = 282.9217
        cy = 495.5409

        u = (contour[:,1] - cy + 480) / fy
        v = (contour[:,0] - cx) / fx

        ell_A, ell_B, ell_C, ell_D, ell_E, ell_F, _ = uu.Ellipse(v, u)
        print(ell_A, ell_B, ell_C, ell_D, ell_E, ell_F)

        # evalute ellipse fitting
        xx = _[0]
        yy = _[1]
        zz = _[2]

        plt.plot(np.array(v), np.array(u), color='pink')
        plt.plot(xx, yy, color='red')
        plt.plot(xx, zz, color='green')

        val = cal_gaze(ell_A, ell_B, ell_C, ell_D, ell_E, ell_F)
        normals = np.concatenate((normals, val))

    plt.title('Ellipse fittig evaluation')
    plt.ylabel('contour points in height')
    plt.show()
    # show gaze
    vs.show_gaze(resized, normals)
    print normals

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
