import cv2
import numpy as np
import math
from scipy import optimize
from matplotlib import pyplot as plt

framecount = 0

#the right and left edge of central column
x_colr = 610
x_coll = 446

def nothing(*argv):
    pass

def main():
    cap = cv2.VideoCapture('180907002cap.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output3.avi', fourcc, 25.0, (1280, 800))

    R = videoprocess(cap,out)
    x = np.linspace(1,framecount,framecount)

    i=0
    while(i<len(R)):
        if R[i]==0 or R[i]>500:
            R.pop(R[i])
            x=np.delete(x,i)
        else:
            i+=1

    plt.plot(x,R)
    plt.xlabel('frame')
    plt.ylabel('r')
    plt.ylim(0,300)
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

def videoprocess(video,writevieo):
    rc=[]
    while(video.isOpened()):
        try:
            ret,frame = video.read()
            xc,yc,r,frame = imageprocess(frame)
            rc.append(r)
            global framecount
            framecount +=1
            print(framecount)

            if ret == True:
                writevieo.write(frame)
        except:
            break
    return rc

def imageprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),sigmaX=2,sigmaY=2)
    gray = histEqual(gray)

    ## /thresholding and finding contours/
    mean = np.mean(gray)
    th = 2.0 - framecount/300
    ths = th*mean
    ret, thresh = cv2.threshold(gray, ths, 255, 0)
    cntimg,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    arcLenth = []
    for i in contours:
        arcLenth.append(i.size)
    Index=arcLenth.index(max(arcLenth))
    cnt = contours[Index]

    rightEdge = findRightEdge(cnt,img)
    leftEdge = findLeftEdge(cnt,img)
    xc_r, yc_r, R_r = circlefit(rightEdge)
    xc_l, yc_l, R_l = circlefit(leftEdge)
    cv2.circle(img, (xc_r, yc_r), R_r, (255, 255, 255), 2)
    cv2.circle(img, (xc_l, yc_l), R_l, (255, 255, 255), 2)
    # cv2.imshow('img', img)
    global R_0
    if R_r<R_l:
        return xc_r,yc_r,R_r,img
    else:
        return xc_l,yc_l,R_l,img

## /histogram equalization/
def histEqual(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    for i in range(1):
       im = clahe.apply(im)
    # im = cv2.GaussianBlur(im,(9,9),sigmaX=2,sigmaY=2)
    return im

## /find right edge/
def findRightEdge(cnt,img):
    edge = np.array(cnt)

    #capture the right half contour
    i = 0
    while i < int(edge.size / 2):
        if edge[i][0][0] < x_colr:
            edge = np.delete(edge, i, 0)
        else:
            i += 1

    ## /contour ellipse fit/
    try:
        ellipse = cv2.fitEllipse(edge)  # ellipse((x0,y0),(2b,2a),angle),while angle=90,a is horizontal
        # cv2.ellipse(img, ellipse, (255, 0, 0), 1)

        x0, y0, b, a = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)
        xm = 0  # the right edge
        for x in range(x0, x0 + a - 1):
            k = -((x - x0) * b * b) / (b * math.sqrt(1 - (x - x0) * (x - x0) / (a * a)) * a * a)
            if k < -0.8:
                xm = x
                break

        i = 0
        while i < int(edge.size / 2):
            if edge[i][0][0] < xm:
                edge = np.delete(edge, i, 0)
            else:
                i += 1
    except:
        print('rightedge not found')
        return np.zeros_like(cnt)

    return edge

def findLeftEdge(cnt,img):
    edge = np.array(cnt)

    # capture the left half contour
    i = 0
    while i < int(edge.size / 2):
        if edge[i][0][0] > x_coll:
            edge = np.delete(edge, i, 0)
        else:
            i += 1

    ## /contour ellipse fit/
    try:
        ellipse = cv2.fitEllipse(edge)  # ellipse((x0,y0),(2b,2a),angle),while angle=90,a is horizontal
        # cv2.ellipse(img, ellipse, (255, 0, 0), 1)

        x0, y0, b, a = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)
        xm = 0  # the right edge
        for x in range(x0 - a + 1, x0):
            k = -((x - x0) * b * b) / (b * math.sqrt(1 - (x - x0) * (x - x0) / (a * a)) * a * a)
            if k < 0.8:
                xm = x
                break

        edge = np.array(cnt)
        i = 0
        while i < int(edge.size / 2):
            if edge[i][0][0] > xm:
                edge = np.delete(edge, i, 0)
            else:
                i += 1
    except:
        print('leftedge not found')
        return np.zeros_like(cnt)

    return edge

## /circle fit/
def circlefit(edge):
    x, y = edge[:, 0, 0], edge[:, 0, 1]
    x_m, y_m = np.mean(x), np.mean(y)

    def calc_R(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - np.mean(Ri)

    center_estimate = x_m, y_m
    try:
        center_2, ier = optimize.leastsq(f_2, center_estimate)
    except:
        print('fitting error!')
        return 0,0,0

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = np.mean(Ri_2)
    return int(xc_2), int(yc_2), int(R_2)


if __name__ == '__main__':
    main()
