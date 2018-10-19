import cv2
import numpy as np
import math
from scipy import optimize
from matplotlib import pyplot as plt

# img = cv2.imread('capture3.jpg')

def main():
    cap = cv2.VideoCapture('180907002cap.mp4')
    R = videoprocess(cap)
    plt.plot(R)

def nothing(*argv):
    pass

def videoprocess(video):
    rc=[]
    while(video.isOpened()):
        ret,frame = video.read()
        xc,yc,r = imageprocess(frame)
        rc.append(r)
    return rc

## /histogram equalization/
def histEqual(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    for i in range(1):
       im = clahe.apply(im)
    im = cv2.GaussianBlur(im,(9,9),sigmaX=2,sigmaY=2)
    return im

def imageprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),sigmaX=2,sigmaY=2)
    gray = histEqual(gray)

    ## /thresholding and finding contours/
    mean = np.mean(gray)
    print(mean)
    th = 1.9
    ths = th*mean
    ret, thresh = cv2.threshold(gray, ths, 255, 0)
    cntimg,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    arcLenth = []
    for i in contours:
        arcLenth.append(i.size)
    Index=arcLenth.index(max(arcLenth))
    cnt = contours[Index]

    ## /contour ellipse fit/
    ellipse = cv2.fitEllipse(cnt) #ellipse((x0,y0),(2b,2a),angle),while angle=90,a is horizontal
    cv2.ellipse(img,ellipse,(255,0,0),1)
    x0,y0,b,a=int(ellipse[0][0]),int(ellipse[0][1]),int(ellipse[1][0]/2),int(ellipse[1][1]/2)

    xm = 0 #the right edge
    for x in range(x0-a+1,x0):
        k = -((x-x0)*b*b)/(b*math.sqrt(1-(x-x0)*(x-x0)/(a*a))*a*a)
        if k<0.8: #angle = 90
            xm = x
            break

    edge = np.array(cnt)
    i=0
    while i<int(edge.size/2):
        if edge[i][0][0]>xm:
            edge = np.delete(edge,i,0)
        else:
            i+=1

    ## /circle fit/
    x,y=edge[:,0,0],edge[:,0,1]
    x_m,y_m=np.mean(x),np.mean(y)

    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - np.mean(Ri)

    center_estimate = x_m,y_m
    center_2, ier = optimize.leastsq(f_2,center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = int(np.mean(Ri_2))
    # residu_2   = sum((Ri_2 - R_2)**2)

    xc_2,yc_2 = int(xc_2),int(yc_2)
    # cv2.circle(img,(xc_2,yc_2),R_2,(255,255,255),2)
    return xc_2,yc_2,R_2

if __name__ == '__main__':
    main()
