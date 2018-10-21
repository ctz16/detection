import cv2
import numpy as np
import math
from scipy import optimize
from skimage import measure
from matplotlib import pyplot as plt

img = cv2.imread('capture3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

def nothing(*argv):
    pass

## /GaussianBlur/
gray = cv2.GaussianBlur(gray,(9,9),sigmaX=2,sigmaY=2)
# cv2.imshow('gray_blur',gray)

## /histogram equalization/
def histEqual(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    for i in range(1):
       im = clahe.apply(im)
    # im = cv2.GaussianBlur(im,(9,9),sigmaX=2,sigmaY=2)
    return im
gray = histEqual(gray)
cv2.imshow('gray_equ',gray)
mean = np.mean(gray)
print(mean)
th = 2.0

# ## /normal thresholding debug/
# cv2.namedWindow('thresh')
# cv2.createTrackbar('ths','thresh',0,80,nothing)
#
# while(True):
#     th = cv2.getTrackbarPos('ths','thresh')
#     ret, thresh = cv2.threshold(gray, th, 255, 0)
#     cv2.imshow('thresh', thresh)
#     k = cv2.waitKey(1)&0xFF
#     if k==27:
#         break

# ## /Canny detection debug /
# cv2.namedWindow('canny')
# cv2.createTrackbar('ths1','canny',1,80,nothing)
# cv2.createTrackbar('ths2','canny',1,80,nothing)
# while(True):
#     ths1 = cv2.getTrackbarPos('ths1', 'canny')
#     ths2 = cv2.getTrackbarPos('ths2', 'canny')
#     canny = cv2.Canny(gray,ths1,ths2,)
#     cv2.imshow('canny', canny)
#     k = cv2.waitKey(1)&0xFF
#     if k==27:
#         break

## /thresholding and finding contours/

ths = th*mean
ret, thresh = cv2.threshold(gray, ths, 255, 0)
cntimg,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
arcLenth = []
for i in contours:
    arcLenth.append(i.size)
Index=arcLenth.index(max(arcLenth))
cnt = contours[Index]

cv2.imshow('thresh',thresh)
cv2.drawContours(img,contours,Index,(0,255,0),3)
cv2.imshow('cnt',img)

while(True):
    k = cv2.waitKey(1)&0xFF
    if k==27:
        break


## /contour ellipse fit/
ellipse = cv2.fitEllipse(cnt) #ellipse((x0,y0),(2b,2a),angle),while angle=90,a is horizontal
cv2.ellipse(img,ellipse,(255,0,0),1)


## /find right edge/
def findRightEdge(ellipse,cnt):
    x0,y0,b,a=int(ellipse[0][0]),int(ellipse[0][1]),int(ellipse[1][0]/2),int(ellipse[1][1]/2)
    xm = 0 #the right edge
    for x in range(x0,x0+a-1):
        k = -((x-x0)*b*b)/(b*math.sqrt(1-(x-x0)*(x-x0)/(a*a))*a*a)
        if k<-0.8: #angle = 90
            xm = x
            break

    edge = np.array(cnt)
    i=0
    while i<int(edge.size/2):
        if edge[i][0][0]<xm:
            edge = np.delete(edge,i,0)
        else:
            i+=1
    return edge


def findLeftEdge(ellipse, cnt):
    x0, y0, b, a = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)
    xm = 0  # the right edge
    for x in range(x0-a+1, x0):
        k = -((x - x0) * b * b) / (b * math.sqrt(1 - (x - x0) * (x - x0) / (a * a)) * a * a)
        if k < 0.8:  # angle = 90
            xm = x
            break

    edge = np.array(cnt)
    i = 0
    while i < int(edge.size / 2):
        if edge[i][0][0] > xm:
            edge = np.delete(edge, i, 0)
        else:
            i += 1
    return edge

## /circle fit/
def circlefit(edge):
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
    R_2        = np.mean(Ri_2)
    residu_2   = sum((Ri_2 - R_2)**2)
    return int(xc_2),int(yc_2),int(R_2)


rightEdge = findRightEdge(ellipse,cnt)
leftEdge = findLeftEdge(ellipse,cnt)
xc_r,yc_r,R_r = circlefit(rightEdge)
xc_l,yc_l,R_l = circlefit(leftEdge)
cv2.circle(img,(xc_r,yc_r),R_r,(255,255,255),2)
cv2.circle(img,(xc_l,yc_l),R_l,(255,255,255),2)
cv2.imshow('img',img)


## /mouse callback/
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,255), thickness = 1)
        cv2.imshow("contour0",img)

cv2.namedWindow('contour0')
cv2.setMouseCallback('contour0', on_EVENT_LBUTTONDOWN)
cv2.imshow('contour0', img)

while(True):
    k = cv2.waitKey(1)&0xFF
    if k==27:
        break

# ## /filling /
# ROI = np.zeros(img.shape,np.uint8)
# cntimg,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]
# cv2.drawContours(ROI,contours,0,(255,255,255),-1)
# cv2.imshow('contour',ROI)
# ROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
# img_cut = cv2.bitwise_and(ROI,gray)
# cv2.imshow('cut', img_cut)


# while(True):
#     k = cv2.waitKey(1)&0xFF
#     if k==27:
#         break
#


#
#
# ## /dilation and erode /
# # kernel1 = np.ones((5,5),np.uint8)
# # kernel2 = np.ones((3,3),np.uint8)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# dilation = cv2.dilate(canny,kernel1,iterations = 5)
# dilation = cv2.erode(dilation, kernel2, iterations = 5)
# # dilation = cv2.dilate(dilation,kernel1, iterations = 1)
# cv2.imshow('dilation2',dilation)

# ## /draw hist/
# histr = cv2.calcHist([gray],[0],None,[256],[0,256])
# plt.plot(histr)
# plt.xlim([0,40])
# plt.show()


# ## /adaptive thresholding debug/
# cv2.namedWindow('adaptive_thresh')
# cv2.createTrackbar('blocksize','adaptive_thresh',0,200,nothing)
# # cv2.createTrackbar('C','adaptive_thresh',0,20,nothing)
#
# while(True):
#     blocksize = cv2.getTrackbarPos('blocksize','adaptive_thresh')
#     # C = cv2.getTrackbarPos('C','adaptive_thresh')
#     th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                                cv2.THRESH_BINARY, 2*blocksize+3, 5)
#     cv2.imshow('adaptive_thresh', th)
#     k = cv2.waitKey(1)&0xFF
#     if k==27:
#         break


# ## /Hough circles/
# cv2.namedWindow('detected_circles')

# cv2.createTrackbar('param1','detected_circles',35,50,nothing)
# cv2.createTrackbar('param2','detected_circles',30,50,nothing)
#
# while(True):
#     P1 = cv2.getTrackbarPos('param1','detected_circles')
#     P2 = cv2.getTrackbarPos('param2', 'detected_circles')
#     circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,500,
#                                 param1=P1,param2=P2,minRadius=50,maxRadius=300)
#
#     circles = np.uint16(np.around(circles))
#     for i in circles[0,:]:
#         # draw the outer circle
#         cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#         # draw the center of the circle
#         cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
#
#     cv2.imshow('detected_circles',img)
#     k = cv2.waitKey(1)&0xFF
#     if k==27:
#         break

# circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,500,
#                             param1=35,param2=30,minRadius=100,maxRadius=500)
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
#
# cv2.imshow('detected circles',img)


## /blobs detection/
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
# params.minThreshold = 0
# params.maxThreshold = 150
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 100
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.5
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.5
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.5
#
# # Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3:
#     detector = cv2.SimpleBlobDetector(params)
# else:
#     detector = cv2.SimpleBlobDetector_create(params)
#
# # Detect blobs.
# keypoints = detector.detect(gray)
#
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# # the size of the circle corresponds to the size of blob
#
# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # Show blobs
# cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()

# def findLeft(im):
#     h,w = im.shape[:-1]
#     for i in range(w):
#         for j in range(h):
#             if im[i,j] != 0:
#                 return [i,j]
#     return 0
#
# def findStart(im,loc):
#     h, w = im.shape[:-1]
#     for i in range(w-loc[1]):
#         if im[loc[0],loc[1]+i+1] == 0 and im[loc[0],loc[1]+i] != 0:
#             return [loc[0],loc[1]+i]
#     return 0
#
# def findEdgePoint(im,loc,oldloc):
#     l,h = loc[0],loc[1]
#     grid = ((l-1,h-1),(l,h-1),(l+1,h-1), \
#             (l+1, h),(l+1, h+1),(l, h+1), \
#             (l-1, h+1),(l-1,h))
#     edgepoint=[]
#     for i in range(8):
#         if im[grid[i]]!=im[grid[i+1]]:
#             if im[grid[i]]!=0:
#                 edgepoint.append(grid[i])
#             else:
#                 edgepoint.append(grid[i+1])
#     if edgepoint[0] == oldloc:
#         return edgepoint[1]
#     else:
#         return edgepoint[0]
#
# def findEdge(im,ed,startloc):
#     oldpoint = startloc


# leftloc = findLeft(dilation)
# startloc = findStart(dilation,leftloc)
# edge = np.zeros(img.shape,np.uint8)
# edge = findLeftEdge(dilation,edge,startloc)