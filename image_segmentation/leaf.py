import cv2
import numpy as np 

def viewImage(image):
    cv2.namedWindow('Dispaly', cv2.WINDOW_NORMAL)
    cv2.imshow('Dispaly', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

green = np.uint8([[[0, 255, 0]]])
green_hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print(green_hsv)

image = cv2.imread('D:\Code\image segmentation\leaf.png')
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
viewImage(hsv_img)

green_low = np.array([45, 100, 50])
green_high = np.array([75, 255, 255])
curr_mask = cv2.inRange(hsv_img, green_low, green_high)
hsv_img[curr_mask > 0] = ([75, 255, 200])
viewImage(hsv_img)
##将HSV图片灰化才能应用轮廓

RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_again,cv2.COLOR_RGB2GRAY)
viewImage(gray)
ret, threshold = cv2.threshold(gray, 90, 255, 0)
viewImage(threshold)

binary, contours, hirerchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
viewImage(image)

def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while(i < total_contours):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        i += 1
    
    return largest_area, largest_contour_index

cnt = contours[13]
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

largest_area, largest_contour_index = findGreatesContour(contours)

print(largest_area)
print(largest_contour_index)

print(len(contours))

print(cX)
print(cY)