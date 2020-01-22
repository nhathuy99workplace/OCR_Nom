from __future__ import division
from __future__ import print_function

import cv2 as cv 
import numpy as np

def image_processing(filename):
    img = cv.imread(filename)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel = np.ones((5,5),np.uint8)

    img = cv.GaussianBlur(img, (5,5),cv.BORDER_DEFAULT)

    img = cv.fastNlMeansDenoising(img, None, 10, 7, 21 )

    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    
    opening = cv.dilate(img,kernel,iterations = 1)

    return cv.adaptiveThreshold(opening, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

def findContour(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    listOfRect = printExternalContour(image, contours, hierarchy)
    for rect in listOfRect:
        cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)

    return listOfRect, image


def printExternalContour(img, contours, hierarchy):
    listOfHierarchy = hierarchy[0]
    listOfRect, listOfLevel, listOfContours, done = ([] for i in range(4))
    currLv = 0

    for hierarchY in listOfHierarchy:
        listOfLevel.append(0)
        done.append(0)

    for i in range(len(listOfHierarchy)):
        if done[i] == 0:
            done[i] = 1
            next = listOfHierarchy[i][0]
            while next != -1:
                done[next] = 1
                listOfLevel[next] = currLv
                next = listOfHierarchy[next][0]
            currLv+=1

    for i in range(len(listOfHierarchy)):
        # if listOfLevel[i]%2 != 0:
        listOfContours.append(contours[i])
        
    contours_poly = [None]*len(listOfContours)

    for i, c in enumerate(listOfContours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        tmp = cv.boundingRect(contours_poly[i])
        if tmp[2]*tmp[3] < 6000 and tmp[2]*tmp[3] > 850:
            listOfRect.append(cv.boundingRect(contours_poly[i]))
    # print(listOfLevel)
    return listOfRect

if __name__ == "__main__":
    #unit test
    filename = './page002.jpg'
    img = image_processing(filename)

    cv.imshow("Frame", img)
    cv.waitKey(0)  

    _, image = findContour(img)    
     
    cv.imshow("Frame", image)
    cv.waitKey(0)   


