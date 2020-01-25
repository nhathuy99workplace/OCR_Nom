import nlp_prep_modules.cv_function as cv_f
import nlp_prep_modules.stat_function as stat_f 
import nlp_prep_modules.pos_process as pos_f 

import cv2 as cv

filename = './nlp_prep_modules/page002.jpg'

raw = cv.imread(filename)
img = cv_f.image_processing(filename)

cv.imshow("Frame", img)
cv.waitKey(0)  

listOfRect, image = cv_f.findContour(img)    
     
cv.imshow("Frame", image)
cv.waitKey(0)   

standardW, standardH = stat_f.statCalculator(listOfRect)

extractedX = pos_f.extractCoordinatePartition(listOfRect, 0)
extractedY = pos_f.extractCoordinatePartition(listOfRect, 1)

# print(extractedX)
# print(extractedY)

neededX = pos_f.coordinateNeeded(standardW, extractedX, 0)
neededY = pos_f.coordinateNeeded(standardH, extractedY, 1)

print(neededX)
print(neededY)

listOfNewRect = []

i = len(neededX) -1
j = len(neededY) -1 
while i>=0:
    while j>=0:
        rect = (neededX[i], neededY[j], int(standardW), int(standardH))
        listOfNewRect.append(rect)
        j-=1
    j = len(neededY) -1 
    i-=1

print(listOfNewRect)

with open("coorOutput.txt", "w")  as file:
    for rect in listOfNewRect:
        file.write("%s %s %s %s\n" % (rect[0], rect[1], rect[2], rect[3]))

for rect in listOfNewRect:
    cv.rectangle(raw, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)

cv.imshow("Frame", raw)
cv.waitKey(0)   