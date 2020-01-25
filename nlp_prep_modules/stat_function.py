from __future__ import division
from __future__ import print_function

import collections
import statistics as stats
import matplotlib.pyplot as plot
import scipy.stats as stat
import numpy as np

def meanCal(listIn):
    mean = stats.mean(listIn)
    return mean

def statCalculator(listOfRect):
    print(listOfRect)
    x, y, width, height, area = ([] for i in range(5))

    for rect in listOfRect:
        x.append(rect[0])
        y.append(rect[1])
        width.append(rect[2])
        height.append(rect[3])
        area.append(rect[2]*rect[3])

    #Cross out any elements in the paremeter lists that is out of interquartile range  
    for element in width:
        if element > np.quantile(width, .75) or element < np.quantile(width, .25) :
            width.remove(element)
    print('\n')       
    for element in height:
        if element > np.quantile(height, .75)  or element < np.quantile(height, .25) :
            height.remove(element)

    #Return mean width and height of the bounding boxes
    return meanCal(width), meanCal(height)
    

if __name__ == "__main__":
    #unit test
    with open("output.txt", "r")  as file:
    
        boards2 = collections.defaultdict(lambda: 0)
        boards2[file.readline()] = boards2[file.readline()] + 1
    
        while file.readline() != '': 
            boards2[file.readline()] = boards2[file.readline()] + 1

    listOfRect = []
    for board in boards2:
        newString = board.strip().split(' ')
        if newString != ['']:
            tmp = list(map(int, newString ))
            listOfRect.append(tmp)
    print(statCalculator(listOfRect))