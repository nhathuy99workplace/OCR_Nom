def extractCoordinatePartition(listOfRects, choice):
    result = []
    for rect in listOfRects:
        result.append(rect[choice])
    return result

def coordinateNeeded(standard, coordinates, choice): 
    result = []
    sortedCoordinates = sorted(set(coordinates))
    counter, curr = (0 for i in range(2))
    for coordinate in sortedCoordinates:
        counter+=1
        if coordinate > sortedCoordinates[curr] + standard :
            if counter == 1:
                result.append(sortedCoordinates[curr])
            else:
                if choice == 1:
                    result.append(sortedCoordinates[curr])
                else :
                    result.append(sortedCoordinates[curr+1])
            curr = counter
            counter = 0

    if counter > 0:
        if counter == 1:
            result.append(sortedCoordinates[curr])
        else:
            if choice == 1:
                result.append(sortedCoordinates[curr])
            else :
                result.append(sortedCoordinates[curr+1])

    return result


if __name__ == "__main__":
    #unit test
    