import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import sys

class City:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"(x: {self.x},y:{self.y})"

def distance(origin,destination):
        a = abs(origin.x - destination.x)**2
        b = abs(origin.y - destination.y)**2
        return np.sqrt(a + b)

def findNearest(origin, cityList):
    shortest = sys.maxsize
    choice = -1
    for i in range(len(cityList)):
        if (distance(origin,cityList[i]) < shortest):
            shortest = distance(origin,cityList[i])
            choice = i
    return cityList[choice]

def createRoute(first,cityList):
    route = []
    route.append(first)
    route.append(findNearest(first,cityList))
    cityList.remove(route[1])
    k = 1
    while(cityList != []):
        route.append(findNearest(route[k], cityList))
        #for i in range(len(route)-1):
        #    plt.plot([route[i ].x,route[(i+1) % len(route)].x], [route[i].y,route[(i+1) % len(route)].y],'ro-')
        #plt.xlabel(f'Distance: {0}, Iteration: {i}')
        
        #plt.pause(0.15)
        #plt.clf()
        cityList.remove(route[k+1])
        k += 1
    return route

def swap(route, j, k):
    for i in range(0,(k-j + 1)//2):
        tmp = route[j+i]
        route[j+i] = route[k-i]
        route[k-i] = tmp
    return route

def routeDistance(route):
    s = 0
    for i in range(len(route)-1):
        s += distance(route[i],route[i+1])
    s += distance(route[-1],route[0])
    return s

def swapDistance(route,j,k):
    res = 0
    size = len(route)
    for i in range(0,j-1):
        res += distance(route[i],route[i+1])
    for i in range(j,k):
        res += distance(route[i],route[i+1])

    res+= distance(route[j-1],route[k])
    if k < size-1:
        res+= distance(route[j],route[k+1])
    for i in range(k+1,size-1):
        res += distance(route[i],route[i+1])

    if k == size:
        res += distance(route[j],route[0])
    else:
        res += distance(route[-1],route[0])
    return res

def main():
    cityList = []
    x = []
    y = []
    random.seed(123)
    for i in range(0,100):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
        x.append(cityList[i].x)
        y.append(cityList[i].y)
    first = cityList[0]
    print(swap([1,2,3,4],1,2))
    plt.scatter(x,y)
    plt.pause(1)
    cityList.remove(first)
    route = createRoute(first,cityList)
    route.append(City(x=route[0].x,y=route[0].y))
    print("Start swapping \n")
   # route = [City(x=100,y=0),City(x=100,y=100),City(x=0,y=0),City(x=50,y=100),City(x=0,y=100)]
    oldDistance = routeDistance(route)
    #print(f"Old distance:{oldDistance}")
    #print(f"New: {swapDistance(route,2,3)}")
    size = len(route)
    newDistance = oldDistance
    for i in range(len(route)):
        plt.plot([route[i ].x,route[(i+1) % len(route)].x], [route[i].y,route[(i+1) % len(route)].y],'ro-')
    plt.xlabel(f'Distance: {routeDistance(route)}, Iteration: {i}')
    plt.pause(0.1)
    plt.clf()

    while True:
        gotoStart = False
        for i in range(1,size-1):
            for j in range(i+1,size):
                #newRoute = swap(route.copy(),i,j)
                #distance = routeDistance(newRoute)
                distance = swapDistance(route,i,j)
        
                if distance < oldDistance:
                    newDistance = distance
                    route = swap(route,i,j)
                    gotoStart = True
                if gotoStart:
                    break
            if gotoStart:
                break
        if newDistance == oldDistance:
            break
        oldDistance = newDistance
        for i in range(len(route)):
            plt.plot([route[i ].x,route[(i+1) % len(route)].x], [route[i].y,route[(i+1) % len(route)].y],'ro-')
        plt.xlabel(f'Distance: {routeDistance(route)}, Iteration: {i}')
        plt.pause(0.1)
        plt.clf()
    print("Finished!")
    for i in range(len(route)):
        plt.plot([route[i ].x,route[(i+1) % len(route)].x], [route[i].y,route[(i+1) % len(route)].y],'ro-')
    plt.xlabel(f'Distance: {routeDistance(route)}, Iteration: {i}')
    plt.show()

if __name__ == "__main__":
    main()