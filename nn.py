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
        for i in range(len(route)-1):
            plt.plot([route[i ].x,route[(i+1) % len(route)].x], [route[i].y,route[(i+1) % len(route)].y],'ro-')
        plt.xlabel(f'Distance: {0}, Iteration: {i}')
        
        plt.pause(0.05)
        
        cityList.remove(route[k+1])
        k += 1
    return route


def main():
    cityList = []
    x = []
    y = []
    random.seed(3)
    for i in range(0,50):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
        x.append(cityList[i].x)
        y.append(cityList[i].y)
    first = cityList[0]
    
    plt.scatter(x,y)
    plt.pause(1)
    cityList.remove(first)
    route = createRoute(first,cityList)

    for i in range(len(route)):
        plt.plot([route[i ].x,route[(i+1) % len(route)].x], [route[i].y,route[(i+1) % len(route)].y],'ro-')
    plt.xlabel(f'Distance: {0}, Iteration: {i}')
    plt.show()

if __name__ == "__main__":
    main()