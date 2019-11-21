import numpy as np
import random
import operator
import matplotlib.pyplot as plt

class City:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def distance(self,city):
        a = abs(self.x - city.x)**2
        b = abs(self.y - city.y)**2
        return np.sqrt(a + b)

    def __repr__(self):
        return f"(x: {self.x},y:{self.y})"

class Fitness:
    def __init__(self,route):
        self.route = route
        self.fitness = 0.0
        self.distance = 0
        self.distance = self.routeDistance()

    def routeDistance(self):
        if(self.distance == 0):
            for i in range(0,len(self.route)):
                self.distance += self.route[i].distance(self.route[(i+1) % len(self.route)])
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1.0 / self.routeDistance()
        return self.fitness


def createRoute(cityList):
    return random.sample(cityList,len(cityList))

def createInitialPopulation(populationSize,cityList):
    population = []
    for _ in range(populationSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    result = []
    for i in range(len(population)):
        result.append(Fitness(population[i]))
    return sorted(result,key=operator.attrgetter('distance'))

def selection(populationRanked, eliteSize):
    selectionResults = []
   # selectionResults.append(populationRanked[0].route)
    for i in range(eliteSize):
       selectionResults.append(populationRanked[i].route)
    p = []
    sum_routes = 0
    for elem in populationRanked:
        sum_routes += elem.distance
    for i in range(len(populationRanked)):
        p.append(populationRanked[i].distance / sum_routes)
    fitResults = np.random.choice(populationRanked,size=(len(populationRanked)-eliteSize-1),replace=False,p=p.reverse())
    for elem in fitResults:
        selectionResults.append(elem.route)
    return selectionResults

def breed(parent1,parent2):
    size = len(parent1)
    child1 = []
    child2 = []
    gene1 = random.randrange(size) 
    gene2 = random.randrange(size)
    start = min(gene1,gene2)
    end = max(gene1,gene2)

    for i in range(start,end):
        child1.append(parent1[i])
        
    child2 = [item for item in parent2 if item not in child1]
   
    return child2[0:start] + child1 + child2[start:]

def breedPopulation(selectionResults, eliteSize):
    children = []
    for i in range(eliteSize):
        children.append(selectionResults[i])
    
    for i in range(len(selectionResults)-eliteSize):
        child = breed(selectionResults[i],selectionResults[len(selectionResults)-i-1])
        children.append(child)

    return children

def mutate(route, mutationRate):
    for swapped in range(len(route)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(route))
            tmp = route[swapWith]
            route[swapWith] = route[swapped]
            route[swapped] = tmp
    return route

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    mutatedPop.append(population[0].copy())
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen,eliteSize,mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionRes = selection(popRanked,eliteSize)
    children = breedPopulation(selectionRes, eliteSize)
    nextGeneration = mutatePopulation(children,mutationRate)
    return nextGeneration

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = createInitialPopulation(popSize, population)
    progress = []
    progress.append(rankRoutes(pop)[0].distance)
    
    for k in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(rankRoutes(pop)[0].distance)
        if(k % 100) == 0:
            for i in range(len(pop[0])):
                plt.plot([pop[0][i].x,pop[0][(i+1) % len(pop[0])].x], [pop[0][i].y,pop[0][(i+1) % len(pop[0])].y],'ro-')
            plt.xlabel(f'Distance: {round(progress[-1])}, Iteration: {k}')
            plt.pause(0.10)
            plt.clf()
    for i in range(len(pop[0])):
        plt.plot([pop[0][i ].x,pop[0][(i+1) % len(pop[0])].x], [pop[0][i].y,pop[0][(i+1) % len(pop[0])].y],'ro-')
        plt.xlabel(f'Distance: {round(progress[-1])}, Iteration: {i}')
    plt.show()


def main():
    cityList = []
    random.seed(123)
    np.random.seed(1243)
    for i in range(0,25):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
    geneticAlgorithmPlot(population=cityList,popSize=100,eliteSize=20,mutationRate=0.02,generations=30000000)

if __name__ == "__main__":
    main()