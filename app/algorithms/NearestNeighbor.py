import random
import time
import numpy as np
from itertools import combinations

from .utils import Solution


class NearestNeighbor:
    classAbbreviation = 'nn'

    def __init__(self, dataModel, maxIterations=10, timeLimit=1000):
        self.__dataModel = dataModel
        self.__distanceMatrix = dataModel.distanceMatrix
        self.__bestSolution = None
        self.__bestFitness = float('Inf')
        self.__size = dataModel.distanceMatrix.shape[0]
        self.__maxIterations = maxIterations
        self.__timeLimit = timeLimit
        self.comment = f'NearestNeighbor - maxIterations: {maxIterations} '

    @classmethod
    def construct(cls, dataModel, constantDict, timeLimit):
        return cls(dataModel, constantDict['MAX_ITERATION'], timeLimit)


    def __bestNeighbor(self, customer):
        rank = np.zeros((self.__size, 2), dtype=int)
        for i in range(self.__size):
            rank[i, 0] = self.__distanceMatrix[i, customer]
            rank[i, 1] = i
        rank = rank[rank[:, 0].argsort()]
        return rank[:self.__size, 1]


    def __initialSolution(self):
        candidateList = [index for index in range(self.__size)]
        customerList = [random.choice(candidateList)]

        for i in range(self.__size):
            count = 0
            rand = random.random()

            if len(customerList) < self.__size:
                nextCustomer = self.__bestNeighbor(customerList[-1])[count]

                while nextCustomer in customerList:
                    count = np.clip(count + 1, 0, self.__size - 1)
                    nextCustomer = self.__bestNeighbor(customerList[-1])[count]

                customerList.append(nextCustomer)
                candidateList.remove(nextCustomer)

        return Solution(customerList, self.__distanceMatrix)

    def solve(self):
        timeStart = time.time()
        solution = self.__initialSolution()
        self.__bestSolution = solution
        self.__bestFitness = solution.totalDistance

        # Loop 2-OPT
        localSearchModel = LocalSearch(self.__dataModel, solution)
        currentSolution = localSearchModel.solve(timeStart, self.__timeLimit)

        if currentSolution.totalDistance < self.__bestFitness:
            self.__bestSolution = currentSolution
            self.__bestFitness = currentSolution.totalDistance

        print(f'Total Distance: {self.__bestFitness}\n')
        return self.__bestSolution


    def multiSolve(self):
        print(self.comment)

        timeStart = time.time()
        count = 0

        while count < self.__maxIterations:

            timeInterval = time.time() - timeStart
            if timeInterval > self.__timeLimit:
                break

            # Initial Solution
            solution = self.__initialSolution()

            # Loop 2-OPT
            localSearchModel = LocalSearch(self.__dataModel, solution)
            currentSolution = localSearchModel.solve(timeStart, self.__timeLimit)

            # Compare Fitness Function
            if currentSolution.totalDistance < self.__bestFitness:
                self.__bestSolution = currentSolution
                self.__bestFitness = currentSolution.totalDistance

            count = count + 1
            
            print(f'Iterations = {count: 1d} \tDistance: {self.__bestFitness}')

        print(f'Total Distance: {self.__bestFitness}\n')
        return self.__bestSolution


class LocalSearch:
    def __init__(self, dataModel, solution, loop=20):
        self.__loop = loop
        self.__distanceMatrix = dataModel.distanceMatrix
        self.__size = self.__distanceMatrix.shape[0]
        self.__bestSolution = solution

    def __createNeighborList(self):
        neighborsList = list(combinations(range(self.__size), 2))
        return neighborsList

    def __distanceVariesTwoOpt(self, neighbor, solution):
        customerList = solution.customerList
        maxIdx = max(neighbor)
        minIdx = min(neighbor)
        maxCustomer = customerList[maxIdx]
        minCustomer = customerList[minIdx]
        totalDistanceAfter = 0
        totalDistanceBefore = 0

        if minIdx == maxIdx:
            return 0
        elif minIdx == 0 and maxIdx + 1 == len(customerList):
            return 0
        elif minIdx == 0 and maxIdx + 2 == len(customerList):
            return 0
        elif maxIdx + 1 == len(customerList):
            nextCustomer = customerList[0]
            previousCustomer = customerList[minIdx - 1]
        else:
            nextCustomer = customerList[maxIdx + 1]
            previousCustomer = customerList[minIdx - 1]

        totalDistanceBefore += self.__distanceMatrix[previousCustomer, minCustomer]
        totalDistanceBefore += self.__distanceMatrix[nextCustomer, maxCustomer]

        totalDistanceAfter += self.__distanceMatrix[previousCustomer, maxCustomer]
        totalDistanceAfter += self.__distanceMatrix[nextCustomer, minCustomer]

        totalChange = totalDistanceBefore - totalDistanceAfter
        return totalChange

    @staticmethod
    def swap2Edges(indexNeighborChange, solution, distanceVariesChange):
        minIdx = min(indexNeighborChange)
        maxIdx = max(indexNeighborChange)
        neighborList = solution.customerList
        solution.customerList = neighborList[: minIdx] + neighborList[minIdx: maxIdx + 1][::-1] + neighborList[maxIdx + 1:]
        solution.totalDistance -= distanceVariesChange

    def solve(self, timeStart, timeLimit):
        count = 0
        for i in range(self.__loop):
            if time.time() - timeStart > timeLimit:
                break
            neighborsList = self.__createNeighborList()
            goodNeighbors = [neighbor for neighbor in neighborsList if self.__distanceVariesTwoOpt(neighbor, self.__bestSolution) > 0]
            for goodNeighbor in goodNeighbors:
                distanceVariesChange = self.__distanceVariesTwoOpt(goodNeighbor, self.__bestSolution)
                if distanceVariesChange > 0:
                    count = 0
                    self.swap2Edges(goodNeighbor, self.__bestSolution, distanceVariesChange)
                else:
                    count += 1
            # if count >= 100:
            #     break
        return self.__bestSolution