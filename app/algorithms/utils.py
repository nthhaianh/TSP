import itertools
import math
import numpy as np
import os
import random
import pandas as pd
import warnings
from pathlib import Path


class DataModel:
    """
    Contains a list of customers and dataDescription
    """
    def __init__(self, distanceMatrix, dataDescription, customerArray=None):
        self.customerArray = customerArray
        self.distanceMatrix = distanceMatrix
        self.dataDescription = dataDescription

    def __repr__(self):
        description = ''
        for kw in self.dataDescription:
            description += f'{kw}: {self.dataDescription[kw]}\n'
        description += 'Customers data:\n'
        if self.customerArray:
            for i in range(len(self.customerArray)):
                description += f'Customer ID :{i + 1}, '
                description += f'latitude: {self.customerArray[i][0]}, longitude: {self.customerArray[i][1]}\n'
        description = description[:-1]
        return description


class Solution:
    """
    Contains a list of customers and distanceMatrix
    """
    def __init__(self, customerList, distanceMatrix):
        self.customerList = customerList
        self.distanceMatrix = distanceMatrix
        self.totalDistance = self.__getTotalDistance()

    def __repr__(self):
        return str([idx + 1 for idx in self.solutionFormat(self.customerList.copy())])

    def __copy__(self):
        return self.__class__(self.customerList.copy(), self.distanceMatrix)

    def __eq__(self, other):
        return self.customerList == other.customerList

    def getBestNeighbor(self, diff, size=50):
        """
        Return best neighbor found
        """
        neighborList = []
        iter = 0
        while iter < size:
            candidate = self.customizedTwoOpt(diff)
            neighborList.append(candidate)
            iter += 1
        neighborList = sorted(neighborList, key=lambda neighbor: neighbor.totalDistance)
        return neighborList[0]

    def customizedTwoOpt(self, diff):
        """
        Swaps two edges by swapping two nodes with diff condition
        """
        size = len(self.customerList)

        while True:
            nodeIdx1 = random.choice(range(size))
            nodeIdx2 = random.choice(range(size))
            while nodeIdx1 == nodeIdx2:
                nodeIdx2 = random.choice(range(size))

            if self.get2EdgesSwapDiff(nodeIdx1, nodeIdx2) < diff:
                neighbor = self.__copy__()
                neighbor.customerList[nodeIdx1:nodeIdx2] = reversed(neighbor.customerList[nodeIdx1:nodeIdx2])
                return Solution(neighbor.customerList, self.distanceMatrix)

    def get2EdgesSwapDiff(self, nodeIdx1, nodeIdx2):
        """
        Return difference when swapping two edges
        """
        size = len(self.customerList)
        d = 0
        if nodeIdx1 == nodeIdx2:
            return d
        elif nodeIdx1 > nodeIdx2:
            nodeIdx1, nodeIdx2 = nodeIdx2, nodeIdx1

        prevNodeIdx1 = nodeIdx1 - 1 if nodeIdx1 != 0 else size - 1
        nextNodeIdx2 = nodeIdx2 + 1 if nodeIdx2 != size - 1 else 0

        d -= self.distanceMatrix[self.customerList[prevNodeIdx1], self.customerList[nodeIdx1]]
        d -= self.distanceMatrix[self.customerList[nodeIdx2], self.customerList[nextNodeIdx2]]
        d += self.distanceMatrix[self.customerList[prevNodeIdx1], self.customerList[nodeIdx2]]
        d += self.distanceMatrix[self.customerList[nodeIdx1], self.customerList[nextNodeIdx2]]
        return d

    def insertCustomer(self, customerIdx, pos=-1):
        """
        Insert a customer to the customerList, default position is the end of the customerList
        """
        d = 0
        if pos == -1 or pos == 0:
            leftCustomerIdx = self.customerList[-1]
            rightCustomerIdx = self.customerList[0]
            d -= self.distanceMatrix[leftCustomerIdx, rightCustomerIdx]

            if pos == -1:
                self.customerList.append(customerIdx)
            else:
                self.customerList.insert(pos, customerIdx)

            d += self.distanceMatrix[leftCustomerIdx, customerIdx]
            d += self.distanceMatrix[customerIdx, rightCustomerIdx]
        else:
            leftCustomerIdx = self.customerList[pos - 1]
            rightCustomerIdx = self.customerList[pos]
            d -= self.distanceMatrix[leftCustomerIdx, rightCustomerIdx]

            self.customerList.insert(pos, customerIdx)

            d += self.distanceMatrix[leftCustomerIdx, customerIdx]
            d += self.distanceMatrix[customerIdx, rightCustomerIdx]

        self.totalDistance += d

    @staticmethod
    def solutionFormat(customerList):
        firstNumberIdx = customerList.index(0)
        customerList = customerList[firstNumberIdx:] + customerList[:firstNumberIdx]
        if customerList[-1] < customerList[1]:
            customerList = [customerList[0]] + customerList[1:][::-1]
        return customerList

    def writeSolution(self, problemName, dimension, algoName, executionTime, dateTime, outputFolder, stepByStep=None,
                      comment=''):
        """
        Input: problemName: str, name of problem
                dimension: int, number of cities
                algoName: str, name of algorithm
                executionTime: int, algorithm run time
                dateTime: str, time when the algorithm runs
                stepByStep: list, a list contain Solution objects through when algorithm running
        """
        fileName = f'{problemName}.{algoName}.{dateTime}.tour'
        distance = self.totalDistance
        tourSection = self.solutionFormat(self.customerList.copy())

        resultFile = open(outputFolder / fileName, 'x')

        resultFile.write(f'NAME: {problemName} \n'
                         f'COMMENT: {comment} \n'
                         f'DIMENSION: {dimension} \n'
                         f'DISTANCE: {distance} \n'
                         f'EXECUTION_TIME: {executionTime: 4.2f}s \n'
                         f'TOUR_SECTION \n{" ".join(str(i + 1) for i in tourSection)} \n'
                         )

        if stepByStep is not None:
            resultFile.write('\nSTEP_BY_STEP_TOURS')
            stepByStep = [self.solutionFormat(solution.customerList) for solution in stepByStep]
            for customerList in stepByStep:
                resultFile.write(f'\n{" ".join(str(i + 1) for i in customerList)}')

        resultFile.write(f'\nEOF')
        resultFile.close()
        return None

    def __getTotalDistance(self):
        distance = 0
        for i, j in zip(self.customerList, self.customerList[1:] + [self.customerList[0]]):
            distance += self.distanceMatrix[i, j]
        return distance

    def __getErrorTotalDistance(self):
        """
        Check if the total distance is valid
        Return None if the total distance is valid,
        otherwise, return a list that contains the valid and the current total distances
        """
        validDistance = self.__getTotalDistance()
        if self.totalDistance == validDistance:
            return None
        else:
            return [validDistance, self.totalDistance]

    def __getErrorCustomersNumber(self):
        """
        Check if the number of customers is valid
        Return None if the number of customers is valid,
        otherwise, return a list that contains the number of customers and the problem's dimension
        """
        dimension = len(self.distanceMatrix)
        if len(self.customerList) == dimension:
            return None
        return [dimension, len(self.customerList)]

    def __getErrorCustomers(self):
        """
        Check if exists duplicate customers
        Return None if every customer appears once,
        otherwise, return a list of duplicates
        """
        dimension = len(self.distanceMatrix)
        if len(self.customerList) == dimension:
            return None
            
        customerSet = set()
        duplicates = set()
        for customer in self.customerList:
            if customer in customerSet:
                duplicates.add(customer)
            customerSet.add(customer)
        return list(duplicates)

    def getErrors(self):
        """
        Check the solution's valid conditions
        Return a list that contains strings of the errors
        """
        validityConditions = []
        validityCustomerNumber = self.__getErrorCustomersNumber()
        validityCustomers = self.__getErrorCustomers()
        validityTotalDistance = self.__getErrorTotalDistance()

        if validityCustomerNumber is not None:
            validityConditions.append(f'Solution length is not valid! '
                                      f'Dimension: {validityCustomerNumber[0]} - '
                                      f'Solution length: {validityCustomerNumber[1]}')
        if validityTotalDistance is not None:
            validityConditions.append(f'Total distance is not valid! '
                                      f'Correct total distance: {validityTotalDistance[0]} - '
                                      f'Solution total distance: {validityTotalDistance[1]}.')
        if validityCustomers is not None:
            validityConditions.append(f'Duplicate customer(s): {", ".join(map(str, validityCustomers))}')
        return validityConditions


class Constants:
    """
    Contains constants provided in the documentation
    """
    PI = 3.141592
    EARTH_RADIUS = 6378.388


def getDistance(customer1Idx, customer2Idx, customerArray, edgeWeightType):
    """
    Return distance between two customers
    """
    d = 0
    if customer1Idx == customer2Idx:
        return d
    elif edgeWeightType == 'EUC_2D':
        d = getEUC2DDistance(customer1Idx, customer2Idx, customerArray)
    elif edgeWeightType == 'MAN_2D':
        d = getMAN2DDistance(customer1Idx, customer2Idx, customerArray)
    elif edgeWeightType == 'MAX_2D':
        d = getMAX2DDistance(customer1Idx, customer2Idx, customerArray)
    elif edgeWeightType == 'GEO':
        d = getGEODistance(customer1Idx, customer2Idx, customerArray)
    elif edgeWeightType == 'ATT':
        d = getATTDistance(customer1Idx, customer2Idx, customerArray)
    elif edgeWeightType == 'CEIL_2D':
        d = getCEIL2DDistance(customer1Idx, customer2Idx, customerArray)
    return d


def getEUC2DDistance(customer1Idx, customer2Idx, customerArray):
    squaredDx = (getDX(customer1Idx, customer2Idx, customerArray)) ** 2
    squaredDy = (getDY(customer1Idx, customer2Idx, customerArray)) ** 2
    d = math.sqrt(squaredDx + squaredDy)
    return int(d + 0.5)


def getMAN2DDistance(customer1Idx, customer2Idx, customerArray):
    dx = abs(getDX(customer1Idx, customer2Idx, customerArray))
    dy = abs(getDY(customer1Idx, customer2Idx, customerArray))
    d = dx + dy
    return int(d + 0.5)


def getMAX2DDistance(customer1Idx, customer2Idx, customerArray):
    dx = abs(getDX(customer1Idx, customer2Idx, customerArray))
    dy = abs(getDY(customer1Idx, customer2Idx, customerArray))
    return max(int(dx + 0.5), int(dy + 0.5))


def getGEODistance(customer1Idx, customer2Idx, customerArray):
    q1 = math.cos(getDY(customer1Idx, customer2Idx, customerArray))
    q2 = math.cos(getDX(customer1Idx, customer2Idx, customerArray))
    q3 = math.cos(customerArray[customer1Idx, 0] + customerArray[customer2Idx, 0])
    return int(Constants.EARTH_RADIUS * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)


def getATTDistance(customer1Idx, customer2Idx, customerArray):
    dx = getDX(customer1Idx, customer2Idx, customerArray)
    dy = getDY(customer1Idx, customer2Idx, customerArray)
    r = math.sqrt((dx ** 2 + dy ** 2) / 10.0)
    t = int(r + 0.5)
    if t < r:
        return t + 1
    return t


def getCEIL2DDistance(customer1Idx, customer2Idx, customerArray):
    squaredDx = (getDX(customer1Idx, customer2Idx, customerArray)) ** 2
    squaredDy = (getDY(customer1Idx, customer2Idx, customerArray)) ** 2
    d = math.sqrt(squaredDx + squaredDy)
    return math.ceil(d)


def getDX(customer1Idx, customer2Idx, customerArray):
    """
    Return x-distance of two customers
    """
    return customerArray[customer1Idx, 0] - customerArray[customer2Idx, 0]


def getDY(customer1Idx, customer2Idx, customerArray):
    """
    Return y-distance of two customers
    """
    return customerArray[customer1Idx, 1] - customerArray[customer2Idx, 1]


def getLatitude(xCoordinate):
    """
    Return converted-latitude of the x coordinate
    """
    deg = int(xCoordinate)
    min = xCoordinate - deg
    latitude = Constants.PI * (deg + 5.0 * min / 3.0) / 180.0
    return latitude


def getLongitude(yCoordinate):
    """
    Return converted-longitude of the y coordinate
    """
    deg = int(yCoordinate)
    min = yCoordinate - deg
    longitude = Constants.PI * (deg + 5.0 * min / 3.0) / 180.0
    return longitude


def getDataInformation(problemName, dataFolder):
    """
    Return dataDescription and customerArray
    """
    # e.g. 'a280' -> 'a280.tsp'
    fileName = f'{problemName}.tsp'

    # Navigate to data file
    dataPath = dataFolder / fileName

    with open(dataPath) as fObject:
        nodeData = fObject.readlines()

    NAME = ''
    TYPE = ''
    COMMENT = ''
    DIMENSION = 0
    EdgeWeightType = ''
    EdgeWeightFormat = ''
    DisplayDataType = ''
    DisplayDataSectionIndex = 0
    EdgeWeightSectionIndex = 0
    NodeCoordSectionIndex = 0

    for i in range(len(nodeData)):
        node = nodeData[i].split()
        if ':' in node:
            node.remove(':')

        if len(node) != 0:
            if 'NAME' in node[0]:
                NAME = node[1]
            if 'TYPE' in node[0]:
                TYPE = node[1]
            if 'COMMENT' in node[0]:
                COMMENT = ' '.join(node[1:])
            if 'DIMENSION' in node[0]:
                DIMENSION = int(node[1])
            if 'EDGE_WEIGHT_TYPE' in node[0]:
                EdgeWeightType = node[1]
            if 'EDGE_WEIGHT_FORMAT' in node[0]:
                EdgeWeightFormat = node[1]
            if 'DISPLAY_DATA_TYPE' in node[0]:
                DisplayDataType = node[1]
            if 'EDGE_WEIGHT_SECTION' in node[0]:
                EdgeWeightSectionIndex = i + 1
            if 'DISPLAY_DATA_SECTION' in node[0]:
                DisplayDataSectionIndex = i + 1
            if 'NODE_COORD_SECTION' in node[0]:
                NodeCoordSectionIndex = i + 1

    dataDescription = {'NAME': NAME, 'TYPE': TYPE, 'COMMENT': COMMENT, 'DIMENSION': DIMENSION}
    if NodeCoordSectionIndex > 0:
        dataDescription['GIVEN_COORDINATES'] = True
    else:
        dataDescription['GIVEN_COORDINATES'] = False

    nodeStartIndex = 0
    if EdgeWeightType in ['EUC_2D', 'EUC_3D', 'MAN_2D', 'MAN_3D', 'MAX_2D', 'MAX_3D', 'GEO', 'ATT', 'CEIL_2D']:
        nodeStartIndex = NodeCoordSectionIndex
    elif EdgeWeightType == 'EXPLICIT':
        nodeStartIndex = DisplayDataSectionIndex

    customerArray = None
    if nodeStartIndex:
        customerCoordList = []
        while nodeStartIndex < len(nodeData) and 'EOF' not in nodeData[nodeStartIndex] and nodeData[nodeStartIndex] != '':
            if ' ' in nodeData[nodeStartIndex]:
                customerData = nodeData[nodeStartIndex].split(' ')
            elif '\t' in nodeData[nodeStartIndex]:
                customerData = nodeData[nodeStartIndex].split('\t')
            customerData = [i for i in customerData if i != '' and i != '\n']
            customerData[-1] = customerData[-1][:-1]

            if EdgeWeightType == 'GEO':
                customerCoordList.append([getLatitude(float(customerData[1])), getLongitude(float(customerData[2]))])
            else:
                customerCoordList.append([float(customerData[1]), float(customerData[2])])
            nodeStartIndex += 1

        customerArray = np.array(customerCoordList)
    return dataDescription, customerArray


def getDistanceMatrix(problemName, dataFolder, customerArray=None):
    """
    Get distance matrix
    """
    # e.g. 'a280' -> 'a280.tsp'
    fileName = f'{problemName}.tsp'

    # Navigate to data file
    dataPath = dataFolder / fileName

    with open(dataPath) as fObject:
        nodeData = fObject.readlines()

    DIMENSION = 0
    EdgeWeightType = ''
    EdgeWeightFormat = ''
    EdgeWeightSectionIndex = 0

    for i in range(len(nodeData)):
        node = nodeData[i].split()
        if ':' in node:
            node.remove(':')

        if len(node) != 0:
            if 'DIMENSION' in node[0]:
                DIMENSION = int(node[1])
            if 'EDGE_WEIGHT_TYPE' in node[0]:
                EdgeWeightType = node[1]
            if 'EDGE_WEIGHT_FORMAT' in node[0]:
                EdgeWeightFormat = node[1]
            if 'EDGE_WEIGHT_SECTION' in node[0]:
                EdgeWeightSectionIndex = i + 1

    distanceMatrix = np.zeros((DIMENSION, DIMENSION), dtype=int)

    if EdgeWeightType == 'EXPLICIT':
        distanceDataList = []

        distancesStartIndex = EdgeWeightSectionIndex

        while 'DISPLAY_DATA_SECTION' not in nodeData[distancesStartIndex] and 'EOF' not in nodeData[distancesStartIndex]:
            rowData = nodeData[distancesStartIndex].split(' ')
            rowData = [num for num in rowData if num != '' and num != '\n']
            distanceDataList += rowData
            distancesStartIndex += 1

        if EdgeWeightFormat == 'FULL_MATRIX':
            for i in range(DIMENSION):
                for j in range(i, DIMENSION):
                    distanceMatrix[i, j] = distanceDataList[i * DIMENSION + j]
                    distanceMatrix[j, i] = distanceMatrix[i, j]

        if EdgeWeightFormat == 'UPPER_ROW':
            dataCount = 0
            for i in range(DIMENSION):
                for j in range(DIMENSION - 1 - i):
                    distanceMatrix[i, i + 1 + j] = distanceDataList[dataCount]
                    dataCount += 1
            for i, j in itertools.combinations_with_replacement(range(DIMENSION - 1, -1, -1), 2):
                distanceMatrix[i, j] = distanceMatrix[j, i]

        if EdgeWeightFormat == 'LOWER_ROW':
            dataCount = 0
            for i in range(DIMENSION):
                for j in range(i):
                    distanceMatrix[i, j] = distanceDataList[dataCount]
                    dataCount += 1
            for i, j in itertools.combinations_with_replacement(range(DIMENSION), 2):
                distanceMatrix[i, j] = distanceMatrix[j, i]

        if EdgeWeightFormat == 'UPPER_DIAG_ROW':
            dataCount = 0
            for i in range(DIMENSION):
                for j in range(DIMENSION - i):
                    distanceMatrix[i, i + j] = distanceDataList[dataCount]
                    dataCount += 1
            for i, j in itertools.combinations_with_replacement(range(DIMENSION - 1, -1, -1), 2):
                distanceMatrix[i, j] = distanceMatrix[j, i]

        if EdgeWeightFormat == 'LOWER_DIAG_ROW':
            dataCount = 0
            for i in range(DIMENSION):
                for j in range(i + 1):
                    distanceMatrix[i, j] = distanceDataList[dataCount]
                    dataCount += 1
            for i, j in itertools.combinations_with_replacement(range(DIMENSION), 2):
                distanceMatrix[i, j] = distanceMatrix[j, i]
    else:
        for i, j in itertools.combinations_with_replacement(range(DIMENSION), 2):
            distanceMatrix[i, j] = getDistance(i, j, customerArray, EdgeWeightType)
            distanceMatrix[j, i] = distanceMatrix[i, j]
    return distanceMatrix


def readTSPLib(problemName, dataFolder, distanceMatricesFolder):
    """
    Return cost and dataModel in the specific problemName
    """
    dataDescription, customerArray = getDataInformation(problemName, dataFolder)
    if f'{problemName}.matrix.tsp' in os.listdir(distanceMatricesFolder):
        distanceMatrix = readDistanceMatrix(problemName, distanceMatricesFolder)
    else:
        print(f'Currently the distance matrix file for problem {problemName} is not available.')
        print('Therefore, the distance matrix for this problem will now be calculated.')
        warnings.warn('The distance matrix files should be created by running saveDistanceMatrix.py in order to speed up solving.')
        distanceMatrix = getDistanceMatrix(problemName, dataFolder, customerArray)
    return DataModel(distanceMatrix, dataDescription, customerArray)


def readOPTLib(problemName, dataFolder, distanceMatricesFolder, optimalResultsFolder):
    """
    Return optimal solution from the opt.tour file
    """
    # get the cost from the corresponding data file
    dataModel = readTSPLib(problemName, dataFolder, distanceMatricesFolder)

    # e.g. get 'a280.opt.tour'
    optFileName = f'{problemName}.opt.tour'

    # Navigate to opt file
    optPath = optimalResultsFolder / optFileName

    with open(optPath) as fObject:
        optData = fObject.readlines()

    indexFirstCustomer = 0
    indexLastCustomer = 0
    optSolution = []

    for line in optData:
        if 'TOUR_SECTION' in line:
            indexFirstCustomer = optData.index(line) + 1
            break

    for index in range(len(optData) - 1, 0, -1):
        if '-1' in optData[index]:
            indexLastCustomer = index - 1
            break

    # Find optSolution
    if optData[indexFirstCustomer] == optData[indexLastCustomer]:
        optSolution = optData[indexFirstCustomer].strip('\n').split(' ')
    else:
        if len(optData[indexFirstCustomer]) == 2:
            for line in optData[indexFirstCustomer: indexLastCustomer + 1]:
                optSolution.append(line.strip('\n'))
        elif len(optData[indexFirstCustomer]) > 2:
            for line in optData[indexFirstCustomer: indexLastCustomer + 1]:
                customerInLine = line.strip('\n').split(' ')
                customerInLine = [i for i in customerInLine if i != '']
                optSolution.extend(customerInLine)

    optSolution = [int(i) - 1 for i in optSolution]
    return Solution(optSolution, dataModel.distanceMatrix)


def saveDistanceMatrix(problemName, dataFolder, distanceMatricesFolder):
    """
    Save distance matrix of a data file into a txt file
    """
    dataDescription, customerArray = getDataInformation(problemName, dataFolder)
    distanceMatrix = getDistanceMatrix(problemName, dataFolder, customerArray)

    # Get distance matrix fileName from problemName
    # e.g. get 'a280.matrix.tsp' from 'a280'
    problemName = dataDescription['NAME']
    if problemName.split('.')[-1] == 'tsp':
        problemName = problemName.split('.')[0]
    fileName = f'{problemName}.matrix.tsp'

    # Navigate to distanceMatrices file
    distanceMatrixPath = distanceMatricesFolder / fileName
    print(f'Saving {fileName}...')
    np.savetxt(distanceMatrixPath, distanceMatrix, fmt='%i')


def readDistanceMatrix(problemName, distanceMatricesFolder):
    """
    Read distance matrix from a txt file
    """
    fileName = f'{problemName}.matrix.tsp'

    # Navigate to distanceMatrices file
    distanceMatrixPath = distanceMatricesFolder / fileName

    distanceMatrix = np.loadtxt(distanceMatrixPath, dtype=int)
    return distanceMatrix


def setTimeLimit(dimension, regex):
    caseList = []
    maxSupportedDimension = 10000

    for option in regex.split('|'):
        caseList.append([int(option.split('~')[0]), int(option.split('~')[1])])

    isSupport = 1 <= dimension <= maxSupportedDimension
    if not isSupport:
        raise ValueError('No support for problem with illegal dimension')

    for i in range(0, len(caseList)):
        if i == 0:
            if 1 <= dimension <= caseList[i][0]:
                timeLimit = caseList[i][1]

        else:
            if caseList[i - 1][0] < dimension <= caseList[i][0]:
                timeLimit = caseList[i][1]

    return timeLimit


def readParameter(parameterFolder):
    """
    Return a dictionary include all parameters
    """
    constantDict = {}

    with open(parameterFolder, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == '#' or line == '\n':
            continue
        else:
            param = line.split(':')
            # Use ":".join() to get the string after the first colon
            constantDict[param[0]] = ":".join(param[1:]).replace('\n', '')

    paramType = {
        'TSP_FOLDER': Path,
        'PROBLEM_NAMES': list,
        'ALGORITHM_NAMES': list,
        'TIME_LIMIT_OPTIONS': str,
        'DATA_FOLDER': Path,
        'DATA_INSTANCES_FOLDER': Path,
        'DISTANCE_MATRICES_FOLDER': Path,
        'EVALUATE_IMAGE_FOLDER': Path,
        'LEADER_BOARD_FOLDER': Path,
        'STATISTICS_IMAGE_FOLDER': Path,
        'OPTIMAL_RESULTS_FOLDER': Path,
        'OUTPUT_FOLDER': Path,
        'LOGS_FOLDER': Path,
        'EXPORT_SOLUTION': bool,
        'DATE_RUN': str,
        'VERBOSE': int,
        'TABUS_SIZE': float,
        'NEIGHBOR_SIZE': int,
        'MAX_ITERATION': int,
        'GREEDINESS_VALUE': float,
        'TEMPERATURE': float,
        'COOLING_RATE': float,
        'STOPPING_TEMPERATURE': float,
        'STOPPING_ITER': int,
        'TIMES': int,
        'COLONY_SIZE': int,
        'POP_SIZE': int,
        'NUM_GENERATION': int,
        'PROB_MUTATE': float,
        'TYPE_SOLVE': str,
    }

    constantDict['TSP_FOLDER'] = Path(constantDict['TSP_FOLDER'])

    for key in constantDict:
        try:
            if key == 'PROBLEM_NAMES':
                continue

            if paramType[key] == bool:
                constantDict[key] = constantDict[key] == 'yes'

            elif paramType[key] == list:
                constantDict[key] = constantDict[key].split(',')

            elif paramType[key] == Path and key != 'TSP_FOLDER':
                constantDict[key] = constantDict['TSP_FOLDER'] / constantDict[key]

            else:
                constantDict[key] = paramType[key](constantDict[key])
        except:
            raise ValueError(f'Parameter {key} not found')

    if constantDict['PROBLEM_NAMES'] == 'ALL':

        #Read all file in DATA_FOLDER and remove xray.problems
        constantDict['PROBLEM_NAMES'] = [file.split('.')[0] for file in os.listdir(constantDict['DATA_FOLDER'])]
        constantDict['PROBLEM_NAMES'].remove('xray')

    else:
        constantDict['PROBLEM_NAMES'] = constantDict['PROBLEM_NAMES'].split(',')

    return constantDict


def readResultFile(problemName, algoName, date, outputFolder):
    """
    Arguments:
        problemName: name of problem
        algoName: name of algorithm
        date: string date
        outputFolder: directory to output folder
    Return:
        A dictionary that stores the result file's data
    """
    fileName = f'{problemName}.{algoName}.{date}.tour'
    with open(outputFolder / fileName) as fileObj:
        data = fileObj.readlines()

    for line in data:
        lineSplit = line.strip().split(':')
        sign = lineSplit[0].strip()
        if sign == 'NAME':
            NAME = lineSplit[1].strip()
        elif sign == 'COMMENT':
            COMMENT = lineSplit[1].strip()
        elif sign == 'DISTANCE':
            DISTANCE = int(lineSplit[1].strip())
        elif sign == 'EXECUTION_TIME':
            EXECUTION_TIME = lineSplit[1].strip()
        elif sign.startswith('1'):  # To format of tour, the first element of tour is 1.
            tourString = sign.split(' ')
            TOUR = [int(i) for i in tourString]
        else:
            continue

    dataDict = {'NAME': NAME, 'COMMENT': COMMENT, 'DISTANCE': DISTANCE, 'EXECUTION_TIME': EXECUTION_TIME,
                'TOUR': TOUR}

    return dataDict


def getStatistics(problemNameList, algorithmNameList, outputFolder, dataInstancesFolder, date):
    """
    Arguments:
        problemNameList: a list contains all problems that you want
        algorithmNameList: a list contains all algorithms that you want
        outputFolder: directory to output folder
        dataInstancesFolder: directory dataInstances folder
        date: date that your program runs those files
    Return:
        dataFrame: a dataframe about statical information of results
    """
    # Get all files in the output folder
    fileList = os.listdir(outputFolder)
    # Ending of the file whose format is "problemName.algorithmName.date.tour"
    endsWith = f'{date}.tour'
    fileList = [file for file in fileList if file.endswith(endsWith)]

    # A list that contains the dictionaries of each problem's information
    allRecords = []

    dataInstances = pd.read_json(dataInstancesFolder / 'dataInstances.json', lines=True)
    dataInstances.set_index('name', inplace=True)

    for problemName in problemNameList:
        # Initial keys
        keys = ['problemName', 'dimension']
        for algo in algorithmNameList:
            key = f'{algo}Result'
            keys.append(key)
        keys.append('bestKnownSolution')
        # Initialize value that corresponds key in keys list
        values = [0] * len(keys)
        # Assign problemName corresponds to 'problemName'
        values[0] = problemName

        for fileName in fileList:
            elementName = fileName.split('.')
            instance = elementName[0]  # get problem name
            algoName = elementName[1]  # get used algorithm name

            if instance == problemName:  # only get instance with the problemName
                # get a dictionary of the problem's information
                dataDict = readResultFile(instance, algoName, date, outputFolder)
                result = dataDict['DISTANCE']
                if algoName in algorithmNameList:
                    indexAlgo = algorithmNameList.index(algoName)
                    indexValues = keys.index('dimension') + indexAlgo + 1
                    values[indexValues] = result

        # Assign dimension in 'keys' list corresponds to 'dimension'
        # and dimension is at index (1) of keys
        dimension = dataInstances.loc[problemName, 'dimension']
        values[1] = dimension

        # Assign bestKnownSolution in 'keys' list corresponds to 'bestKnownSolution'
        # and bestKnownSolution is at last index of keys
        bestKnownSolution = dataInstances.loc[problemName, 'bestKnownSolution']
        values[-1] = bestKnownSolution

        # create dict
        record = dict(zip(keys, values))
        # append record in list
        allRecords.append(record)

    # convert all records to dataframe
    dataFrame = pd.DataFrame(allRecords)
    dataFrame.drop_duplicates(subset=['problemName'], inplace=True)

    return dataFrame
