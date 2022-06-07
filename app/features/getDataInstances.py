import pandas as pd
import os
import sys

from app import readParameter, getDataInformation


constantDict = readParameter(sys.argv[1])
dataInstancesFolder = constantDict['DATA_INSTANCES_FOLDER']
dataFolder = constantDict['DATA_FOLDER']
optimalResultsFolder = constantDict['OPTIMAL_RESULTS_FOLDER']
problemNamesList = constantDict['PROBLEM_NAMES']
# Make folder if not exist
os.makedirs(dataInstancesFolder, exist_ok=True)

optimalTourList = []
for fileName in os.listdir(optimalResultsFolder):
    # fileName (e.g. "a280") is added
    optimalTourList.append(fileName.split('.')[0])

bestKnownSolutionDF = pd.read_json(dataInstancesFolder / 'TSPLIB_BestKnownSolution.json', lines=True)

dataList = []

for problemName in problemNamesList:
    dataDescription = getDataInformation(problemName, dataFolder)[0]
    dimension = dataDescription['DIMENSION']
    if problemName in bestKnownSolutionDF['instance'].tolist():
        # Get bestKnownSolution value according to the problem in the instance column
        bestKnownSolution = bestKnownSolutionDF.loc[bestKnownSolutionDF['instance']
                                                    == problemName, 'bestKnownSolution'].iloc[0]
    else:
        bestKnownSolution = None
    existOptimalTour = problemName in problemNamesList
    givenCoordinates = dataDescription['GIVEN_COORDINATES']

    dataDict = {}
    dataDict['name'] = problemName
    dataDict['dimension'] = dimension
    dataDict['bestKnownSolution'] = bestKnownSolution
    dataDict['existOptimalTour'] = existOptimalTour
    dataDict['givenCoordinates'] = givenCoordinates
    dataList.append(dataDict)

df = pd.DataFrame(dataList)

result = df.to_json(dataInstancesFolder / 'dataInstances.json', orient='records', lines=True)
