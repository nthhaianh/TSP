import sys
import os

from app import readParameter, getDataInformation, saveDistanceMatrix


constantDict = readParameter(sys.argv[1])
dataFolder = constantDict['DATA_FOLDER']
distanceMatricesFolder = constantDict['DISTANCE_MATRICES_FOLDER']
problemNamesList = constantDict['PROBLEM_NAMES']
# Make folder if not exist
os.makedirs(distanceMatricesFolder, exist_ok=True)

for problemName in problemNamesList:
    saveDistanceMatrix(problemName, dataFolder, distanceMatricesFolder)
print('Done.')
