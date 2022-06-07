import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import warnings

from app import getStatistics, readParameter

# ignore all warning while running program
warnings.filterwarnings('ignore')

constantDict = readParameter(sys.argv[1])
# get all necessary arguments
problemNameList = constantDict['PROBLEM_NAMES']
algorithmNameList = constantDict['ALGORITHM_NAMES']
outputFolder = constantDict['OUTPUT_FOLDER']
dataInstancesFolder = constantDict['DATA_INSTANCES_FOLDER']
date = constantDict['DATE_RUN']
statisticResults = getStatistics(problemNameList, algorithmNameList, outputFolder, dataInstancesFolder, date)

# Plot images for evaluating
evaluateImageFolder = constantDict['EVALUATE_IMAGE_FOLDER']
# Make folder if not exist
os.makedirs(evaluateImageFolder, exist_ok=True)

algoResults = [f'{algo}Result' for algo in algorithmNameList]
bestKnownSolution = statisticResults['bestKnownSolution']
gapAlgoNameList = []

for algo, algoResult in zip(algorithmNameList, algoResults):
    columnName = f'{algo}Gap'
    gapAlgoNameList.append(columnName)
    # add new column that calculating deviation of result to frame
    statisticResults[columnName] = abs(statisticResults[algoResult] - bestKnownSolution) / bestKnownSolution

statisticResults.sort_values(by='dimension', inplace=True)  # Sort frame to dimension
threshold = 0.25
gapStandards = []

for algo in gapAlgoNameList:
    columnName = f'{algo}Standard'
    gapStandards.append(columnName)
    statisticResults[columnName] = statisticResults[algo].apply(lambda x: x if x <= threshold else threshold)

meanAlgorithms = [statisticResults[gap].mean() for gap in gapAlgoNameList]
indexMeanAlgoSorted = np.argsort(meanAlgorithms)
legend = [f'{algorithmNameList[index]:8.5s}{meanAlgorithms[index]:1.4f}'
          for index in indexMeanAlgoSorted]

fig, ax = plt.subplots(figsize=(15, 10))
for index in indexMeanAlgoSorted:
    algo = gapStandards[index]
    plt.plot(statisticResults['problemName'], statisticResults[algo])
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.7, alpha=0.5)
plt.xticks(rotation=70)
plt.title('Gap of tour from best known solution', fontsize=20)
plt.ylabel('Gap', fontsize=12)
plt.xlabel('Problem name', fontsize=12)
plt.legend(legend, prop={'family': 'monospace'}, title='Mean Gap')

# save file
fileName = f'statisticToParameters.{date}.png'
plt.savefig(evaluateImageFolder / fileName)

# Make leader board
columns = ['problemName', 'bestKnownSolution'] + gapAlgoNameList
leaderBoard = statisticResults[columns]
numAlgorithms = len(algorithmNameList)
orders = []

# make columns to arrange performance of algorithms
for index in range(numAlgorithms):
    if index == 0:
        orderName = f'{index+1}st'
        orders.append(orderName)
    elif index == 1:
        orderName = f'{index+1}nd'
        orders.append(orderName)
    elif index == 2:
        orderName = f'{index+1}rd'
        orders.append(orderName)
    else:
        orderName = f'{index + 1}th'
        orders.append(orderName)

# arrange performance to 1st, 2nd, 3rd, 4th, ...
for index, row in leaderBoard.iterrows():
    gaps = row[gapAlgoNameList]
    rankInd = np.argsort(gaps)
    content = [f"{algo}-{np.round(row[f'{algo}Gap'] * 100, 2)}%" for algo in algorithmNameList]
    content = np.array(content)[rankInd]
    leaderBoard.loc[index, orders] = content

# remove unnecessary columns
leaderBoard.drop(columns=gapAlgoNameList, inplace=True)
# Save leaderboard
fileName = f'leader_board.{date}.json'
leaderBoardFolder = constantDict['LEADER_BOARD_FOLDER']

# Make folder if not exist
os.makedirs(leaderBoardFolder, exist_ok=True)

leaderBoard.to_json(leaderBoardFolder / fileName, orient='records', lines=True)
