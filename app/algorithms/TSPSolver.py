import time

from .NearestNeighbor import NearestNeighbor

class TSPSolver:
    def __init__(self, algoName, dataModel, constantDict, timeLimit):
        algoList = [NearestNeighbor]
        self.algorithm = None

        for algo in algoList:
            if algo.classAbbreviation == algoName:
                self.algorithm = algo

        if self.algorithm is None:
            raise ValueError(f'Invalid algoName: {self.algoName}')

        self.dataModel = dataModel
        self.constantDict = constantDict
        self.executionTime = 0
        self.timeLimit = timeLimit

    @property
    def comment(self):
        try:
            return self.model.comment
        except:
            return f'Solve by {self.algorithm.classAbbreviation}'

    def solve(self):
        timeStart = time.time()
        self.model = self.algorithm.construct(self.dataModel, self.constantDict, self.timeLimit)
        solution = self.model.solve()
        self.executionTime = time.time() - timeStart
        return solution
    
    def multiSolve(self):
        timeStart = time.time()
        self.model = self.algorithm.construct(self.dataModel, self.constantDict, self.timeLimit)
        solution = self.model.multiSolve()
        self.executionTime = time.time() - timeStart
        return solution
