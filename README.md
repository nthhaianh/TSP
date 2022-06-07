# TSP - Traveling Salesman Problems
Traveling Salesman Problems, Christofides Algorithm, Nearest Neighbors

# Installation

***Note: The directory in Linux is different from that in Windows. Therefore, if errors with '/' and '\\' encountered, the path need to be modified.*** 

Example:
- On Linux:	`https://github.com/optimahus`
- On Windows:	`https:\\github.com\optimahus`

## Clone repository
First of all, you need to download the repository. You can either run the script below on the command-line or terminal:

`git clone https://github.com/optimahus/TSP.git`

or download zip file of repository and unzip. 

If you have the problem related to personal access token, try following the steps below:
- Log in to your GitHub account
- Go to `Settings`, then choose `Developer Settings`
- Choose `Personal access tokens`
- Generate and copy your tokens
- Clone again and paste tokens to password space

## Change directory
Change the `path` that points to your TSP folder.

```
cd path/to/TSP
```

## Set PYTHONPATH
Add `TSP/` directory pathname to PYTHONPATH.
```
set_env.bat # For cmd
set_env.ps1 # For powershell
```
## Create python virtual environment:
* Create environment by following command:
    ```console
    python3 -m venv env
    ```

* Activate environment:
    ```console
    env/Scripts/activate # For windows
    source env/bin/activate # For Linux or MaxOS
    ```

* Install libraries in by:
  ```console
  pip install -r requirements.txt
  ```

* Deactivate environment:
    ```console
    env/Scripts/deactivate # For windows
    source env/bin/deactivate # For Linux or MaxOS
    ```

# Features
## 1. <a name="saveDistanceMatrix"></a>Create distance matrix files
One can generate and save distance matrix files locally by running the following:
```console
python3 app/features/saveDistanceMatrix.py path/to/parameter.txt
```
The distance matrix files (e.g. `a280.matrix.tsp`) will then be generated and saved in `DISTANCE_MATRICES_FOLDER` (in the parameter file) folder.

## 2. Create json file that stores data information
One can get data instances' information summary in a json file `dataInstances.json` by running the following:
```console
python3 app/features/getDataInstances.py path/to/parameter.txt
```
The `dataInstances.json` will then be saved in `dataInstances` (in the parameter file) folder.

## 3. Solve TSP
One can solve the TSP problems using the algorithms above by running the following:
```console
python3 app/features/solve.py path/to/parameters.txt
```
The results will then be saved in `OUTPUT_FOLDER` (in the parameter file) folder.

**Note**: 
* The distance matrix files should be generated locally beforehand ([feature 1](#saveDistanceMatrix)) for faster performance.
* Rerunning the same parameter file will not guarantee the same solution

## 4. Evaluate results
One can evaluate the results after solving by running the following:
```console
python3 app/feature/evaluateResults.py path/to/parameters.txt
```
The algorithms' results comparison and the leader board for the algorithms will then be saved in `EVALUATE_IMAGE_FOLDER` and `LEADER_BOARD_FOLDER` (in the parameter file) folders , respectively.
