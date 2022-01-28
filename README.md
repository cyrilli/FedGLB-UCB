# Code for "Communication Efficient Federated Learning for Generalized Linear Bandits"

This repository contains implementation of FedGLB-UCB, and baseline algorithms for comparison:
- Three variants of FedGLB-UCB: FedGLB-UCB-v1, FedGLB-UCB-v2, FedGLB-UCB-v3 (see Section 4.2 and appendix of the paper for details)
- DisLinUCB
- One-UCB-GLM, N-UCB-GLM
- N-ONS-GLM

For experiments on the synthetic dataset, directly run:
```console
python RunSyntheticDataset.py
```
To experiment with different environment settings, specify parameters:
- T: length of time horizon
- N: number of users per time step
- d: dimension of the context vector

Detailed description of the federated generalized linear bandit setting can be found in Section 3.3 of the paper.

Experiment results can be found in "./SimulationResults/" folder, which contains:
- "PlotRegretComm\_[startTime]\_[namelabel].png": plot of accumulated regret over iteration and accumulated communication cost (measured by total number of integers or real numbers transferred across the learning system) for each algorithm
- "AccRegret\_[startTime]\_[namelabel].csv": cumulative regret at each time step for each algorithm
- "AccCommCost\_[startTime]\_[namelabel].csv": cumulative communication cost at each time step for each algorithm
- "ParameterEstimation\_[startTime]\_[namelabel].csv": l2 norm between estimated and ground-truth parameter at each time step for each algorithm

For experiments on MovieLens dataset, 
- First download the movielens 20m dataset from https://grouplens.org/datasets/movielens/ and put the extracted cvs files under the "./Dataset/ml-20m/raw\_data/" folder
- Then pre-process the dataset to simulate contextual bandit environment as described in Section 5. Example scripts are provided under "./Dataset folder", which are used to create processed files for event sequence and tf-idf arm feature under the "./Dataset/processed\_data/" folder.
- Run experiments on MovieLens dataset by
```console
python RunRealworldDataset.py
```