# KBCS_SC42050_BCO
Imitation learning technique’s aim is for robots to mimic expert techniques to perform a certain task. One of the main obstacles in imitation learning, is that it is expensive/not always possible to infer the exact actions taken by the expert. An approach to circumvent this bottleneck was proposed by Faraz Torabi, Garrett Warnell and Peter Stone in their paper\cite{torabi2018behavioral} “Behavioural Cloning from Observation (2018)”. This method uses an additional step where the robot explores the environment and trains to recognise the inverse dynamics (action which causes change from one state to another). This is then utilized by the robot to infer the missing actions of the expert, and thus learn the expert trajectories. In this work, an attempt is made to reproduce the results of this paper and discuss the algorithm while considering possible alternatives to improve the method.

## Python Dependencies:  
-Matplotlib  
-OS  
-TQDM  
-Pytorch
-Gym  
-Pickle  
-Mujoco  
-Numpy 1.20.2

## Scripts
The BCO models for Ant-v2, Reacher-v2 and Mountain Car-v0 environments are implemented in their own python files (Jupyter Notebooks are also provided under "/Python Notebooks/"). Cartpole implementation is provided as Jupyter Notebook alone in the immediate directory.

## BCO Configuration
To optimize the execution of algorithm, the parameters and hyperparameters can be tuned at the beginning of every script provided. 
```
## BCO(alpha) Parameters
alpha = 1e-2
alpha_iterations = 20000
M = 5000
policy_best = 10            ## Randomly initiate best validation loss for BCO(alpha) policy
policy_patience = 10        ## Setup Patience for validation loss to stop BCO(alpha)
policy_patience_cnt = 0

## Variables for Plot
store_ls_ep = []
store_ls_val_ep = []
epoch_cumulative = 0

## Hyper Params for NN Training
epochs = 1000
patience = 100
```
In ant.py, additional variables are facilitated to make use of previously generated exploration data and model trained in inverse dynamics. Change the flag values to make use of such data/ model(s).
```
## Change the flag to 0 to use data of saved exploration phase created using random policy
data_flag = -1

## Change the flag to 0 to use pre trained ID model
model_flag = -1
```
In order use this fearture for ant.py, download the data file from [here](https://drive.google.com/file/d/1xGtZBdC4JUzr_VNKhjZrGhblIpnq5QtV/view?usp=sharing). Place the downloaded .npy file in the immediate directory.

## Testing
In order to run one of the models:  
```
python3 reacher.py
```
