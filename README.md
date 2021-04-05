# KBCS_SC42050_BCO

Python Dependencies:  
-Numpy: 1.20.2
-Time  
-Matplotlib  
-OS  
-TQDM  
-Pytorch
-Gym  
-Pickle  
-Mujoco  

# Scripts
The BCO models for Ant-v2, Reacher-v2 and Mountain Car-v0 environments are implemented in their own python files (Jupyter Notebooks are also provided under "/Python Notebooks"). Cartpole implementation is provided as Jupyter Notebook alone in the immediate directory.

# BCO Configuration
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

# Testing
In order to run one of the models:  
```
python3 reacher.py
```
