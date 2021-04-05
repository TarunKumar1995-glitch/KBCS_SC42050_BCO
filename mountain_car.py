##########################################################################################################################################
##########################################################################################################################################
## The script has been developed as a part of BCO reproducibility project for the Masters Course "Knowledge Based Control Systems"
## at TU Delft.
##
## Authors:
## Srimannarayana Baratam
## Anish Sridharan
## Jeroen Zwanepol
## Iva Surana
##########################################################################################################################################
##########################################################################################################################################

import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tqdm import trange
import numpy as np
import time
import torch as T
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch
import gym
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

env = gym.make('MountainCar-v0')

random_seed = 42

#######################################################################################################################
#######################################################################################################################

## BCO(alpha) Parameters
alpha = 1e-2
alpha_iterations = 10000
M = 2000
policy_best = 10            ## Randomly initiate best validation loss for BCO(alpha) policy
policy_patience = 10        ## Setup Patience for validation loss to stop BCO(alpha)
policy_patience_cnt = 0

## Variables for Plot
store_ls_ep = []
store_ls_val_ep = []
epoch_cumulative = 0

## Hyper Params for NN Training
epochs = 20000
patience = 500
optim_ID = 5e-4
optim_Pol = 8e-4

#######################################################################################################################
#######################################################################################################################

## Classes for creating Datasets 
class DS_Inv(Dataset):
    def __init__(self, trajs):
        self.dat = []
        
        for traj in trajs:
            for dat in traj:
                obs, act, new_obs = dat
                
                self.dat.append([obs, new_obs, act])
    
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, idx):
        obs, new_obs, act = self.dat[idx]
        
        return obs, new_obs, act

class DS_Policy(Dataset):
    def __init__(self, traj):
        self.dat = []
        
        for dat in traj:
            obs, act = dat
                
            self.dat.append([obs, act])
    
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, idx):
        obs, act = self.dat[idx]
        
        return obs, act

## Create a BCO Class to define the Neural Nets for Inverse Dynamics and Policy models
class BCO(nn.Module):
    def __init__(self, env, policy='mlp'):
        super(BCO, self).__init__()
        
        self.policy = policy
        self.act_n = env.action_space.n
        
        if self.policy=='mlp':
            self.obs_n = env.observation_space.shape[0]

            self.pol = nn.Sequential(*[nn.Linear(self.obs_n, 8), nn.LeakyReLU(),
                                       nn.Linear(8, 8), nn.LeakyReLU(),
                                       nn.Linear(8, self.act_n)])

            self.inv = nn.Sequential(*[nn.Linear(self.obs_n*2, 8), nn.LeakyReLU(),
                                       nn.Linear(8, 8), nn.LeakyReLU(),
                                       nn.Linear(8, self.act_n)])
        
        elif self.policy=='cnn':
            pass
    
    def pred_act(self, obs):
        out = self.pol(obs)
        
        return out
    
    def pred_inv(self, obs1, obs2):
        obs = T.cat([obs1, obs2], dim=1)
        out = self.inv(obs)
        
        return out

## Data Loader
def train_valid_loader(dataset, batch_size, validation_split, shuffle_dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

## Function to Train ID and Policy NN Models
def train_NN(train_loader, NN):
    
    with tqdm(train_loader, desc='Training',  disable = True, position=0, leave=True) as TQ:
        ls_ep = 0
        correct = 0
        total = 0
        
        if (NN == 'inv'):
            for obs1, obs2, act in TQ:
                out = model.pred_inv(obs1.float().cuda(), obs2.float().cuda())
                ls_bh = loss_func(out, act.cuda())
                
                optim.zero_grad()
                ls_bh.backward()
                optim.step()

                ls_bh = ls_bh.cpu().detach().numpy()
                TQ.set_postfix(loss_policy='%.8f' % (ls_bh))
                ls_ep += ls_bh
                total += obs1.shape[0]
                
        elif(NN == 'pred'):
            for obs, act in TQ:
                out = model.pred_act(obs.float().cuda())
                out_act = torch.argmax(out.cpu().detach(), axis=1)
                ls_bh = loss_func(out, act.cuda())

                optim.zero_grad()
                ls_bh.backward()
                optim.step()

                ls_bh = ls_bh.cpu().detach().numpy()
                TQ.set_postfix(loss_policy='%.8f' % (ls_bh))
                ls_ep += ls_bh
                total += obs.shape[0]
            
        ls_ep /= len(TQ)
        
    return ls_ep

## Function to Validate ID and Policy NN Models
def validate_NN(validation_loader, NN):
    
    with tqdm(validation_loader, desc='Validate', disable = True, position=0, leave=True) as TQ:
        ls_val_ep = 0
        correct = 0
        total = 0
        
        if (NN == 'inv'):
            for obs1, obs2, act in TQ:
                out = model.pred_inv(obs1.float().cuda(), obs2.float().cuda())
                ls_bh = loss_func(out, act.cuda())
                ls_bh = ls_bh.cpu().detach().numpy()
                TQ.set_postfix(loss_policy='%.8f' % (ls_bh))
                ls_val_ep += ls_bh
                total += obs1.shape[0]
        elif (NN == 'pred'):
            for obs, act in TQ:
                out = model.pred_act(obs.float().cuda())
                ls_bh = loss_func(out, act.cuda())
                ls_bh = ls_bh.cpu().detach().numpy()
                TQ.set_postfix(loss_policy='%.8f' % (ls_bh))
                ls_val_ep += ls_bh
                total += obs.shape[0]
            
        ls_val_ep /= len(TQ)
        
        return ls_val_ep

## Initiate Class Instance of BCO
POLICY = 'mlp'
model = BCO(env, policy=POLICY).cuda()

## Load Expert Demos from Demo Flolder
trajs_demo = pickle.load(open('Demo/demo_mountaincar.pkl', 'rb'))
ld_demo = DataLoader(DS_Inv(trajs_demo), batch_size=5, drop_last=True, shuffle = True)

## Loss Function and Optimizer
loss_func = nn.CrossEntropyLoss().cuda()
optim = T.optim.Adam(model.parameters(), lr=5e-4)


tqdm_alpha = trange(alpha_iterations+1, position=0, desc='alpha:', leave=True)

for e in tqdm_alpha:
    # step1, generate inverse samples
    
    trajs_inv = []
    tqdm_alpha.set_description("alpha_iteration: %i, Step1: Exploration" % e,refresh=True)
    time.sleep(1)
    cnt = 0 #count
    epn = 0 #Episode number

    rews = 0 #Rewards

    while True:
        traj = []
        rew = 0
        N=0 
        obs = env.reset()
        while True:
            inp = T.from_numpy(obs).view(((1, )+obs.shape)).float().cuda()
            out = model.pred_act(inp).cpu().detach().numpy()
            if e==0:
                act = env.action_space.sample()               
            else:
                act = np.argmax(out, axis=1)[0]


            new_obs, r, done, _ = env.step(act)

            traj.append([obs, act, new_obs])
            obs = new_obs
            rew += r

            cnt += 1
            tqdm_alpha.set_description("alpha_iteration: %i, Step1: Exploration - %i" % (e,cnt),refresh=True)
            N+=1   
            if done==True or cnt >= M:
                rews += rew
                trajs_inv.append(traj)

                epn += 1

                break

        if cnt >= M:
            break

    rews /= epn
    tqdm_alpha.set_description("alpha_iteration: %i, step1: Exploration, Reward: %.2f" % (e,rews),refresh=True)
    time.sleep(1)
      
    
    # step2, update inverse model

    ls_val_best = 5
    patience_cnt = 0
    optim = T.optim.Adam(model.parameters(), lr=optim_ID)
    tqdm_alpha.set_description("alpha_iteration: %i, Step2: Update Inverse Model" % e,refresh=True)
    time.sleep(1)
    tqdm_epoch = trange(epochs, position=0, desc='Epoch:', leave=True)


    for i in  tqdm_epoch:
        dataset=DS_Inv(trajs_inv)
        train_loader, validation_loader = train_valid_loader(dataset, batch_size=8, 
                                                                validation_split=0.3,
                                                                shuffle_dataset=True)
        
        ls_ep = train_NN(train_loader, NN = 'inv')
        ls_val_ep = validate_NN(validation_loader, NN = 'inv')
        
        tqdm_epoch.set_description("ID Model Update - Epoch: %i, val loss: %.11f" % (i,ls_val_ep),refresh=True)
        
        if ls_val_ep < ls_val_best:
            ls_val_best = ls_val_ep
            patience_cnt = 0
    
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                break
    

    # step3, predict actions for demo trajectories
    traj_policy = []
    tqdm_alpha.set_description("alpha_iteration: %i, Step3: Predict most probable actions for expert demos" % e,refresh=True)
    tqdm_alpha.set_description("alpha_iteration: %i, Policy Patience Counter: %i" % (e,policy_patience_cnt),refresh=True)
    obs_cnt = 0
    for obs1, obs2, _ in ld_demo:
        out = model.pred_inv(obs1.float().cuda(), obs2.float().cuda())
        obs = obs1.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        out = np.argmax(out, axis=1)
        for i in range(len(obs1)):
            traj_policy.append([obs[i], out[i]])
        obs_cnt+=1
        if obs_cnt==4:
            print("demo break")

            break

    # step4, update policy via demo samples
    ls_val_best = 5
    patience_cnt = 0
    optim = T.optim.Adam(model.parameters(), lr=optim_Pol)
    tqdm_alpha.set_description("alpha_iteration: %i, Step4: Update Policy" % e,refresh=True)
    tqdm_epoch = trange(epochs, position=0, desc='Epochs', leave=True)

    for i in  tqdm_epoch:
        dataset=DS_Policy(traj_policy)
        train_loader, validation_loader = train_valid_loader(dataset, batch_size=5, 
                                                             validation_split=0.3,
                                                             shuffle_dataset=True)
        
        ls_ep = train_NN(train_loader, NN = 'pred')
        ls_val_ep = validate_NN(validation_loader, NN = 'pred')

        store_ls_ep.append(ls_ep)
        store_ls_val_ep.append(ls_val_ep)
        
        tqdm_epoch.set_description("Policy Update - Epoch: %i, val loss: %.8f" % (i,ls_val_ep),refresh=True)
        
        if ls_val_ep < ls_val_best:
            ls_val_best = ls_val_ep
            patience_cnt = 0
            T.save(model.state_dict(), 'Model/model_mountain_car.pt')
    
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                break

    # step5, save model
    if ls_val_ep < policy_best:
        policy_best = ls_val_ep
        policy_patience_cnt = 0
        T.save(model.state_dict(), 'Model/model_mountain_car_best.pt')

    else:
        policy_patience_cnt += 1
    
    epoch_cumulative += i+1
    if policy_patience_cnt==policy_patience:
        break

    if e==0:    
        M *= alpha
        optim_ID = 5e-5
        optim_Pol = 8e-5


## Visualization of Policy Learning
x_ax = np.arange(1,epoch_cumulative + 1)
plt.plot(x_ax, store_ls_ep, color = 'b', label="Training Loss")
plt.plot(x_ax, store_ls_val_ep, color = 'r', label="Validation Loss")
plt.savefig("Mountain_car_BCO_demo10.svg", format='svg', dpi=1200)
plt.show()


## Evaluation over 5000 Trajectories

## Load the best model saved while training for evaluation
model = BCO(env, policy=POLICY).cuda()
model.load_state_dict(torch.load('Model/model_mountain_car_best.pt'))

reward = 0
reward_per_obs=np.array([])
episodes = 5000


tqdm_episodes = trange(episodes, position=0, desc='Episode', leave=True)

for i_episode in tqdm_episodes:
    observation = env.reset()
    rews=0
    t=0
    while True:
        inp = T.from_numpy(observation).view(((1, )+observation.shape)).float().cuda()
        out = model.pred_act(inp).cpu().detach().numpy()
        act = np.argmax(out, axis=1)[0]
#         env.render()
        observation, reward, done, info = env.step(act)
        rews+=reward
        t+=1

        if done:
            tqdm_episodes.set_description("Episode: %i, Reward: %i" % (i_episode+1,rews),refresh=True)
            reward_per_obs=np.append(reward_per_obs,rews)
            break
print("Mean Reward: ",np.mean(reward_per_obs))
