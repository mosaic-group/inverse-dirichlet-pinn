import os
import time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn

from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset

from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')

from pinnutils import *

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    
class Sin(nn.Module):
    
    def forward(self, x):
        return torch.sin(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ll     = np.int(sys.argv[1])       # num layer
nl     = np.float(sys.argv[2])     # noise level
seed   = np.int(sys.argv[3])       # seed
dms    = np.int(sys.argv[4])       # domain size
method = np.int(sys.argv[5])
act    = np.int(sys.argv[6])

np.random.seed(seed)
torch.manual_seed(seed)

window = 100

print("running active turbulence parameter inference...")
print("parameters", "seed", seed, "layer", ll, "noise", nl, "dms", dms, "method", method, "pt", pt, "tol", tol)

data = sio.loadmat("../data/meso_bacterial_turbulence_IFRK4_N{}_2pi_dt0p01_Tmax10.mat".format(dms))

cut_lb   = 64
cut_ub   = 128
interpol = 50

ts = 420
te = ts + interpol

u = np.transpose(data["store_data_u"], (2,0,1))[ts:te, cut_lb:cut_ub, cut_lb:cut_ub]
v = np.transpose(data["store_data_v"], (2,0,1))[ts:te, cut_lb:cut_ub, cut_lb:cut_ub]
w = np.transpose(data["store_data"], (2,0,1))[ts:te, cut_lb:cut_ub, cut_lb:cut_ub]

dt    = 0.01
t_min = ts * dt
t_max = te * dt

t  = np.arange(t_min, t_max, dt)
tt = np.zeros(u.shape)
for i in range(u.shape[0]):
    tt[i,:,:] = np.full(u.shape[1:], t[i])
    
xx  = np.zeros(u.shape)
yy  = np.zeros(u.shape)

dom = np.linspace(0, 2*np.pi, dms)
print(len(dom))

dom_x = dom[cut_lb:cut_ub]
dom_y = dom[cut_lb:cut_ub]
for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        xx[i,:,j] = dom_x
        yy[i,j,:] = dom_y

# add noise to training data
np.random.seed(seed)
un = u + nl*np.std(u)*np.random.randn(*u.shape)
vn = v + nl*np.std(v)*np.random.randn(*v.shape)

# setup training data
X = np.concatenate([xx.reshape(-1)[:,None], yy.reshape(-1)[:,None], tt.reshape(-1)[:,None]], 1)
Y = np.concatenate([un.reshape(-1)[:,None], vn.reshape(-1)[:,None]], 1)

X_train = torch.tensor(X, dtype=torch.float32)
Y_train = torch.tensor(Y, dtype=torch.float32)

# setup data loaders
batch_s      = 4096
loader       = FastTensorDataLoader(X_train, Y_train, batch_size=batch_s, shuffle=True)

# compute mean and std of training data
X_mean = torch.tensor(np.mean(X, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(X, axis=0, keepdims=True), dtype=torch.float32, device=device)

def save_model(suff):
    # save model for given suff, i.e. ep1000, best_test, full, ...

    model_name = "latent/model_active_s{}_l{}_n100_{}_nl{}_param_inf_dms{}_act1_meth{}_act{}_activ_reconstruction".format(seed,ll,suff,nl,dms,method,act)
    
    torch.save({
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss
    }, "models/" + model_name + ".pth")
    
def evaluate(suff):
    net.eval()
    save_model(suff)

# setup of neural net and training loop
layers = [3] + ll*[100] + [2]

activ  = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), Sin()][act]
net    = PINNNoWN(layers, mean=X_mean, std=X_std, seed=seed, activation=activ).to(device)

params = [
    {'params': net.parameters(), 'lr': 1e-3}
]

n_param = sum(p.numel() for p in net.parameters() if p.requires_grad)

epochs = 2500

milestones = [[3000, 6000]]

optimizer = Adam(params)
scheduler = MultiStepLR(optimizer, milestones[0], gamma=0.1)

# parameters for adaptive weights
lamb  = torch.tensor(1.0, dtype=torch.float32, device=device)
lambr = torch.tensor(1.0, dtype=torch.float32, device=device)
lambd = torch.tensor(1.0, dtype=torch.float32, device=device)
c_    = np.ones(3)

mm        = 5
alpha_ann = 0.5

# inits for saving
loss  = 0.0
epoch = 0

# saves losses per batch
lambs     = []
losses_u  = []
losses_r  = []
losses_r1 = []

pretrain = pt
start    = time.time()

# initalize first evaluation
evaluate("ep0")
for epoch in range(epochs):
    # training loop
    for i, (X_batch, Y_batch) in enumerate(loader):
        net.train()
        optimizer.zero_grad()
        
        X_batch = X_batch.to(device).requires_grad_(True)
        Y_batch = Y_batch.to(device)
        
        # prediction error
        pred = net(X_batch)
        l_u  = torch.sum(torch.mean((Y_batch - pred)**2, dim=0))
        
        loss = l_u

        loss.backward()
        optimizer.step()
        
        if i == 0 and epoch % 100 == 0 and epoch > 0:
            print("epoch {}; batch {}/{}; loss {}".format(epoch, i, len(loader), round(loss.item(), 7)))
    
    scheduler.step()
         
     # evaluate and save model
    if (epoch+1) % 100 == 0:
        suff = "ep" + str(epoch+1)
        evaluate(suff)
                
print()
print("time taken:", time.time()-start)