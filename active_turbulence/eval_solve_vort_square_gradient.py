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

from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')

from pinnutils import *

# forward problem in VV form without weight norm and stacked annealing methods

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
tol    = np.float(sys.argv[6])

np.random.seed(seed)
torch.manual_seed(seed)

print("running active turbulence solution approximation...")
print("parameters", "seed", seed, "layer", ll, "noise", nl, "dms", dms, "method", method)

data = sio.loadmat("../data/meso_bacterial_turbulence_IFRK4_N{}_2pi_dt0p01_Tmax10.mat".format(dms))

# updated sampling radius for equivalent number of points
cut_lb = 64
cut_ub = 128

interpol = 50

ts = 420
te = ts + interpol

u = np.transpose(data["store_data_u"], (2,0,1))[ts:te, cut_lb:cut_ub, cut_lb:cut_ub]
v = np.transpose(data["store_data_v"], (2,0,1))[ts:te, cut_lb:cut_ub, cut_lb:cut_ub]

dt    = 0.01
t_min = ts * dt
t_max = te * dt

t  = np.arange(t_min, t_max, dt)
tt = np.zeros(u.shape)
for i in range(u.shape[0]):
    tt[i,:,:] = np.full(u.shape[1:], t[i])
    
xx  = np.zeros(u.shape)
yy  = np.zeros(u.shape)

N = dms; domain_size = 2.0*np.pi; h = (domain_size)/N;
dom = [h*i for i in range(1,N+1)]

#dom = np.linspace(0, 2*np.pi, dms)
#print(len(dom))

dom_x = dom[cut_lb:cut_ub]
dom_y = dom[cut_lb:cut_ub]
for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        xx[i,:,j] = dom_x
        yy[i,j,:] = dom_y
        
def rc(inp):
    return inp.reshape(-1)[:,None]

# add noise to training data
np.random.seed(seed)
un = u
vn = v

# setup training data
X       = np.concatenate([xx[:,1:-1,1:-1].reshape(-1)[:,None], yy[:,1:-1,1:-1].reshape(-1)[:,None], tt[:,1:-1,1:-1].reshape(-1)[:,None]], 1)
X_train = torch.tensor(X, dtype=torch.float32)
print("#samples", X_train.shape[0])

# setup boundary terms
XLbx = np.concatenate([rc(xx[:,0,:]),  rc(yy[:,0,:]),  rc(tt[:,0,:])], 1)
XRbx = np.concatenate([rc(xx[:,-1,:]), rc(yy[:,-1,:]), rc(tt[:,-1,:])], 1)
XLby = np.concatenate([rc(xx[:,:,0]),  rc(yy[:,:,0]),  rc(tt[:,:,0])], 1)
XRby = np.concatenate([rc(xx[:,:,-1]), rc(yy[:,:,-1]), rc(tt[:,:,-1])], 1)

YLbx = np.concatenate([rc(un[:,0,:]),  rc(vn[:,0,:])], 1)
YRbx = np.concatenate([rc(un[:,-1,:]), rc(vn[:,-1,:])], 1)
YLby = np.concatenate([rc(un[:,:,0]),  rc(vn[:,:,0])], 1)
YRby = np.concatenate([rc(un[:,:,-1]), rc(vn[:,:,-1])], 1)

XB = np.concatenate([XLbx, XRbx, XLby, XRby], 0)
YB = np.concatenate([YLbx, YRbx, YLby, YRby], 0)

# setup initial conditions
X0 = np.concatenate([rc(xx[0,:,:]), rc(yy[0,:,:]), rc(tt[0,:,:])], 1)
Y0 = np.concatenate([rc(un[0,:,:]), rc(vn[0,:,:])], 1)

# setup boundary and 
XB_train = torch.tensor(XB, dtype=torch.float32, device=device)
YB_train = torch.tensor(YB, dtype=torch.float32, device=device)

X0_train = torch.tensor(X0, dtype=torch.float32, device=device)
Y0_train = torch.tensor(Y0, dtype=torch.float32, device=device)

# setup boundary and 
XB_train = torch.tensor(XB, dtype=torch.float32, device=device)
YB_train = torch.tensor(YB, dtype=torch.float32, device=device)

X0_train = torch.tensor(X0, dtype=torch.float32, device=device)
Y0_train = torch.tensor(Y0, dtype=torch.float32, device=device)

# setup data loaders
batch_s      = 4000
loader       = FastTensorDataLoader(X_train, batch_size=batch_s, shuffle=True)

TD = np.concatenate([X, XB, X0], 0)

# compute mean and std of training data
X_mean = torch.tensor(np.mean(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)

def load_model(suff):
    # retrieve model name for given epoch insert (e.g., "ep5000" or "full" (last model))
    model_name = ""
    if method == 3:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_eps{}_surya_bc_nown_upt".format(seed,ll,suff,nl,dms,method,tol)
    else:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_surya_bc_nown".format(seed,ll,suff,nl,dms,method)
        
    return model_name

# setup of neural net and training loop
layers = [3] + ll*[100] + [2]
net = PINNNoWN(layers, mean=X_mean, std=X_std, seed=seed, activation=Sin()).to(device)

# setup parameters
l0   = 3.5
alp  = -1
beta = 0.5
gam0 = -0.045
gam2 = 0.045**3

params = [
    {'params': net.parameters(), 'lr': 1e-3}
]

epochs = 7_000
milestones = [[3000,6_000]]

optimizer = Adam(params)
scheduler = MultiStepLR(optimizer, milestones[0], gamma=0.1)

lamb0 = 1
lambb = 1
lambd = 1

alpha_ann = 0.5
c_ = [1,1,1,1]

mm = 5
lambs = []

grads_0 = []
grads_b = []
grads_r1 = []
grads_r2 = []

pretrain = 0

def identity(item, lambg=1.0):
    # returns identity of input
    return item

def max_aggr(item, lambg=1.0):
    # returns max of absolute gradients
    return torch.max(torch.abs(item))

def mean_aggr(item, lambg=1.0):
    # returns mean of absolute gradients
    return torch.mean(torch.abs(lambg*item))

def std_aggr(item, lambg=1.0):
    # returns std of gradients
    return torch.std(item)

def to_numpy(item):
    # converts cuda tensor to numpy array
    return item.cpu().data.numpy()

def get_gradient(loss, net, aggregator, lambg=1.0):
    # multiplies lambg to gradients and aggregates according to aggregator function
    # returns values for lambda computation and numpy array of gradients
    grads = network_gradient(loss, net)
    return aggregator(grads, lambg=lambg), to_numpy(grads)

start = time.time()
for epoch in range(100,7001,100):
    for i, X_batch in enumerate(loader):
        if i >= 1:
            # only consider first batch at epoch
            continue
        
        # load model at current epoch following naming convention
        model_name = load_model("ep"+str(epoch) if epoch < 7000 else "full")
        checkpoint = torch.load("models/" + model_name + ".pth")
        net.load_state_dict(checkpoint["model"])
        
        # load previous lambdas for STD and MaxAvg
        param_dict = sio.loadmat("results/" + model_name)
        if (method == 5) or (method == 6):
            lamb0 = param_dict["lambs"][-1][0]
            lambb = param_dict["lambs"][-1][1]
            lambd = param_dict["lambs"][-1][2]
        
        net.train()
        optimizer.zero_grad() 
        
        # initial condition error
        pred0 = net(X0_train)
        l_u = torch.sum(torch.mean((Y0_train - pred0)**2, dim=0))
        
        # compute gradients for initital condition lambda
        g0   = 0.0  # placeholder for aggregated gradients
        grd0 = None
        with torch.no_grad():
            if method == 5:
                # max avg
                g0, grd0 = get_gradient(l_u, net, mean_aggr, lambg=lamb0)
            elif method == 6:
                # std
                g0, grd0 = get_gradient(l_u, net, std_aggr)
            elif method == 3:
                # pareto
                g0, grd0 = get_gradient(l_u, net, identity)
            else:
                # only write numpy gradients for vanilla
                _, grd0 = get_gradient(l_u, net, identity)
                    
        # boundary condition error           
        predbc = net(XB_train)
        l_bc = torch.sum(torch.mean((YB_train - predbc)**2, dim=0))
        
        # compute gradients for boundary condition lambda
        gb   = 0.0  # placeholder for aggregated gradients
        grdb = None
        with torch.no_grad():
            if method == 5:
                # max avg
                gb, grdb = get_gradient(l_bc, net, mean_aggr, lambg=lambb)
            elif method == 6:
                # std
                gb, grdb = get_gradient(l_bc, net, std_aggr)
            elif method == 3:
                # pareto
                gb, grdb = get_gradient(l_bc, net, identity)
            else:
                # only write numpy gradients for vanilla
                _, grdb = get_gradient(l_bc, net, identity)
        
        # get interior points for current batch and predict
        X_batch = X_batch[0].to(device).requires_grad_(True)
        pred = net(X_batch)
        
        # compute residual
        res1, res2 = residual_vort(pred, X_batch, l0, alp, beta, gam0, gam2) 
        l_reg1 = torch.mean(res1**2)
        l_reg2 = torch.mean(res2**2)
        
        grdr1 = None  
        grdr2 = None 
        # compute lambdas
        if(method==3):
            # mgda
            with torch.no_grad():
                G_reg1, grdr1 = get_gradient(l_reg1, net, identity)
                G_reg2, grdr2 = get_gradient(l_reg2, net, identity)

                # construct gradient matrix
                M = torch.zeros((torch.numel(G_reg1), 4), dtype=torch.float32, device=device)

                M[:,0] = g0
                M[:,1] = gb
                M[:,2] = G_reg1
                M[:,3] = G_reg2

                # solve for optimal parameters
                c_ = solver_mine(torch.matmul(M.T, M).cpu().numpy(), 4, tol, maxiter=1000)
                lambs.append([c_[0].item(), c_[1].item(), c_[2].item(), c_[3].item()])

            loss = c_[0]*l_u +  c_[1]*l_bc + c_[2]*l_reg1 + c_[3]*l_reg2
        elif(method==5):
            # stacked max avg
            with torch.no_grad():
                maxr, grdr1  = get_gradient(l_reg1, net, max_aggr)
                meanr, grdr2 = get_gradient(l_reg2, net, mean_aggr, lambg=lambd)
                
                lamb_hat = maxr/g0
                lamb0    = (1-alpha_ann)*lamb0 + alpha_ann*lamb_hat

                lamb_hat = maxr/gb
                lambb    = (1-alpha_ann)*lambb + alpha_ann*lamb_hat

                lamb_hat = maxr/meanr
                lambd     = (1-alpha_ann)*lambd + alpha_ann*lamb_hat
                lambs.append([lamb0.item(), lambb.item(), lambd.item()])

            loss = lamb0*l_u + lambb*l_bc + l_reg1 + lambd*l_reg2
        elif(method==6):
            # inverse dirichlet
            with torch.no_grad():
                stdr1, grdr1 = get_gradient(l_reg1, net, std_aggr)
                stdr2, grdr2 = get_gradient(l_reg2, net, std_aggr)

                lamb_hat = stdr1/g0    
                lamb0 = (1-alpha_ann)*lamb0 + alpha_ann*lamb_hat

                lamb_hat = stdr1/gb    
                lambb = (1-alpha_ann)*lambb + alpha_ann*lamb_hat

                lamb_hat = stdr1/stdr2    
                lambd = (1-alpha_ann)*lambd + alpha_ann*lamb_hat
                lambs.append([lamb0.item(), lambb.item(), lambd.item()])

            loss = lamb0*l_u + lambb*l_bc + l_reg1 + lambd*l_reg2
        else:
            # vanilla
            with torch.no_grad():
                _, grdr1 = get_gradient(l_reg1, net, identity)
                _, grdr2 = get_gradient(l_reg2, net, identity)
            
            loss = l_u + l_bc + l_reg1 + l_reg2
        
        # multiply lambdas for different methods
        if method == 3:
            # pareto
            grd0  = grd0*c_[0].item()
            grdb  = grdb*c_[1].item()
            grdr1 = grdr1*c_[2].item()
            grdr2 = grdr2*c_[3].item()
        elif method == 5 or method == 6:
            # STD or MaxAvg
            # residual doesn't have a lambda
            grd0  = grd0*lamb0.item()
            grdb  = grdb*lambb.item()
            grdr2 = grdr2*lambd.item()
        else:
            # in case of vanilla nothing needs to be added
            pass
            
        # write out final gradients
        grads_0.append(grd0)
        grads_b.append(grdb)
        grads_r1.append(grdr1)
        grads_r2.append(grdr2)
        
        # clear graph
        loss.backward()
        optimizer.step()
        
save_dict = {
    "g0": grads_0,
    "gb": grads_b,
    "gr1": grads_r1,
    "gr2": grads_r2,
    "lambs": lambs
}

sio.savemat("evals/gradients_squared_meth{}_full".format(method), save_dict)
