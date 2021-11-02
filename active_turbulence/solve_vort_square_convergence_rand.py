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
frac   = np.float(sys.argv[7])

np.random.seed(seed)
torch.manual_seed(seed)

print("running active turbulence solution approximation...")
print("parameters", "seed", seed, "layer", ll, "noise", nl, "dms", dms, "method", method)

data = sio.loadmat("../data/meso_bacterial_turbulence_IFRK4_N{}_2pi_dt0p01_Tmax10.mat".format(dms))

cut_lb = 64
cut_ub = 128

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

# compute number of data points for given fraction
_fr = np.size(xx[0])/(frac**2)
print("num before", np.size(xx[0]), "num elements", _fr)

# retrieve random indices corresponding to _fr
np.random.seed(seed)
idxs = np.random.choice(np.size(xx[0,1:-1,1:-1]), np.int(_fr), replace=False)

# setup training data
Xs = None
for t_ in range(len(t)):
    sub = np.concatenate([xx[t_,1:-1,1:-1].reshape(-1)[:,None], yy[t_,1:-1,1:-1].reshape(-1)[:,None], tt[t_,1:-1,1:-1].reshape(-1)[:,None]], 1)
    if Xs is None:
        Xs = sub[idxs]
    else:
        Xs = np.concatenate([Xs, sub[idxs]], 0)
    
X_train = torch.tensor(Xs, dtype=torch.float32)
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

# setup data loaders
batch_s      = 4000
loader       = FastTensorDataLoader(X_train, batch_size=batch_s, shuffle=True)

TD = np.concatenate([Xs, XB, X0], 0)

# compute mean and std of training data
X_mean = torch.tensor(np.mean(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)

# full input data for evaluation
Xf = np.concatenate([xx.reshape(-1)[:,None], yy.reshape(-1)[:,None], tt.reshape(-1)[:,None]], 1)

def save_model(suff):
    # save model for given suff, i.e. ep1000, best_test, full, ...
    model_name = ""
    if method == 3:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_eps{}_frac{}_surya_bc_nown_conv_equal_rand".format(seed,ll,suff,nl,dms,method,tol,frac)
    else:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_frac{}_surya_bc_nown_conv_equal_rand".format(seed,ll,suff,nl,dms,method,frac)
        
    torch.save({
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
    }, "models/" + model_name + ".pth")

    mdict = {
        "epoch": epoch,
        "lambs": lambs,
        "lu": losses_u,
        "lb": losses_b,
        "lr": losses_r,
        "lrd": losses_r1,
        "t": (time.time()-start)
    }
    sio.savemat("results/" + model_name, mdict)
    
def get_eval_name():
    save_name = ""
    if method == 3:
        save_name = "evals/solve_vv/eval_s{}_l{}_nl{}_dms{}_meth{}_eps{}_frac{}_surya_bc_nown_conv_equal_rand".format(seed,ll,nl,dms,method,tol,frac)
    else:
        save_name = "evals/solve_vv/eval_s{}_l{}_nl{}_dms{}_meth{}_frac{}_surya_bc_nown_conv_equal_rand".format(seed,ll,nl,dms,method,frac)
        
    return save_name

def evaluate(suff):
    net.eval()
    
    # determine if first evaluation
    isfirst = False
    if suff is not "full":
        isfirst = int(suff[2:]) == 0
        
    # load evaluation dict
    mdict = {}
    if not isfirst:
        mdict = sio.loadmat(get_eval_name())
        
    # override current reconstruction of full data
    X_full        = torch.tensor(Xf, dtype=torch.float32, device=device, requires_grad=True)
    out_full      = net(X_full)
    uhf, vhf, whf = compute_vort_param(out_full, X_full)
    
    # evaluate error
    ut_err  = np.linalg.norm(uhf - u.reshape(-1)[:,None])/np.linalg.norm(u.reshape(-1)[:,None])
    vt_err  = np.linalg.norm(vhf - v.reshape(-1)[:,None])/np.linalg.norm(v.reshape(-1)[:,None])
    wt_err  = np.linalg.norm(whf - w.reshape(-1)[:,None])/np.linalg.norm(w.reshape(-1)[:,None])
    
    # read previous evaluations
    uerrs   = mdict["train_u"][0].tolist() if not isfirst else []
    verrs   = mdict["train_v"][0].tolist() if not isfirst else []
    werrs   = mdict["train_w"][0].tolist() if not isfirst else []
    
    # append current update
    uerrs.append(ut_err)
    verrs.append(vt_err)
    werrs.append(wt_err)
    
    # write errors to file
    mdict["train_u"] = uerrs
    mdict["train_v"] = verrs
    mdict["train_w"] = werrs
    
    # write out full reconstruction snapshots
    mdict["u_full"] = uhf.reshape(u.shape)
    mdict["v_full"] = vhf.reshape(u.shape)
    mdict["w_full"] = whf.reshape(u.shape)
    
    # save updated dictionary
    sio.savemat(get_eval_name(), mdict)
    
    # save model checkpoint and remaining stuff, e.g. losses, lambdas, ...
    save_model(suff)

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
lambr = 1
lambd = 1
alpha_ann = 0.5
alpha_reg = 0.5
c_ = [1,1,1,1]
mm = 5
lambs = []

loss = 0.0
epoch = 0

# saves estimated parameters every 100 epochs
ps = []

# saves losses per batch
losses_u = []
losses_b = []
losses_r = []
losses_r1 = []

times = [[],[],[]]

pretrain = 0

start = time.time()
evaluate("ep0")
for epoch in range(epochs):
    for i, X_batch in enumerate(loader):
        net.train()
        optimizer.zero_grad() 
        
        # initial condition error
        pred0 = net(X0_train)
        start_fit = time.time()
        l_u = torch.sum(torch.mean((Y0_train - pred0)**2, dim=0))
        times[0].append(time.time()-start_fit)
        
        # compute gradients for initital condition
        stdu = 0.0
        with torch.no_grad():
            if epoch % mm == 0 and i == 0:
                if method == 5:
                    _, stdu = loss_grad_max_full(l_u, net, lambg=lamb0)
                elif method == 6 or method == 7:
                    stdu = loss_grad_std_full(l_u, net)
                elif method == 3:
                    stdu = network_gradient(l_u, net)
                    
        # boundary condition error           
        predbc = net(XB_train)
        start_bc = time.time()
        l_bc = torch.sum(torch.mean((YB_train - predbc)**2, dim=0))
        times[1].append(time.time()-start_bc)
        
        # compute gradients for boundary condition
        stdb = 0.0
        with torch.no_grad():
            if epoch % mm == 0 and i == 0:
                if method == 5:
                    _, stdb = loss_grad_max_full(l_bc, net, lambg=lambb)
                elif method == 6 or method == 7: 
                    stdb = loss_grad_std_full(l_bc, net)
                elif method == 3:
                    stdb = network_gradient(l_bc, net)
        
        l_reg1 = 0
        l_reg2 = 0
        if epoch >= pretrain:
            # turn on only after initial data fitting phase
            X_batch = X_batch[0].to(device).requires_grad_(True)
            
            pred = net(X_batch)
            
            start_reg=time.time()
            res1, res2 = residual_vort(pred, X_batch, l0, alp, beta, gam0, gam2) 
            l_reg1 = torch.mean(res1**2)
            l_reg2 = torch.mean(res2**2)
            times[2].append(time.time()-start_reg)
            
            if(method==3):
                # mgda
                with torch.no_grad():
                    if epoch% mm == 0 and i == 0:
                        G_reg1 = network_gradient(l_reg1, net)
                        G_reg2 = network_gradient(l_reg2, net)
                        
                        # construct gradient matrix
                        M = torch.zeros((torch.numel(G_reg1), 4), dtype=torch.float32, device=device)

                        M[:,0] = stdu
                        M[:,1] = stdb
                        M[:,2] = G_reg1
                        M[:,3] = G_reg2

                        # solve for optimal parameters
                        c_ = solver_mine(torch.matmul(M.T, M).cpu().numpy(), 4, tol, maxiter=1000)
                        lambs.append([c_[0].item(), c_[1].item(), c_[2].item(), c_[3].item()])
                        
                loss = c_[0]*l_u +  c_[1]*l_bc + c_[2]*l_reg1 + c_[3]*l_reg2
            elif(method==5):
                # max avg
                with torch.no_grad():
                    if epoch % mm == 0 and i == 0:
                        maxr, _ = loss_grad_max_full(l_reg1, net)
                        _, meanr = loss_grad_max_full(l_reg2, net, lambg=lambd)
                        lamb_hat = maxr/stdu
                        lamb0    = (1-alpha_ann)*lamb0 + alpha_ann*lamb_hat
                        
                        lamb_hat = maxr/stdb
                        lambb    = (1-alpha_ann)*lambb + alpha_ann*lamb_hat
                        
                        lamb_hat = maxr/meanr
                        lambd     = (1-alpha_ann)*lambd + alpha_ann*lamb_hat
                        lambs.append([lamb0.item(), lambb.item(), lambd.item()])
                        
                loss = lamb0*l_u + lambb*l_bc + l_reg1 + lambd*l_reg2
            elif(method==6):
                # inverse dirichlet / std
                with torch.no_grad():
                    if epoch % mm == 0 and i == 0:
                        stdr1 = loss_grad_std_full(l_reg1, net)
                        stdr2 = loss_grad_std_full(l_reg2, net)
                        
                        lamb_hat = stdr1/stdu    
                        lamb0 = (1-alpha_ann)*lamb0 + alpha_ann*lamb_hat
                        
                        lamb_hat = stdr1/stdb    
                        lambb = (1-alpha_ann)*lambb + alpha_ann*lamb_hat
                        
                        lamb_hat = stdr1/stdr2    
                        lambd = (1-alpha_ann)*lambd + alpha_ann*lamb_hat
                        lambs.append([lamb0.item(), lambb.item(), lambd.item()])
                        
                loss = lamb0*l_u + lambb*l_bc + l_reg1 + lambd*l_reg2
            else:
                # vanilla
                loss = l_u + l_bc + l_reg1 + l_reg2
        
        # write out losses
        losses_u.append(l_u.item())
        losses_b.append(l_bc.item())
        losses_r.append(0 if isinstance(l_reg1, int) else l_reg1.item())
        losses_r1.append(0 if isinstance(l_reg2, int) else l_reg2.item())

        loss.backward()
        optimizer.step()
        
        if i == 0 and epoch % 100 == 0 and epoch > 0:
            print("epoch {}; batch {}/{}; loss {}".format(epoch, i, len(loader), round(loss.item(), 7)))
    
    scheduler.step()
    
    # evaluate and save model
    if (epoch+1) % 100 == 0:
        suff = "full" if epoch+1 == epochs else "ep" + str(epoch+1)
        evaluate(suff)
                
print()
print("time taken:", time.time()-start)
