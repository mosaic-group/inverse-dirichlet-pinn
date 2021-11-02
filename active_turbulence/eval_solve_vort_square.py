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

from pinn import *
from pinnutils import *
    
class Sin(nn.Module):
    
    def forward(self, x):
        return torch.sin(x)

device = torch.device("cpu")
print(device)

ll     = np.int(sys.argv[1])       # num layer
nl     = np.float(sys.argv[2])     # noise level
seed   = np.int(sys.argv[3])       # seed
dms    = np.int(sys.argv[4])       # domain size
method = np.int(sys.argv[5])
tol    = np.float(sys.argv[6])

app = ""
if len(sys.argv) > 7:
    app   = str(sys.argv[7])

np.random.seed(seed)
torch.manual_seed(seed)

print("running active turbulence parameter inference...")
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

# setup training data for interior points / exclude boundary points
X       = np.concatenate([xx[:,1:-1,1:-1].reshape(-1)[:,None], yy[:,1:-1,1:-1].reshape(-1)[:,None], tt[:,1:-1,1:-1].reshape(-1)[:,None]], 1)
X_train = torch.tensor(X, dtype=torch.float32)

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

TD = np.concatenate([X, XB, X0], 0)

# compute mean and std of training data
X_mean = torch.tensor(np.mean(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)

X       = np.concatenate([xx.reshape(-1)[:,None], yy.reshape(-1)[:,None], tt.reshape(-1)[:,None]], 1)
X_train = torch.tensor(X, dtype=torch.float32)

def compute_vort(pred, data):
    out_shape = pred[:, None, 0]

    uh = pred[:, None, 0]
    vh = pred[:, None, 1]

    # vorticity
    du = grad(outputs=uh, inputs=data, grad_outputs=torch.ones_like(out_shape), retain_graph=True)[0].detach()

    dudx = du[:,0:1]
    dudy = du[:,1:2]

    dv = grad(outputs=vh, inputs=data, grad_outputs=torch.ones_like(out_shape), retain_graph=True)[0].detach()

    dvdx = dv[:,0:1]
    dvdy = dv[:,1:2]

    wh = dvdx - dudy

    uh = uh.cpu().detach().data.numpy()
    vh = vh.cpu().detach().data.numpy()
    wh = wh.cpu().detach().data.numpy()
    
    return uh, vh, wh

def compute_div(pred, data):
    out_shape = pred[:, None, 0]

    uh = pred[:, None, 0]
    vh = pred[:, None, 1]

    # vorticity
    du = grad(outputs=uh, inputs=data, grad_outputs=torch.ones_like(out_shape), retain_graph=True)[0].detach()

    dudx = du[:,0:1]
    dudy = du[:,1:2]

    dv = grad(outputs=vh, inputs=data, grad_outputs=torch.ones_like(out_shape), retain_graph=True)[0].detach()

    dvdx = dv[:,0:1]
    dvdy = dv[:,1:2]

    div = dudx + dvdy
    div = div.cpu().detach().data.numpy()
    
    return div

def get_model_name(seed,ll,suff,nl,dms,method,tol,app):
    model_name = ""
    if method == 3:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_eps{}_surya_bc_nown_upt".format(seed,ll,suff,nl,dms,method,tol,app)
    else:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_surya_bc_nown".format(seed,ll,suff,nl,dms,method,app)
    
    return model_name

# load model
layers = [3] + ll*[100] + [2]
net = PINNNoWN(layers, mean=X_mean, std=X_std, seed=seed, activation=Sin()).to(device)

suff       = "full"
model_name = get_model_name(seed,ll,suff,nl,dms,method,tol,app)
checkpoint = torch.load("models/" + model_name + ".pth", map_location=device)

net.load_state_dict(checkpoint["model"])

# write out results on full interpolation data
X_in = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
out_full = net(X_in)

uhf, vhf, whf = compute_vort(out_full, X_in)

save_dict = {}
save_dict["u_full"] = uhf.reshape(un.shape)
save_dict["v_full"] = vhf.reshape(un.shape)
save_dict["w_full"] = whf.reshape(un.shape)

#divf = compute_div(out_full, X_in)
#save_dict["div_full"] = divf.reshape(un.shape)

# write train and test results over different epochs
u_train_err = []
v_train_err = []
w_train_err = []
p_train_err = []
div_train_err = []

for epoch in range(100, 7001, 100):
    # load model for given epoch
    suff = "ep" + str(epoch) if epoch < 7000 else "full"
    model_name = get_model_name(seed,ll,suff,nl,dms,method,tol,app)
    checkpoint = torch.load("models/" + model_name + ".pth", map_location=device)

    net.load_state_dict(checkpoint["model"])
    
    # training errors
    out_train = net(X_train.requires_grad_(True))
    
    uht, vht, wht =  compute_vort(out_train, X_train)
    
    ut_err = np.linalg.norm(uht - u.reshape(-1)[:,None])/np.linalg.norm(u.reshape(-1)[:,None])
    vt_err = np.linalg.norm(vht - v.reshape(-1)[:,None])/np.linalg.norm(v.reshape(-1)[:,None])
    wt_err = np.linalg.norm(wht - w.reshape(-1)[:,None])/np.linalg.norm(w.reshape(-1)[:,None])
    
    #phtm = p_norm(pht)
    #pt   = p_norm(p_train)
    
    #pt_err = np.linalg.norm(phtm - pt)/np.linalg.norm(pt)
    
    u_train_err.append(ut_err)
    v_train_err.append(vt_err)
    w_train_err.append(wt_err)
    #p_train_err.append(pt_err)
    
    divt = compute_div(out_train, X_train)
    divt_err = np.linalg.norm(divt)
    
    divt_m_err = np.mean((divt)**2)
    
    div_train_err.append(divt_m_err)
    
    # write out lambdas losses
    if epoch == 7000:
        data = sio.loadmat("results/" + model_name)
        save_dict["lambs"] = data["lambs"]
        save_dict["lu"] = data["lu"]
        save_dict["lb"] = data["lb"]
        save_dict["lr"] = data["lr"]
        save_dict["ld"] = data["lrd"]
        
        #if "param" in app:
        #    save_dict["ps"] = data["ps"]

        #if "div" in app:
        #    save_dict["lrd"] = data["lrd"]
            
# write out train and validation errors
save_dict["train_u"] = u_train_err
save_dict["train_v"] = v_train_err
save_dict["train_w"] = w_train_err
save_dict["train_div"] = div_train_err

save_name = ""
if method == 3:
    save_name = "evals/solve_vv/eval_s{}_l{}_nl{}_dms{}_meth{}_eps{}{}".format(seed,ll,nl,dms,method,tol,app)
else:
    save_name = "evals/solve_vv/eval_s{}_l{}_nl{}_dms{}_meth{}{}".format(seed,ll,nl,dms,method,app)
        
sio.savemat(save_name, save_dict)
