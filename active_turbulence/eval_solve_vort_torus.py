import os
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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

ll     = np.int(sys.argv[1])       # num layer
nl     = np.float(sys.argv[2])     # noise level
seed   = np.int(sys.argv[3])       # seed
dms    = np.int(sys.argv[4])       # domain size
method = np.int(sys.argv[5])
tol    = np.float(sys.argv[6])

app = ""
if len(sys.argv) > 7:
    app   = str(sys.argv[7])

def create_circular_mask(height, width, radius):
    center = (int(width/2), int(height/2))
    Y, X = np.meshgrid(np.arange(height), np.arange(width))
    dist = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist <= radius
    return mask

def collect_within(height,width,r):
    # collect masks for inner and outer circles
    m1 = create_circular_mask(height,width,r[0])
    m2 = create_circular_mask(height,width,r[1])
    
    # join masks and return indeces within torus
    return ~np.logical_or(m1, ~m2)

int_r = (11,34) # interior (not including boundary)
bc_ri = (10,11) # inner boundary
bc_ro = (34,35) # outer boundary
ini_r = (10,35) # initial condition

class Sin(nn.Module):
    
    def forward(self, x):
        return torch.sin(x)

device = torch.device("cpu")

dms = 256

data = sio.loadmat("../data/meso_bacterial_turbulence_IFRK4_N{}_2pi_dt0p01_Tmax10.mat".format(dms))

cut_lb = 60
cut_ub = 132

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

def ca(arr,r):
    # get *circle array* data by applying masks for radii r
    # and reshape into column vector
    return arr[collect_within(72, 72, r)][:,None]

# full input coordinates for eval
Xf     = np.concatenate([xx.reshape(-1)[:,None], yy.reshape(-1)[:,None], tt.reshape(-1)[:,None]], 1)
X_full = torch.tensor(Xf, dtype=torch.float32, requires_grad=True)

un = u
vn = v

# radii for intirior, boundary (inner,outer) and initial conditions
int_r = (11,34) # interior (not including boundary)
bc_ri = (10,11) # inner boundary
bc_ro = (34,35) # outer boundary
ini_r = (10,35) # initial condition

# initial condition
X0 = np.concatenate([ca(xx[0], ini_r), ca(yy[0], ini_r), ca(tt[0], ini_r)], 1)
Y0 = np.concatenate([ca(un[0], ini_r), ca(vn[0], ini_r)], 1)

# interior and boundary points
# setup training data by iterating over time slices and extracting circles
X = None
XB = None
YB = None
for i in range(len(t)):
    # interior points
    xi = np.concatenate([ca(xx[i], int_r), ca(yy[i], int_r), ca(tt[i], int_r)], 1)
    
    # boundary points (inner/outer)
    xbi = np.concatenate([ca(xx[i], bc_ri), ca(yy[i], bc_ri), ca(tt[i], bc_ri)], 1)
    xbo = np.concatenate([ca(xx[i], bc_ro), ca(yy[i], bc_ro), ca(tt[i], bc_ro)], 1)
    
    xb = np.concatenate([xbi,xbo], 0)
    
    ybi = np.concatenate([ca(un[i], bc_ri), ca(vn[i], bc_ri)], 1)
    ybo = np.concatenate([ca(un[i], bc_ro), ca(vn[i], bc_ro)], 1)
    
    yb = np.concatenate([ybi,ybo], 0)
    
    if X is None:
        X  = xi
        XB = xb
        YB = yb
    else:
        # stack together over all time slices
        X  = np.concatenate([X, xi], 0)
        XB = np.concatenate([XB, xb], 0)
        YB = np.concatenate([YB, yb], 0)

X_train = torch.tensor(X, dtype=torch.float32)
print("#samples", X_train.shape[0])

# setup boundary and 
XB_train = torch.tensor(XB, dtype=torch.float32, device=device)
YB_train = torch.tensor(YB, dtype=torch.float32, device=device)

X0_train = torch.tensor(X0, dtype=torch.float32, device=device)
Y0_train = torch.tensor(Y0, dtype=torch.float32, device=device)

TD = np.concatenate([X, XB, X0], 0)

# compute mean and std of training data
X_mean = torch.tensor(np.mean(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)

def get_model_name(seed,ll,suff,nl,dms,method,tol,app):
    model_name = ""
    if method == 3:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_eps{}_surya_bc_nown_circle{}".format(seed,ll,suff,nl,dms,method,tol,app)
    else:
        model_name = "solve_vv/model_active_s{}_l{}_n100_{}_nl{}_sol_dms{}_act1_meth{}_surya_bc_nown_circle{}".format(seed,ll,suff,nl,dms,method,app)
    
    return model_name

# load model
layers = [3] + ll*[100] + [2]
net = PINNNoWN(layers, mean=X_mean, std=X_std, seed=2, activation=Sin()).to(device)

suff       = "full"
model_name = get_model_name(2,ll,suff,0.0,256,method,tol,"")
checkpoint = torch.load("models/" + model_name + ".pth", map_location=device)

net.load_state_dict(checkpoint["model"])

# write out results on full interpolation data
X_in = torch.tensor(Xf, dtype=torch.float32, device=device, requires_grad=True)
out_full = net(X_in)

uhf, vhf, whf = compute_vort_param(out_full, X_in)

uh = uhf.reshape(un.shape)
vh = vhf.reshape(un.shape)
wh = whf.reshape(un.shape)

mask = collect_within(72,72,ini_r)

save_dict = {}
save_dict["u_full"] = uh
save_dict["v_full"] = vh
save_dict["w_full"] = wh

err_u = []
err_v = []
err_w = []
for ep in range(100,7001,100):
    print(method, ep, end="\r")
    suff       = "ep"+str(ep) if ep < 7000 else "full"
    model_name = get_model_name(2,ll,suff,0.0,256,method,tol,app)

    layers = [3] + ll*[100] + [2]
    net = PINNNoWN(layers, mean=X_mean, std=X_std, seed=2, activation=Sin()).to(device)

    checkpoint = torch.load("models/" + model_name + ".pth", map_location=device)
    net.load_state_dict(checkpoint["model"])

    # write out results on full interpolation data
    X_in = torch.tensor(Xf, dtype=torch.float32, device=device, requires_grad=True)
    out_full = net(X_in)

    uhf, vhf, whf = compute_vort_param(out_full, X_in)

    uh = uhf.reshape(un.shape)
    vh = vhf.reshape(un.shape)
    wh = whf.reshape(un.shape)

    mask = collect_within(72,72,ini_r)
    
    uerr = np.linalg.norm(uh[:,mask].reshape(-1)-u[:,mask].reshape(-1))/np.linalg.norm(u[:,mask].reshape(-1))
    verr = np.linalg.norm(vh[:,mask].reshape(-1)-v[:,mask].reshape(-1))/np.linalg.norm(v[:,mask].reshape(-1))
    werr = np.linalg.norm(wh[:,mask].reshape(-1)-w[:,mask].reshape(-1))/np.linalg.norm(w[:,mask].reshape(-1))
    
    err_u.append(uerr)
    err_v.append(verr)
    err_w.append(werr)

save_dict["u_train"] = err_u
save_dict["v_train"] = err_v
save_dict["w_train"] = err_w
save_dict["mask"] = mask

save_name = ""
if method == 3:
    save_name = "evals/solve_vv/eval_s{}_l{}_nl{}_dms{}_meth{}_eps{}_surya_bc_nown_circle{}".format(seed,ll,nl,dms,method,tol,app)
else:
    save_name = "evals/solve_vv/eval_s{}_l{}_nl{}_dms{}_meth{}_surya_bc_nown_circle{}".format(seed,ll,nl,dms,method,app)
        
sio.savemat(save_name, save_dict)
    