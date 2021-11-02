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

import sys
sys.path.insert(0, '../')

from pinnutils import *

device = torch.device("cpu")

ll     = np.int(sys.argv[1])       # num layer
nl     = np.float(sys.argv[2])     # noise level
seed   = np.int(sys.argv[3])       # seed
dms    = np.int(sys.argv[4])       # domain size
method = np.int(sys.argv[5])       # not used
act    = np.int(sys.argv[6])       # activation

# setup data
data = sio.loadmat("../data/meso_bacterial_turbulence_IFRK4_N256_2pi_dt0p01_Tmax10.mat")

cut_lb = 64
cut_ub = 128

u = np.transpose(data["store_data_u"], (2,0,1))[420:470, cut_lb:cut_ub, cut_lb:cut_ub]
v = np.transpose(data["store_data_v"], (2,0,1))[420:470, cut_lb:cut_ub, cut_lb:cut_ub]
w = np.transpose(data["store_data"], (2,0,1))[420:470, cut_lb:cut_ub, cut_lb:cut_ub]

dt    = 0.01
t_max = len(u)*dt

t  = np.arange(420*dt, 470*dt, dt)
tt = np.zeros(u.shape)
for i in range(u.shape[0]):
    tt[i,:,:] = np.full(u.shape[1:], t[i])

xx  = np.zeros(u.shape)
yy  = np.zeros(u.shape)

dom = np.linspace(0,2*np.pi,256)
dom_x = dom[cut_lb:cut_ub]
dom_y = dom[cut_lb:cut_ub]
for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        xx[i,:,j] = dom_x
        yy[i,j,:] = dom_y

# add noise to training data
np.random.seed(1)
un = u + 0.0*np.std(u)*np.random.randn(*u.shape)
vn = v + 0.0*np.std(v)*np.random.randn(*v.shape)

print(un.shape)

# setup training data
X = np.concatenate([xx.reshape(-1)[:,None], yy.reshape(-1)[:,None], tt.reshape(-1)[:,None]], 1)
Y = np.concatenate([un.reshape(-1)[:,None], vn.reshape(-1)[:,None]], 1)

X_mean = torch.tensor(np.mean(X, axis=0, keepdims=True), dtype=torch.float32)
X_std  = torch.tensor(np.std(X, axis=0, keepdims=True), dtype=torch.float32)

X_train = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
Y_train = torch.tensor(Y, dtype=torch.float32)

batch_s    = 4096
train_data = TensorDataset(X_train, Y_train)
loader     = DataLoader(train_data, batch_size=batch_s, shuffle=False, pin_memory=True, num_workers=8)

def gradient(pred, data):
    du = grad(outputs=pred[:,0:1], inputs=data,
                      grad_outputs=torch.ones_like(pred[:,0:1]), 
                      create_graph=True)[0]
    
    dudy = du[:,1:2]
    
    dv = grad(outputs=pred[:,1:2], inputs=data,
                      grad_outputs=torch.ones_like(pred[:,1:2]), 
                      create_graph=True)[0]
    
    dvdx = dv[:,0:1]
    
    return dvdx-dudy
    
def laplacian(omega, data):
    dpdx = grad(outputs=omega, inputs=data,
                      grad_outputs=torch.ones_like(omega), 
                      create_graph=True)[0][:,0:1]
    
    dpdxx = grad(outputs=dpdx, inputs=data,
                      grad_outputs=torch.ones_like(omega), 
                      create_graph=True)[0][:,0:1]
    
    dpdy = grad(outputs=omega, inputs=data,
                      grad_outputs=torch.ones_like(omega), 
                      create_graph=True)[0][:,1:2]
    
    dpdyy = grad(outputs=dpdy, inputs=data,
                      grad_outputs=torch.ones_like(omega))[0][:,1:2]

    return dpdxx + dpdyy

def gen_range():
    return range(0,2501,100)

# compute spectral ground truth
def rv(inp):
    return inp.reshape(-1)[:,None]

domain_size = 2.0*np.pi;
N = 256;  h = (domain_size)/N;
x = [h*i for i in range(1,N+1)]; y = [h*i for i in range(1,N+1)]

dx = x[1] - x[0]; dy = y[1] - y[0];
print(" dx is ", dx, " dy is ", dy)

xx=np.zeros((N,N), dtype=float); yy=np.zeros((N,N), dtype=float)

for i in range(N):
    for j in range(N):
        xx[i,j] = x[i]
        yy[i,j] = y[j]

k_x = k_y = (2.0*np.pi/domain_size)*1j*np.fft.fftfreq (N , 1./ N). astype ( int );
ik_x = 1j*np.hstack((range(0,int(N/2)+1), range(-int(N/2)+1,0))); ik_y = ik_x; # for anti-alias

KX = KY = np.fft.fftfreq (N , 1./ N). astype ( int );
K = np.array ( np.meshgrid ( KX , KY, indexing ='ij') , dtype = int )
kmax_dealias = N/4.
dealias_cubic = np.array (( abs (K[0]) < kmax_dealias )*( abs (K[1]) < kmax_dealias ) ,dtype = bool )
kmax_dealias = N/3.
dealias_quad  = np.array (( abs (K[0]) < kmax_dealias )*( abs (K[1]) < kmax_dealias ) ,dtype = bool )

ikx   = np.zeros((N,N), dtype=complex);
iky   = np.zeros((N,N), dtype=complex);
kx    = np.zeros((N,N), dtype=complex);  
ky    = np.zeros((N,N), dtype=complex);
kxx   = np.zeros((N,N), dtype=complex); 
kyy   = np.zeros((N,N), dtype=complex);
kxxxx = np.zeros((N,N), dtype=complex);
kyyyy = np.zeros((N,N), dtype=complex);

for i in range(N):
    for j in range(N):
        kx[i,j]    = k_x[i];
        ky[i,j]    = k_y[j];
        ikx[i,j]   = ik_x[i];
        iky[i,j]   = ik_y[j];
        kxx[i,j]   = k_x[i]**2;
        kyy[i,j]   = k_y[j]**2;
        kxxxx[i,j] = k_x[i]**4;
        kyyyy[i,j] = k_y[j]**4;
        
lap  = kxx + kyy ; lap = np.where (lap == 0, 1, lap ) . astype ( complex )

dgt = sio.loadmat("/projects/ppm/surya/spectral_simulations/meso_bacterial_turbulence_IFRK4_N256_2pi_dt0p01_Tmax10.mat")
wf = np.transpose(data["store_data"], (2,0,1))[420:470, :, :]

lapu = np.zeros((50,64,64))
for i in range(len(t)):
    omega_hat = np.fft.fft2(wf[i])
    lapu[i] = np.fft.ifft2(lap*omega_hat).real[cut_lb:cut_ub, cut_lb:cut_ub]
    
rec = []
err = []
for ep in gen_range():
    layers = [3] + ll*[100] + [2]
    activation = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), Sin()][act]

    net = PINNNoWN(layers, mean=X_mean.to(device), std=X_std.to(device), seed=1, activation=activation).to(device)
    
    suff = "ep" + str(ep)
    model_name = "models/latent/model_active_s{}_l{}_n100_{}_nl{}_param_inf_dms{}_act1_meth{}_act{}_activ_reconstruction.pth".format(seed,ll,suff,nl,dms,method,act)
    checkpoint = torch.load(model_name, map_location=device)
    net.load_state_dict(checkpoint["model"])

    out = net(X_train)
    
    # reconstruction
    uh = out[:,0:1]

    # gradient
    gux = gradient(out, X_train)

    # laplacian
    lu = laplacian(gux, X_train)

    # prepare for writing
    uh  = uh.detach().view(50,64,64).cpu().data.numpy().copy()
    gux = gux.detach().view(50,64,64).cpu().data.numpy().copy()
    lu  = lu.detach().view(50,64,64).cpu().data.numpy().copy()

    # compute respective errors
    uerr = np.linalg.norm(uh.reshape(-1)-u[:,:,:].reshape(-1))/np.linalg.norm(u[:,:,:].reshape(-1))
    gerr = np.linalg.norm(gux.reshape(-1)-w[:,:,:].reshape(-1))/np.linalg.norm(w[:,:,:].reshape(-1))
    lerr = np.linalg.norm(lu.reshape(-1)-lapu.reshape(-1))/np.linalg.norm(lapu.reshape(-1))

    rec.append([uh, gux, lu])
    err.append([uerr, gerr, lerr])

    net = None
    
sio.savemat("recons/model_s{}_act{}.mat".format(seed, act), {"err": err, "rec": rec})
