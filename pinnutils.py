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

class BatchNorm(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self, x):
        return (x-self.mean)/self.std

class LayerNoWN(nn.Module):
    def __init__(self, in_features, out_features, seed, activation):
        super(LayerNoWN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
            
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)
        
        self.linear = self.linear
        
    def forward(self, x):
        return self.linear(x)

class PINNNoWN(nn.Module):
    
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(PINNNoWN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.bn = BatchNorm(mean, std)
        
        layer = []
        for i in range(len(sizes)-2):
            linear = LayerNoWN(sizes[i], sizes[i+1], seed, activation)
            layer += [linear, activation]
            
        layer += [LayerNoWN(sizes[-2], sizes[-1], seed, activation)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(self.bn(x))

def compute_vort_param(pred, data):
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

def compute_vort_pressure(pred, data):
    out_shape = pred[:, None, 0]

    uh = pred[:, None, 0]
    vh = pred[:, None, 1]
    ph = pred[:, None, 2]

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
    ph = ph.cpu().detach().data.numpy()
    wh = wh.cpu().detach().data.numpy()
    
    return uh, vh, ph, wh

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

def compute_pressure_grad(pred, data):
    out_shape = pred[:, None, 0]

    ph = pred[:, None, 2]

    # vorticity
    dp = grad(outputs=ph, inputs=data, grad_outputs=torch.ones_like(out_shape), retain_graph=True)[0].detach()

    dpdx = dp[:,0:1]
    dpdy = dp[:,1:2]

    dpdx = dpdx.cpu().detach().data.numpy()
    dpdy = dpdy.cpu().detach().data.numpy()
    
    return dpdx, dpdy

def pressure_eval(t_, dms, ts):
    dfull = sio.loadmat("/projects/ppm/surya/spectral_simulations/meso_bacterial_turbulence_IFRK4_N{}_2pi_dt0p01_Tmax10.mat".format(dms))
    
    u = np.transpose(dfull["store_data_u"], (2,0,1))[ts+t_]
    v = np.transpose(dfull["store_data_v"], (2,0,1))[ts+t_]
    
    N = 256; domain_size = 2.0*np.pi; h = (domain_size)/N;
    x = [h*i for i in range(1,N+1)]; y = [h*i for i in range(1,N+1)]
    dx = x[1] - x[0]; dy = y[1] - y[0];
    print(" dx is ", dx, " dy is ", dy)

    xx = np.zeros((N,N), dtype=float); yy=np.zeros((N,N), dtype=float)
    for i in range(N):
        for j in range(N):
            xx[i,j] = x[i]
            yy[i,j] = y[j]  

    S = -2.5; alpha = -1; beta = 0.5; lambda_0 = 1.0 - S; 

    lambda_1 = 0; eps = 0.1*10**(-13);

    k_x = k_y = (2.0*np.pi/domain_size)*1j*np.fft.fftfreq (N , 1./ N). astype ( int );
    kx    = np.zeros((N,N), dtype=complex);  
    ky    = np.zeros((N,N), dtype=complex);
    kxx   = np.zeros((N,N), dtype=complex); 
    kyy   = np.zeros((N,N), dtype=complex);
    kxxxx = np.zeros((N,N), dtype=complex);
    kyyyy = np.zeros((N,N), dtype=complex);

    for i in range(N):
        for j in range(N):
            kx[i,j]    = k_x[i]; ky[i,j]    = k_y[j];
            kxx[i,j]   = k_x[i]**2; kyy[i,j]   = k_y[j]**2;
            kxxxx[i,j] = k_x[i]**4; kyyyy[i,j] = k_y[j]**4;

    lap  = kxx + kyy + eps;
    inv_lap    = 1./lap;  

    uhat = np.fft.fft2(u); vhat = np.fft.fft2(v);
    ux = np.real(np.fft.ifft2(kx*uhat)); uy = np.real(np.fft.ifft2(ky*uhat))
    vx = np.real(np.fft.ifft2(kx*vhat)); vy = np.real(np.fft.ifft2(ky*vhat))
    adv_uhat = np.fft.fft2(u*ux + v*uy); adv_vhat = np.fft.fft2(u*vx + v*vy); 
    mod      = (u*u + v*v); mod_hat = np.fft.fft2(mod); mod_hat = mod_hat;
    mod_u    = mod*u; mod_v = mod*v; mod_uhat = np.fft.fft2(mod_u); mod_vhat = np.fft.fft2(mod_v);
    phat     = ( -lambda_0*(kx*adv_uhat + ky*adv_vhat) + lambda_1*lap*mod_hat \
                 -beta*(kx*mod_uhat + ky*mod_vhat) )*inv_lap;   

    pxhat = kx*phat; pyhat = ky*phat;
    
    return np.real(np.fft.ifft2(phat)), np.real(np.fft.ifft2(pxhat)), np.real(np.fft.ifft2(pyhat))

def residual_vort(pred, data, l0, alp, beta, gam0, gam2):
    out_shape = pred[:, None, 0]
    
    u = pred[:, None, 0]
    v = pred[:, None, 1]
    
    # modulo |\pmb{v}|^2
    M = u**2 + v**2
    
    dM = grad(outputs=M, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dMdx = dM[:,0:1]
    dMdy = dM[:,1:2]
    
    # vorticity
    du = grad(outputs=u, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dudx = du[:,0:1]
    dudy = du[:,1:2]
    
    dv = grad(outputs=v, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dvdx = dv[:,0:1]
    dvdy = dv[:,1:2]
    
    w = dvdx - dudy
    
    # vorticity gradients
    dw = grad(outputs=w, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dwdx = dw[:,0:1]
    dwdy = dw[:,1:2]
    dwdt = dw[:,2:3]
    
    # 2nd order
    dwdxx = grad(outputs=dwdx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dwdyy = grad(outputs=dwdy, inputs=data,
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # 4th order
    dwdxxx = grad(outputs=dwdxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dwdxxxx = grad(outputs=dwdxxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dwdyyy = grad(outputs=dwdyy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    dwdyyyy = grad(outputs=dwdyyy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # double 2nd order
    dwdxx_dy = grad(outputs=dwdxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    dwdxx_dyy = grad(outputs=dwdxx_dy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    w_term1 = ((u*dwdx) + (v*dwdy))
    w_term2 = w
    w_term3 = ((M*w) + ((dMdx*v) - (dMdy*u)))
    w_term4 = dwdxx + dwdyy
    w_term5 = dwdxxxx + dwdyyyy + (2*dwdxx_dyy)
    
    res1 = dwdt + l0*w_term1 + alp*w_term2 + beta*w_term3 - gam0*w_term4 + gam2*w_term5
    
    res2 = dudx + dvdy + 0*u
    
    return res1, res2

def loss_grad_std_full(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((grad_,w.view(-1), b))
            
    return torch.std(grad_)

def loss_grad_max_full(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = torch.abs(lambg*grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg*grad(loss, m.bias, retain_graph=True)[0])        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = torch.abs(lambg*grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg*grad(loss, m.bias, retain_graph=True)[0])        
            grad_ = torch.cat((grad_,w.view(-1), b))
    
    return torch.max(grad_), torch.mean(grad_)

def network_gradient(loss,net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]        
            grad_ = torch.cat((grad_,w.view(-1), b))
        
    return grad_

def reweight(d_, num_tasks, eps):
    nz_ind = np.where(d_>eps)
    z_ind  = np.delete(np.arange(num_tasks),nz_ind)
    if(len(z_ind)==0):
        return d_
    d_[nz_ind] = d_[nz_ind] - eps/len(d_[nz_ind])
    d_[z_ind]  = eps/len(z_ind)
    return d_

def solver_mine(Q, num_tasks, tol, maxiter=500):
    alphas = (1./num_tasks)*np.ones((num_tasks,))
    direct = np.zeros((num_tasks,2))

    for it in range(0, maxiter):
        ind_vec = np.zeros((num_tasks,));
        grad = Q @ alphas
        idx_oracle   = np.argmin(grad);
        ind_vec[idx_oracle] = 1.0;

        direct[:,0] = ind_vec; direct[:,1] = alphas;
        MM = (direct.T @ Q) @ direct

        if(MM[0,1] >= MM[0,0]):
            step_size = 1.0;
        elif(MM[0,1] >= MM[1,1]):
            step_size = 0;
        else:
            step_size = (MM[1,1] - MM[0,1])/(MM[0,0] + MM[1,1] - MM[0,1] - MM[1,0])

        alphas = (1. - step_size) * alphas
        alphas[idx_oracle] = alphas[idx_oracle] + step_size * ind_vec[idx_oracle]

    return reweight(alphas,num_tasks,tol)

# better/faster weight computation
def loss_grad_std_wn(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    return torch.std(grad_)

def loss_grad_max_wn(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    grad_ = torch.abs(lambg*grad_)
        
    return torch.max(grad_), torch.mean(grad_)

def network_gradient_wn(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    return grad_

def residual_pressure(pred, data, l0, alp, beta, gam0, gam2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_shape = pred[:, None, 0]
    
    u = pred[:, None, 0]
    v = pred[:, None, 1]
    p = pred[:, None, 2]
    
    # pressure gradient
    dp = grad(outputs=p, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dpdx = dp[:,0:1]
    dpdy = dp[:,1:2]
    
    # modulo |\pmb{v}|^2
    M = u**2 + v**2
    
    # u velocity
    du = grad(outputs=u, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dudx = du[:,0:1]
    dudy = du[:,1:2]
    dudt = du[:,2:3]
    
    # u velocity - 2nd order
    dudxx = grad(outputs=dudx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dudyy = grad(outputs=dudy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # u velocity - 4th order
    dudxxx = grad(outputs=dudxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dudxxxx = grad(outputs=dudxxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dudyyy = grad(outputs=dudyy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    dudyyyy = grad(outputs=dudyyy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # u velocity - double 2nd order
    dudxx_dy = grad(outputs=dudxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    dudxx_dyy = grad(outputs=dudxx_dy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # u velocity - residual
    u_term1 = l0*((u*dudx) + (v*dudy))
    u_term2 = alp*u
    u_term3 = beta*M*u
    u_term4 = gam0*(dudxx + dudyy)
    u_term5 = gam2*(dudxxxx + dudyyyy + (2*dudxx_dyy))
    
    res1 = dudt + u_term1 + dpdx + u_term2 + u_term3 - u_term4 + u_term5
    
    # v velocity
    dv = grad(outputs=v, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dvdx = dv[:,0:1]
    dvdy = dv[:,1:2]
    dvdt = dv[:,2:3]
    
    # v velocity - 2nd order
    dvdxx = grad(outputs=dvdx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dvdyy = grad(outputs=dvdy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # v velocity - 4th order
    dvdxxx = grad(outputs=dvdxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dvdxxxx = grad(outputs=dvdxxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,0:1]
    
    dvdyyy = grad(outputs=dvdyy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    dvdyyyy = grad(outputs=dvdyyy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # v velocity - double 2nd order
    dvdxx_dy = grad(outputs=dvdxx, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    dvdxx_dyy = grad(outputs=dvdxx_dy, inputs=data, 
                 grad_outputs=torch.ones_like(out_shape), 
                 create_graph=True)[0][:,1:2]
    
    # v velocity - residual
    v_term1 = l0*((u*dvdx) + (v*dvdy))
    v_term2 = alp*v
    v_term3 = beta*M*v
    v_term4 = gam0*(dvdxx + dvdyy)
    v_term5 = gam2*(dvdxxxx + dvdyyyy + (2*dvdxx_dyy))
    
    res2 = dvdt + v_term1 + dpdy + v_term2 + v_term3 - v_term4 + v_term5
    
    # divergence - residual
    res3 = dudx + dvdy + 0*u
    
    return res1, res2, res3
