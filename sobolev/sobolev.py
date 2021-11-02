import sys
sys.path.insert(0,"/lustre/projects/ppm/surya/")
import numpy as np
import numpy.random as npr
import cvxpy as cp

from platform import python_version
print(" cvxpy version ", cp.__version__) 
print(" python version ", python_version())

import scipy.io as sio
import matplotlib.pyplot as plt

import torch
from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

# raissi plots
from scipy.interpolate import griddata
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from network import *
from PDE_regularization import *
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from tqdm import tqdm_notebook as tqdm 
from scipy.integrate import simps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

method = np.int(sys.argv[1])

N = 128
domain_size = 2.0*np.pi; h = (domain_size)/N;
x = [h*i for i in range(1,N+1)]; y = [h*i for i in range(1,N+1)]
dx = x[1] - x[0]; dy = y[1] - y[0];
print(" dx is ", dx, " dy is ", dy)
xx = np.zeros((N,N), dtype=float); yy=np.zeros((N,N), dtype=float)

for i in range(N):
    for j in range(N):
        xx[i,j] = x[i]
        yy[i,j] = y[j]

L = 2.0*np.pi; Al = -5.0; Ar = 5.0;  wf = (2.0*np.pi/L);
Nrep = 20; 

np.random.seed(1)
Ax = np.zeros((Nrep,))
phix = np.zeros((Nrep,))
lx = np.zeros((Nrep,))
Ay = np.zeros((Nrep,))
phiy = np.zeros((Nrep,))
ly = np.zeros((Nrep,))

num_func = 10;

func = np.zeros((N,N,num_func));
derv = np.zeros((N,N,num_func));
lap  = np.zeros((N,N,num_func));
thd  = np.zeros((N,N,num_func));
bihm = np.zeros((N,N,num_func));
for i in range(0, num_func):
    np.random.seed(i+3)
    for j in range(0, Nrep):
        Ax[j] = (Ar-Al)*np.random.rand(1) + Al; Ay[j] = (Ar-Al)*np.random.rand(1) + Al
        phix[j] = (2.0*np.pi)*np.random.rand(1); phiy[j] = (2.0*np.pi)*np.random.rand(1)
        lx[j] = np.random.randint(1, 5); ly[j] = np.random.randint(1, 5)
#         print(" i ", i, " j ", Ax[j], Ay[j],phix[j], phiy[j],lx[j], ly[j])

    for k in range(0, Nrep):        
        x_c = wf*lx[k]*xx + phix[k]; y_c = wf*ly[k]*yy + phiy[k]
        func[:,:,i] = func[:,:,i] + Ax[k]*np.cos(x_c)* Ay[k]*np.sin(y_c)
        
#         print(" max func ", np.max(func), np.max(xx), np.min(yy))
        
        derv[:,:,i] = derv[:,:,i] - (wf*lx[k]*Ax[k])*np.sin(x_c)       * Ay[k]*np.sin(y_c) \
                                  + Ax[k]*np.cos(x_c)                  * (wf*ly[k]*Ay[k])*np.cos(y_c)
            
        lap[:,:,i]  = lap[:,:,i]  - ((wf*lx[k])**2)*Ax[k]*np.cos(x_c)  * Ay[k]*np.sin(y_c) \
                                  - Ax[k]*np.cos(x_c)                  * ((wf*ly[k])**2)*Ay[k]*np.sin(y_c)
            
        thd[:,:,i]  = thd[:,:,i]  + ((wf*lx[k])**3)*Ax[k]*np.sin(x_c)  * Ay[k]*np.sin(y_c) \
                                  - Ax[k]*np.cos(x_c)                  * ((wf*ly[k])**3)*Ay[k]*np.cos(y_c)
            
        bihm[:,:,i] = bihm[:,:,i] + ((wf*lx[k])**4)*Ax[k]*np.cos(x_c)  * Ay[k]*np.sin(y_c) \
                                  + Ax[k]*np.cos(x_c)                  * ((wf*ly[k])**4)*Ay[k]*np.sin(y_c)
            
print(" max function ",  np.max(func[:,:,0]), np.max(func[:,:,1]), np.max(func[:,:,2]))
    
coeff_1 = np.double(sys.argv[2]);
coeff_2 = np.double(sys.argv[3]);
coeff_3 = np.double(sys.argv[4]);
coeff_4 = np.double(sys.argv[5]);
eps_    = np.double(sys.argv[6]);
func_seed = np.int(sys.argv[7]);
weight_norm = np.int(sys.argv[8])

w = func[:,:,func_seed]; 
grad_w = derv[:,:,func_seed];
Lap_w =  lap[:,:,func_seed];
Thd_w =  thd[:,:,func_seed];
Lap2_w = bihm[:,:,func_seed];

N_train = 8192
x_    = xx[:,:].flatten()[:,None] # NT x 1
y_    = yy[:,:].flatten()[:,None] # NT x 1
w_    = w[:,:].flatten()[:,None]
grad_w_ = grad_w.real[:,:].flatten()[:,None]
Lap_w_  = Lap_w.real[:,:].flatten()[:,None]
Thd_w_  = Thd_w.real[:,:].flatten()[:,None]
Lap2_w_  = Lap2_w.real[:,:].flatten()[:,None]
print(" x_ shape ", x_.shape)
print(" y_ shape ", y_.shape)
print(" w_ shape ", w_.shape)
print(" grad_w_ shape ", grad_w_.shape)
print(" Lap_w_ shape ", Lap_w_.shape)

print(" max gradient ", np.max(np.abs(Lap_w_)), np.max(np.abs(Thd_w_)))

## optimal lamba computation based on epsilon-closeness
num_tasks = 5;
integrate_mat = np.zeros((N,N,num_tasks))
integrate_mat[:,:,0] = w[:,:]; 
integrate_mat[:,:,1] = np.real(grad_w[:,:]); 
integrate_mat[:,:,2] = np.real(Lap_w[:,:]);
integrate_mat[:,:,3] = np.real(Thd_w[:,:]);
integrate_mat[:,:,4] = np.real(Lap2_w[:,:]);

cut_lb = 0; cut_ub = 128;
M_int = np.zeros((num_tasks,))
for task in range(0, num_tasks):
    M_int[task] = simps(simps(integrate_mat[cut_lb:cut_ub,cut_lb:cut_ub,task]**2, y[cut_lb:cut_ub]), x[cut_lb:cut_ub])
    
comb = np.ones((num_tasks,)); total_comb = 0;
for task in range(0, num_tasks):
    value = 1.0;
    for j in range(0, num_tasks):
        if( j!= task ):
            value = value*M_int[j]
    comb[task] = value; total_comb = total_comb + comb[task]
    
print(" total comb ", total_comb)
optimal_lamb = np.zeros((num_tasks,))
for task in range(0, num_tasks):
    optimal_lamb[task] = comb[task]/total_comb
    
print(" optimal lamb ", optimal_lamb)
print(" sum ", np.sum(optimal_lamb))

seed = 1
np.random.seed(seed)
idx = np.random.choice(len(x_), N_train, replace=False)
x_train = x_[idx,:]
y_train = y_[idx,:]
w_train = w_[idx,:]
grad_w_train = grad_w_[idx,:]
Lap_w_train  = Lap_w_[idx,:]
Thd_w_train  = Thd_w_[idx,:]
Lap2_w_train  = Lap2_w_[idx,:]

X = np.concatenate([x_train, y_train], 1)  # 
Y = np.concatenate([w_train, grad_w_train, Lap_w_train, Thd_w_train, Lap2_w_train], 1) # shapes of m,1
X_train = torch.tensor(X, dtype=torch.float32,device=device)
Y_train = torch.tensor(Y, dtype=torch.float32,device=device)

# input normalization
TD = np.concatenate([X], 0)
# compute mean and std of training data
X_mean = torch.tensor(np.mean(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(TD, axis=0, keepdims=True), dtype=torch.float32, device=device)

print(" X-mean ", X_mean, "x-std", X_std, " shapes ", X_mean.size(), X_std.size() )

print(X_train.shape)
print(Y_train.shape)

batch_s = 4096
#train_data = TensorDataset(X_train, Y_train)
#train_loader     = DataLoader(train_data, batch_size=batch_s, shuffle=False, pin_memory=False, num_workers=0)
train_loader = FastTensorDataLoader(X_train, Y_train, batch_size=batch_s, shuffle=True, pin_memory=False, num_workers=0)
print(" number of batches ", len(train_loader))
print(" print size ", X_train.size() ,Y_train.size())

class Sin(nn.Module):
    """Sin activation function."""
    
    def forward(self, x):
        return torch.sin(x)
    
def poisson_res(pred, data,inf_coeff_1, inf_coeff_2, inf_coeff_3, inf_coeff_4):
    out_shape = pred[:, None, 0]
    
    w = pred[:,None,0];
    dw = grad(outputs=w, inputs=data,
              grad_outputs=torch.ones_like(out_shape), 
              create_graph=True)[0]
    
    dwdx = dw[:,0:1]; dwdy = dw[:,1:2];
    
    grad_ = dwdx + dwdy
                                     
    dwdxx = grad(outputs=dwdx, inputs=data, 
              grad_outputs=torch.ones_like(out_shape), create_graph=True)[0][:,0:1]
    dwdyy = grad(outputs=dwdy, inputs=data, 
              grad_outputs=torch.ones_like(out_shape), create_graph=True)[0][:,1:2]
    
    Lap_ = dwdxx + dwdyy
    
    dwdxxx = grad(outputs=dwdxx, inputs=data, 
                     grad_outputs=torch.ones_like(out_shape), 
                     create_graph=True)[0][:,0:1]

    dwdyyy = grad(outputs=dwdyy, inputs=data, 
                     grad_outputs=torch.ones_like(out_shape), 
                     create_graph=True)[0][:,1:2]
                                     
    Thd_   = dwdxxx + dwdyyy
    
    dwdxxxx = grad(outputs=dwdxxx, inputs=data, 
                     grad_outputs=torch.ones_like(out_shape), 
                     create_graph=True)[0][:,0:1]

    dwdyyyy = grad(outputs=dwdyyy, inputs=data, 
                     grad_outputs=torch.ones_like(out_shape), 
                     create_graph=True)[0][:,1:2]

    biharm = (dwdxxxx + dwdyyyy)
    
    return inf_coeff_1*(grad_)+0*w, inf_coeff_2*(Lap_)+0*w, inf_coeff_3*(Thd_)+0*w, inf_coeff_4*(biharm)+0*w

def network_gradient(loss,net):
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

def quad(Q, num_tasks):
    
    nz = num_tasks; nineq = num_tasks;
    z = cp.Variable(nz)
    G = -np.eye(nineq, nz)
    p = np.zeros((nz,1))
    h = np.zeros((nineq,))
    A = np.ones((1,nz))
    b = np.ones((1,1))
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(z, Q) + p.T @ z),
                 [A@z==b,  G@z <= h])
    prob.solve()
    c_ = z.value
    return c_                                  

def renormalize(d_, num_tasks, eps):
    #nz_ind = np.nonzero(d_)
    nz_ind = np.where(d_ > eps)
    z_ind  = np.delete(np.arange(num_tasks),nz_ind)
    if(len(z_ind)==0):
        return d_
    d_[nz_ind] = d_[nz_ind] - eps/len(d_[nz_ind])
    d_[z_ind]  = eps/len(z_ind)
    return d_

def solver_mine(Q, num_tasks, eps_, maxiter=1000):
    
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
        
        if(step_size <= 1e-8):
            return renormalize(alphas, num_tasks, eps_)
        
        alphas = (1. - step_size) * alphas
        alphas[idx_oracle] = alphas[idx_oracle] + step_size * ind_vec[idx_oracle]

    return renormalize(alphas, num_tasks, eps_)

def loss_grad_max(loss, net, lambg=1):
    mean = []
    maxg = []
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
            
        w = torch.abs(lambg*grad(loss, m.weight, create_graph=True)[0])
        b = torch.abs(lambg*grad(loss, m.bias, create_graph=True)[0])
        
        wb = torch.cat((w.view(-1), b))
        
        mean.append(torch.mean(wb))
        maxg.append(torch.max(wb))
        
    meant = torch.tensor(mean, dtype=torch.float32, device=device)
    maxgt = torch.tensor(maxg, dtype=torch.float32, device=device)
    
    return torch.max(maxgt), torch.mean(meant)

def loss_grad_std(loss, net):
    var = []
    siz = []
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        
        w = grad(loss, m.weight, retain_graph=True)[0]
        b = grad(loss, m.bias, retain_graph=True)[0]
        
        wb = torch.cat((w.view(-1), b))
        
        nit  = torch.numel(wb)
        var.append((nit - 1) * torch.var(wb))
        siz.append(nit)

    vart = torch.tensor(var, dtype=torch.float32,device=device)
    sizt = torch.tensor(siz, dtype=torch.float32,device=device)
    
    return torch.sqrt(torch.sum(vart)/(torch.sum(sizt) - len(sizt)))

def loss_grad_std_alt(loss, net):
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        
        w = grad(loss, m.weight, retain_graph=True)[0]
        b = grad(loss, m.bias, retain_graph=True)[0]
        
        if(m==0):
            grad_ = torch.cat((w.view(-1), b))
        else:
            grad_ = torch.cat((grad_,w.view(-1), b))
    
    return torch.var(grad_)

def loss_grad_max_alt(loss, net, lambg=1):
    grad_ = torch.zeros((0), dtype=torch.float32,device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
            
        w = torch.abs(lambg*grad(loss, m.weight, create_graph=True)[0])
        b = torch.abs(lambg*grad(loss, m.bias, create_graph=True)[0])
        
        if(m==0):
            grad_ = torch.cat((w.view(-1), b))
        else:
            grad_ = torch.cat((grad_,w.view(-1), b))
     
    print(" grad_ size ", grad_.size())
        
    meant = torch.mean(grad_)
    maxgt = torch.max(grad_)
    
    return maxgt, meant

def network_gradient_viz(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, lambg*elem.view(-1)))
        
    return grad_.detach().cpu().data.numpy()

import time
ll = 5
layers = [2] + ll*[64] + [1]

net = PINN(sizes=layers, seed=seed, activation=Sin()).to(device)
#net = PINN(sizes=layers,  seed=seed, activation=Sin()).to(device)
inf_coeff_1 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
inf_coeff_2 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
inf_coeff_3 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
inf_coeff_4 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
n_epochs = 20_001
lamb = 1;

num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(" num parameters ", num_parameters)

it_grad = 1000;
grad_viz = np.zeros((num_parameters,num_tasks,int(n_epochs/it_grad)+1))
lambda_store = np.zeros((n_epochs,num_tasks))

alpha_ann = 0.5
mm = 5; lambs = [];
coeff1_track = [];
coeff2_track = [];
coeff3_track = [];
coeff4_track = [];

params = [{'params': net.parameters(), 'lr': 1e-3},
          {'params': inf_coeff_1, 'lr': 1e-3},
          {'params': inf_coeff_2, 'lr': 1e-3},
          {'params': inf_coeff_3, 'lr': 1e-3},
          {'params': inf_coeff_4, 'lr': 1e-3},
         ]
milestones = [[10000,15000]]
optimizer = Adam(params)
scheduler = MultiStepLR(optimizer, milestones[0], gamma=0.1)

alpha_reg = 0.5;
loss_u = [];
loss_r1 = []; loss_r2 = []; loss_r3 = []; loss_r4 = [];
lamb_u = 1.0; lamb_r1 = 1.0; lamb_r2 = 1.0; lamb_r3 = 1.0
import time
start_time = time.time()
pbar = tqdm(range(n_epochs))
for epoch in pbar:
    for i, (x_batch,y_batch) in enumerate(train_loader):

        Y_batch = y_batch.to(device)
        start_comp = time.time()
        optimizer.zero_grad()
        pred = net(x_batch.to(device).requires_grad_(True))
        l_u = torch.sum(torch.mean((Y_batch[:,0:1] - pred[:,0:1])**2, dim=0)) # mean along rows 
        
        res1, res2, res3, res4 = poisson_res(pred, x_batch.to(device).requires_grad_(True), inf_coeff_1, inf_coeff_2, inf_coeff_3, inf_coeff_4)
        l_res1 = torch.sum(torch.mean((Y_batch[:,1:2].to(device) - res1[:,0:1])**2, dim=0))
        l_res2 = torch.sum(torch.mean((Y_batch[:,2:3].to(device) - res2[:,0:1])**2, dim=0))
        l_res3 = torch.sum(torch.mean((Y_batch[:,3:4].to(device) - res3[:,0:1])**2, dim=0))
        l_res4 = torch.sum(torch.mean((Y_batch[:,4:5].to(device) - res4[:,0:1])**2, dim=0))

        if(method==1):
            with torch.no_grad():
                if epoch % mm == 0 and i == 0:
                    G_r1 = network_gradient(l_res1,net)
                    G_r2 = network_gradient(l_res2,net)
                    G_r3 = network_gradient(l_res3,net)
                    G_r4 = network_gradient(l_res4,net)
                    G_fit = network_gradient(l_u,net)

                    Mat = torch.zeros((G_fit.shape[0],num_tasks),device=device)
                    Mat[:,0] = G_r1; Mat[:,1]  = G_r2; Mat[:,2] = G_r3; Mat[:,3] = G_r4; Mat[:,4] = G_fit
                    #c_ = quad(torch.matmul(Mat.T, Mat).cpu().numpy())
                    c_  =  solver_mine(torch.matmul(Mat.T, Mat).cpu().numpy(), num_tasks, eps_, maxiter=1000)
                    
                    if(epoch % it_grad == 0):
                        grad_viz[:,0,int(epoch/it_grad)] = c_[4]*network_gradient_viz(l_u,net)
                        grad_viz[:,1,int(epoch/it_grad)] = c_[0]*network_gradient_viz(l_res1,net)
                        grad_viz[:,2,int(epoch/it_grad)] = c_[1]*network_gradient_viz(l_res2,net)
                        grad_viz[:,3,int(epoch/it_grad)] = c_[2]*network_gradient_viz(l_res3,net)
                        grad_viz[:,4,int(epoch/it_grad)] = c_[3]*network_gradient_viz(l_res4,net)
                    
        if(method==3):
            with torch.no_grad():
                if epoch % mm == 0 and i == 0:
                    
                    stdr1 = loss_grad_std(l_res1, net);
                    stdr2 = loss_grad_std(l_res2, net);
                    stdr3 = loss_grad_std(l_res3, net);
                    stdr4 = loss_grad_std(l_res4, net);
                    stdu  = loss_grad_std(l_u, net);
                
                    if(stdu==0):
                        stdu = 1.0;
                    if(stdr1 == 0):
                        stdr1 = 1.0;
                    if(stdr2 == 0):
                        stdr2 = 1.0;
                    if(stdr3 == 0):
                        stdr3 = 1.0;
    
                    lamb_hat = stdr4/stdu;  lamb_u      = (1-alpha_ann)*lamb_u + alpha_ann*lamb_hat
                    lamb_hat = stdr4/stdr3; lamb_r3     = (1-alpha_ann)*lamb_r3 + alpha_ann*lamb_hat
                    lamb_hat = stdr4/stdr2; lamb_r2     = (1-alpha_ann)*lamb_r2 + alpha_ann*lamb_hat
                    lamb_hat = stdr4/stdr1; lamb_r1     = (1-alpha_ann)*lamb_r1 + alpha_ann*lamb_hat  
                    
                    if(epoch % it_grad == 0):
                        #print(lamb_u.is_cuda, lamb_u.to('cpu').is_cuda, lamb_u.is_cuda)
                        grad_viz[:,0,int(epoch/it_grad)] = lamb_u.to('cpu')*network_gradient_viz(l_u,net)
                        grad_viz[:,1,int(epoch/it_grad)] = lamb_r1.to('cpu')*network_gradient_viz(l_res1,net)
                        grad_viz[:,2,int(epoch/it_grad)] = lamb_r2.to('cpu')*network_gradient_viz(l_res2,net)
                        grad_viz[:,3,int(epoch/it_grad)] = lamb_r3.to('cpu')*network_gradient_viz(l_res3,net)
                        grad_viz[:,4,int(epoch/it_grad)] = network_gradient_viz(l_res4,net)
                    
        if(method==5):
            with torch.no_grad():
                if epoch % mm == 0 and i == 0:
                    
                    stdr1 = loss_grad_std_alt(l_res1, net);
                    stdr2 = loss_grad_std_alt(l_res2, net);
                    stdr3 = loss_grad_std_alt(l_res3, net);
                    stdr4 = loss_grad_std_alt(l_res4, net);
                    stdu  = loss_grad_std_alt(l_u, net);
                
                    if(stdu==0):
                        stdu = 1.0;
                    if(stdr1 == 0):
                        stdr1 = 1.0;
                    if(stdr2 == 0):
                        stdr2 = 1.0;
                    if(stdr3 == 0):
                        stdr3 = 1.0;
    
                    lamb_hat = stdr4/stdu;  lamb_u      = (1-alpha_ann)*lamb_u + alpha_ann*lamb_hat
                    lamb_hat = stdr4/stdr3; lamb_r3     = (1-alpha_ann)*lamb_r3 + alpha_ann*lamb_hat
                    lamb_hat = stdr4/stdr2; lamb_r2     = (1-alpha_ann)*lamb_r2 + alpha_ann*lamb_hat
                    lamb_hat = stdr4/stdr1; lamb_r1     = (1-alpha_ann)*lamb_r1 + alpha_ann*lamb_hat  
                    
        if(method==4):
            with torch.no_grad():
                if epoch % mm == 0 and i == 0:
                    
                    _, meanu  = loss_grad_max(l_u, net, lambg=lamb_u)    
                    _, meanr1 = loss_grad_max(l_res1, net, lambg=lamb_r1)
                    _, meanr2 = loss_grad_max(l_res2, net, lambg=lamb_r2)
                    _, meanr3 = loss_grad_max(l_res3, net, lambg=lamb_r3)
                    maxr4, _  = loss_grad_max(l_res4, net)
                    
                    if(meanu==0):
                        meanu = 1.0;
                    if(meanr1 == 0):
                        meanr1 = 1.0;
                    if(meanr2 == 0):
                        meanr2 = 1.0;
                    if(meanr3 == 0):
                        meanr3 = 1.0;
                    
                    lamb_hat  = maxr4/meanu;   lamb_u      = (1-alpha_ann)*lamb_u + alpha_ann*lamb_hat
                    lamb_hat  = maxr4/meanr3;  lamb_r3     = (1-alpha_ann)*lamb_r3 + alpha_ann*lamb_hat
                    lamb_hat  = maxr4/meanr2;  lamb_r2     = (1-alpha_ann)*lamb_r2 + alpha_ann*lamb_hat
                    lamb_hat  = maxr4/meanr1;  lamb_r1     = (1-alpha_ann)*lamb_r1 + alpha_ann*lamb_hat 
                    
        if(method==6):
            with torch.no_grad():
                if epoch % mm == 0 and i == 0:
                    
                    _, meanu  = loss_grad_max_alt(l_u, net, lambg=lamb_u)    
                    _, meanr1 = loss_grad_max_alt(l_res1, net, lambg=lamb_r1)
                    _, meanr2 = loss_grad_max_alt(l_res2, net, lambg=lamb_r2)
                    _, meanr3 = loss_grad_max_alt(l_res3, net, lambg=lamb_r3)
                    maxr4, _  = loss_grad_max_alt(l_res4, net)
                    
                    if(meanu==0):
                        meanu = 1.0;
                    if(meanr1 == 0):
                        meanr1 = 1.0;
                    if(meanr2 == 0):
                        meanr2 = 1.0;
                    if(meanr3 == 0):
                        meanr3 = 1.0;
                    
                    lamb_hat  = maxr4/meanu;   lamb_u      = (1-alpha_ann)*lamb_u + alpha_ann*lamb_hat
                    lamb_hat  = maxr4/meanr3;  lamb_r3     = (1-alpha_ann)*lamb_r3 + alpha_ann*lamb_hat
                    lamb_hat  = maxr4/meanr2;  lamb_r2     = (1-alpha_ann)*lamb_r2 + alpha_ann*lamb_hat
                    lamb_hat  = maxr4/meanr1;  lamb_r1     = (1-alpha_ann)*lamb_r1 + alpha_ann*lamb_hat
                    
                    if(epoch % it_grad == 0):
                        grad_viz[:,0,int(epoch/it_grad)] = lamb_u.to('cpu')*network_gradient_viz(l_u,net)
                        grad_viz[:,1,int(epoch/it_grad)] = lamb_r1.to('cpu')*network_gradient_viz(l_res1,net)
                        grad_viz[:,2,int(epoch/it_grad)] = lamb_r2.to('cpu')*network_gradient_viz(l_res2,net)
                        grad_viz[:,3,int(epoch/it_grad)] = lamb_r3.to('cpu')*network_gradient_viz(l_res3,net)
                        grad_viz[:,4,int(epoch/it_grad)] = network_gradient_viz(l_res4,net)
                    
        if(method==1):
            loss =  c_[0]*l_res1 + c_[1]*l_res2 + c_[2]*l_res3 + c_[3]*l_res4 + c_[4]*l_u
            lambs.append([c_[0].item(), c_[1].item(), c_[2].item(), c_[3].item(), c_[4].item()])
            
        elif(method==2):
            if(epoch%it_grad==0):
                grad_viz[:,0,int(epoch/it_grad)] = optimal_lamb[0]*network_gradient_viz(l_u,net)
                grad_viz[:,1,int(epoch/it_grad)] = optimal_lamb[1]*network_gradient_viz(l_res1,net)
                grad_viz[:,2,int(epoch/it_grad)] = optimal_lamb[2]*network_gradient_viz(l_res2,net)
                grad_viz[:,3,int(epoch/it_grad)] = optimal_lamb[3]*network_gradient_viz(l_res3,net)
                grad_viz[:,4,int(epoch/it_grad)] = optimal_lamb[4]*network_gradient_viz(l_res4,net) 
                
            lambs.append([optimal_lamb[0].item(), optimal_lamb[1].item(), optimal_lamb[2].item(), optimal_lamb[3].item(), optimal_lamb[4].item()])
            
            loss =  optimal_lamb[0]*l_u + optimal_lamb[1]*l_res1 + optimal_lamb[2]*l_res2 + optimal_lamb[3]*l_res3 + optimal_lamb[4]*l_res4
            
        elif(method==3 or method==4 or method==5 or method==6):
            loss = lamb_r1*l_res1 + lamb_r2*l_res2 + lamb_r3*l_res3 + l_res4 + lamb_u*l_u
            lambs.append([lamb_u.item(), lamb_r1.item(), lamb_r2.item(), lamb_r3.item()])
        else:
            if(epoch%it_grad==0):
                grad_viz[:,0,int(epoch/it_grad)] = network_gradient_viz(l_u,net)
                grad_viz[:,1,int(epoch/it_grad)] = network_gradient_viz(l_res1,net)
                grad_viz[:,2,int(epoch/it_grad)] = network_gradient_viz(l_res2,net)
                grad_viz[:,3,int(epoch/it_grad)] = network_gradient_viz(l_res3,net)
                grad_viz[:,4,int(epoch/it_grad)] = network_gradient_viz(l_res4,net) 
                
            loss = l_res1 + l_res2 + l_res3 + l_res4 + l_u

        loss.backward()
        optimizer.step()
        loss_u.append(l_u)
        loss_r1.append(l_res1)
        loss_r2.append(l_res2)
        loss_r3.append(l_res3)
        loss_r4.append(l_res4)
        coeff1_track.append(inf_coeff_1.item())
        coeff2_track.append(inf_coeff_2.item())
        coeff3_track.append(inf_coeff_3.item())
        coeff4_track.append(inf_coeff_4.item())
    
        if(epoch%10==0):
            torch.save(net, '/projects/ppm/surya/gradpath/model_m'+str(method)+'_seed'+str(func_seed)+'ep_'+str(epoch)+'.pth')
    
        if(i==0):
            print("epoch {}/{}, loss={:.10f}, coeff_1={:.4f}, coeff_2={:.4f}, coeff_3={:.4f}, coeff_4={:.4f}, lr={:,.5f}\t\t\t".format(epoch+1, n_epochs, loss.item(), inf_coeff_1, inf_coeff_2, inf_coeff_3, inf_coeff_4, optimizer.param_groups[0]['lr']), end="\r")
    
    scheduler.step()
    
#     if( epoch % 10000 == 0 ):
#         torch.save(net, './models/sobolev/vizgrad_sobolev_arbt_5terms_meth' + str(method) + '_c1_' + str(coeff_1 ) + '_c2_' + str(coeff_2) + '_c3_' + str(coeff_3) + '_fs' + str(func_seed) + '_eps'+ str(eps_) + '_ep' + str(epoch) + '_' + "sin" + '_weightnorm' + str(weight_norm) + '.pth')
    
mdict = {
    "t": (time.time() - start_time),
    "l_bc":  loss_u,
    "l_res1": loss_r1,
    "l_res2": loss_r2,
    "l_res3": loss_r3,
    "l_res4": loss_r4,
    "lambs" : lambs,
    "coeff1": coeff1_track,
    "coeff2": coeff2_track,   
    "coeff3": coeff3_track, 
    "coeff4": coeff4_track, 
    "gradviz": grad_viz,
    }
sio.savemat("./models/sobolev/vizgrad_sobolev_arbt_5terms_param_track_method{}_coeff1_{}_coeff2_{}_coeff3_{}_coeff4_{}_seed{}_eps{}_wn{}".format(method, coeff_1, coeff_2, coeff_3, coeff_4, func_seed, eps_, weight_norm), mdict)
    
elapsed_time = time.time() - start_time
print('CPU time = ',elapsed_time)