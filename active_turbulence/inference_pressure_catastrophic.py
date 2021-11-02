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
method = np.int(sys.argv[5])       # method used
pt     = 1000
tol    = 1e-6
pseed  = 1
ms     = 0

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

# get train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=seed)

# training data
X_train = torch.tensor(Xtrain, dtype=torch.float32)
Y_train = torch.tensor(Ytrain, dtype=torch.float32)

# test data
X_test = torch.tensor(Xtest, dtype=torch.float32)
Y_test = torch.tensor(Ytest, dtype=torch.float32)

# setup data loaders
batch_s      = 4096
loader       = FastTensorDataLoader(X_train, Y_train, batch_size=batch_s, shuffle=True)
test_loader  = FastTensorDataLoader(X_test, Y_test, batch_size=batch_s, shuffle=True)

# compute mean and std of training data
X_mean = torch.tensor(np.mean(Xtrain, axis=0, keepdims=True), dtype=torch.float32, device=device)
X_std  = torch.tensor(np.std(Xtrain, axis=0, keepdims=True), dtype=torch.float32, device=device)

# setup stuff for evaluation
u_train, u_test, v_train, v_test     = train_test_split(u[:interpol].reshape(-1)[:,None], v[:interpol].reshape(-1)[:,None], test_size=0.3, random_state=seed)
w_train, w_test, p_train, p_test     = train_test_split(w[:interpol].reshape(-1)[:,None], p[:interpol].reshape(-1)[:,None], test_size=0.3, random_state=seed)
px_train, px_test, py_train, py_test = train_test_split(px[:interpol].reshape(-1)[:,None], py[:interpol].reshape(-1)[:,None], test_size=0.3, random_state=seed)

def save_model(suff):
    # save model for given suff, i.e. ep1000, best_test, full, ...
    model_name = ""
    if method == 3:
        model_name = "latent/model_active_s{}_l{}_n100_{}_nl{}_param_inf_dms{}_act1_meth{}_pt{}_eps{}_pseed{}_dot_grad_ms{}_train_divseq_full_reset_max".format(seed,ll,suff,nl,dms,method,pt,tol,pseed,ms)
    else:
        model_name = "latent/model_active_s{}_l{}_n100_{}_nl{}_param_inf_dms{}_act1_meth{}_pt{}_pseed{}_dot_grad_ms{}_train_divseq_full_reset_max".format(seed,ll,suff,nl,dms,method,pt,pseed,ms)
    
    torch.save({
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
        "params": [l0.item(), alp.item(), beta.item(), gam0.item(), gam2.item()],
    }, "models/" + model_name + ".pth")
    
    model_name = ""
    if method == 3:
        model_name = "latent/model_active_s{}_l{}_n100_{}_nl{}_param_inf_dms{}_act1_meth{}_pt{}_eps{}_pseed{}_dot_grad_ms{}_train_divseq_full_reset_max".format(seed,ll,"full",nl,dms,method,pt,tol,pseed,ms)
    else:
        model_name = "latent/model_active_s{}_l{}_n100_{}_nl{}_param_inf_dms{}_act1_meth{}_pt{}_pseed{}_dot_grad_ms{}_train_divseq_full_reset_max".format(seed,ll,"full",nl,dms,method,pt,pseed,ms)

    mdict = {
        "epoch": epoch,
        "lambs": lambs,
        "lu": losses_u,
        "lr": losses_r,
        "lrd": losses_r1,
        "t": (time.time()-start)
    }

    sio.savemat("results/" + model_name, mdict)
    
def get_eval_name():
    save_name = ""
    if method == 3:
        save_name = "evals/latent/eval_s{}_nl{}_dms{}_meth{}_pt{}_eps{}_pseed{}_dot_grad_ms{}_train_divseq_full_reset_max".format(seed,nl,dms,method,pt,tol,pseed,ms)
    else:
        save_name = "evals/latent/eval_s{}_nl{}_dms{}_meth{}_pt{}_pseed{}_dot_grad_ms{}_train_divseq_full_reset_max".format(seed,nl,dms,method,pt,pseed,ms)
        
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
    X_full             = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    out_full           = net(X_full)
    uhf, vhf, phf, whf = compute_vort_pressure(out_full, X_full)
    pxhf, pyhf         = compute_pressure_grad(out_full, X_full)
    
    mdict["u_full"] = uhf.reshape(u.shape)
    mdict["v_full"] = vhf.reshape(u.shape)
    mdict["w_full"] = whf.reshape(u.shape)
    mdict["p_full"] = phf.reshape(u.shape)
    mdict["px_full"] = pxhf.reshape(u.shape)
    mdict["py_full"] = pyhf.reshape(u.shape)
    
    # get prediction error of training data
    X_trn              = torch.tensor(Xtrain, dtype=torch.float32, device=device, requires_grad=True)
    out_train          = net(X_trn)
    uht, vht, pht, wht = compute_vort_pressure(out_train, X_trn)
    pxht, pyht         = compute_pressure_grad(out_train, X_trn)
    
    # compute rel. l2-errors
    ut_err  = np.linalg.norm(uht - u_train)/np.linalg.norm(u_train)
    vt_err  = np.linalg.norm(vht - v_train)/np.linalg.norm(v_train)
    wt_err  = np.linalg.norm(wht - w_train)/np.linalg.norm(w_train)
    pxt_err = np.linalg.norm(pxht - px_train)/np.linalg.norm(px_train)
    pyt_err = np.linalg.norm(pyht - py_train)/np.linalg.norm(py_train)
    
    print("u err", ut_err)
    
    # read previous evaluations
    uerrs   = mdict["train_u"][0].tolist() if not isfirst else []
    verrs   = mdict["train_v"][0].tolist() if not isfirst else []
    werrs   = mdict["train_w"][0].tolist() if not isfirst else []
    pxerrs  = mdict["train_px"][0].tolist() if not isfirst else []
    pyerrs  = mdict["train_py"][0].tolist() if not isfirst else []
    
    # append current update
    uerrs.append(ut_err)
    verrs.append(vt_err)
    werrs.append(wt_err)
    pxerrs.append(pxt_err)
    pyerrs.append(pyt_err)
    
    # write errors to file
    mdict["train_u"] = uerrs
    mdict["train_v"] = verrs
    mdict["train_w"] = werrs
    mdict["train_px"] = pxerrs
    mdict["train_py"] = pyerrs
    
    # get prediction error of test data
    X_tst                  = torch.tensor(Xtest, dtype=torch.float32, device=device, requires_grad=True)
    out_test               = net(X_tst)
    uhtt, vhtt, phtt, whtt = compute_vort_pressure(out_test, X_tst)
    pxhtt, pyhtt           = compute_pressure_grad(out_test, X_tst)
    
    # compute rel. l2-errors
    utt_err  = np.linalg.norm(uhtt - u_test)/np.linalg.norm(u_test)
    vtt_err  = np.linalg.norm(vhtt - v_test)/np.linalg.norm(v_test)
    wtt_err  = np.linalg.norm(whtt - w_test)/np.linalg.norm(w_test)
    pxtt_err = np.linalg.norm(pxhtt - px_test)/np.linalg.norm(px_test)
    pytt_err = np.linalg.norm(pyhtt - py_test)/np.linalg.norm(py_test)
    
    # read previous evaluations
    uterrs   = mdict["test_u"][0].tolist() if not isfirst else []
    vterrs   = mdict["test_v"][0].tolist() if not isfirst else []
    wterrs   = mdict["test_w"][0].tolist() if not isfirst else []
    pxterrs  = mdict["test_px"][0].tolist() if not isfirst else []
    pyterrs  = mdict["test_py"][0].tolist() if not isfirst else []
    
    # append current update
    uterrs.append(utt_err)
    vterrs.append(vtt_err)
    wterrs.append(wtt_err)
    pxterrs.append(pxtt_err)
    pyterrs.append(pytt_err)
    
    # write errors to file
    mdict["test_u"] = uterrs
    mdict["test_v"] = vterrs
    mdict["test_w"] = wterrs
    mdict["test_px"] = pxterrs
    mdict["test_py"] = pyterrs
    
    # read and append inferred parameter
    pars = mdict["ps"].tolist() if not isfirst else []
    pars.append([l0.item(), alp.item(), beta.item(), gam0.item(), gam2.item()])
    
    mdict["ps"] = pars
    
    # save updated dictionary
    sio.savemat(get_eval_name(), mdict)
    
    # save model checkpoint and remaining stuff, e.g. losses, lambdas, ...
    save_model(suff)
    
def residual_div(pred, data):
    out_shape = pred[:, None, 0]

    uh = pred[:, None, 0]
    vh = pred[:, None, 1]

    # vorticity
    du = grad(outputs=uh, inputs=data, grad_outputs=torch.ones_like(out_shape), create_graph=True)[0]

    dudx = du[:,0:1]
    dudy = du[:,1:2]

    dv = grad(outputs=vh, inputs=data, grad_outputs=torch.ones_like(out_shape), create_graph=True)[0]

    dvdx = dv[:,0:1]
    dvdy = dv[:,1:2]

    div = dudx + dvdy + 0*uh
    
    return div

# setup of neural net and training loop
layers = [3] + ll*[100] + [3]
net    = PINNNoWN(layers, mean=X_mean, std=X_std, seed=seed, activation=Sin()).to(device)

# setup parameters
l0   = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
alp  = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
beta = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
gam0 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))
gam2 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device))

params = [
    {'params': net.parameters(), 'lr': 1e-3},
    {'params': l0, 'lr': 1e-3},
    {'params': alp, 'lr': 1e-3},
    {'params': beta, 'lr': 1e-3},
    {'params': gam0, 'lr': 1e-3},
    {'params': gam2, 'lr': 1e-3},
]

n_param = sum(p.numel() for p in net.parameters() if p.requires_grad)

def per_elem_grad(lf, net):
    with torch.no_grad():
        el_grad = torch.zeros((0), dtype=torch.float32, device=device)
        for grd in grad(lf, net.parameters(), retain_graph=True):
            el_grad = torch.cat([el_grad, grd.view(-1).detach()])
            
        return el_grad.cpu().data.numpy()

epochs = 7000

milestones = [[3000, 6000], [1000, 6000], [4000, 6000]]

optimizer = Adam(params)
scheduler = MultiStepLR(optimizer, milestones[ms], gamma=0.1)

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
        
        if i == 0:
            print("epoch", epoch, "batch", i, X_batch)
        
        optimizer.zero_grad()
        
        X_batch = X_batch.to(device).requires_grad_(True)
        Y_batch = Y_batch.to(device)
        
        # prediction error
        pred = net(X_batch)
        l_u  = torch.sum(torch.mean((Y_batch - pred[:,:2])**2, dim=0))
        
        l_reg1 = 0
        l_reg2 = 0
        if epoch >= pretrain:
            if epoch >= (pretrain + 1000):
                # turn on only after initial data fitting phase
                res1, res2, res3 = residual_pressure(pred, X_batch, l0, alp, beta, gam0, gam2)
                l_reg1 = torch.mean(res1**2) + torch.mean(res2**2)
                l_reg2 = torch.mean(res3**2)
            else:
                res3   = residual_div(pred, X_batch)
                l_reg2 = torch.mean(res3**2)
            
            if(method==0):
                # inverse dirichlet
                with torch.no_grad():
                    if epoch % mm == 0 and i == 0:
                        if epoch >= (pretrain + 1000):
                            stdu = loss_grad_std_full(l_u, net)
                            stdr1 = loss_grad_std_full(l_reg1, net)
                            stdr2 = loss_grad_std_full(l_reg2, net)

                            lamb_hat = stdr1/stdu
                            lamb = (1-alpha_ann)*lamb + alpha_ann*lamb_hat

                            lamb_hat = stdr1/stdr2
                            lambd = (1-alpha_ann)*lambd + alpha_ann*lamb_hat

                            lambs.append([lamb.item(), lambd.item()])
                        else:
                            stdu = loss_grad_std_full(l_u, net)
                            stdr2 = loss_grad_std_full(l_reg2, net)

                            lamb_hat = stdr2/stdu
                            lamb = (1-alpha_ann)*lamb + alpha_ann*lamb_hat

                            lambs.append([lamb.item(), 1.0])
                        
                loss = lamb*l_u + l_reg1 + lambd*l_reg2
            elif(method==2):
                # max avg
                with torch.no_grad():
                    if epoch % mm == 0 and i == 0:
                        if epoch >= (pretrain + 1000):
                            lambu_in = 1.0 if epoch == (pretrain + 1000) else lamb
                            maxr, _  = loss_grad_max_full(l_reg1, net)
                            _, meanu = loss_grad_max_full(l_u, net, lambg=lambu_in)
                            _, meanr = loss_grad_max_full(l_reg2, net, lambg=lambd)

                            lamb_hat = maxr/meanu
                            lamb     = (1-alpha_ann)*lamb + alpha_ann*lamb_hat

                            lamb_hat = maxr/meanr
                            lambd     = (1-alpha_ann)*lambd + alpha_ann*lamb_hat

                            lambs.append([lamb.item(), lambd.item()])
                        else:
                            maxr, _  = loss_grad_max_full(l_reg2, net)
                            _, meanu = loss_grad_max_full(l_u, net, lambg=lamb)

                            lamb_hat = maxr/meanu
                            lamb     = (1-alpha_ann)*lamb + alpha_ann*lamb_hat

                            lambs.append([lamb.item(), 1.0])
                        
                loss = lamb*l_u + l_reg1 + lambd*l_reg2
            elif(method==3):
                # mgda
                with torch.no_grad():
                    if epoch % mm == 0 and i == 0:
                        if epoch >= (pretrain + 1000):
                            G_reg1 = network_gradient(l_reg1, net)
                            G_reg2 = network_gradient(l_reg2, net)
                            G_fit  = network_gradient(l_u, net)

                            # construct gradient matrix
                            M = torch.zeros((torch.numel(G_reg1), 3), dtype=torch.float32, device=device)

                            M[:,0] = G_fit
                            M[:,1] = G_reg1
                            M[:,2] = G_reg2

                            # solve for optimal parameters
                            c_ = solver_mine(torch.matmul(M.T, M).cpu().numpy(), 3, tol, maxiter=1000)

                            lambs.append([c_[0].item(), c_[1].item(), c_[2].item()])
                        else:
                            G_reg1 = network_gradient(l_reg2, net)
                            G_fit  = network_gradient(l_u, net)

                            # construct gradient matrix
                            M = torch.zeros((torch.numel(G_reg1), 2), dtype=torch.float32, device=device)

                            M[:,0] = G_fit
                            M[:,1] = G_reg1

                            # solve for optimal parameters
                            c_ = solver_mine(torch.matmul(M.T, M).cpu().numpy(), 2, tol, maxiter=1000)
                            c_ = [c_[0], 0.0, c_[1]]
                            
                            lambs.append([c_[0], c_[1], c_[2]])
                      
                loss = c_[0]*l_u + c_[1]*l_reg1 + c_[2]*l_reg2
            else:
                # vanilla
                loss = l_u + l_reg1 + l_reg2
        
        if epoch < pretrain:
            # during pre-training only optimize fitting
            loss = l_u

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