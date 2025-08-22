#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is for 1D Nagumo equation,
    - The problem is a time-independent problem, solved as 2D problem.
    - 2D WNO is used here.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
directory = os.path.abspath(os.path.join(os.path.dirname('Data'), '.'))
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from wavelet_convolution import WaveConv2dCwt
from timeit import default_timer
from utilities import *
import stochatic_projection as stp
# import gradfree_fun 

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        padding   : scalar, size of zero padding
        """
        
        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.in_channel = in_channel
        self.padding = padding
        
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        self.fc0 = nn.Conv2d(self.in_channel, self.width, 1) # input channel is 2: (a(x), x)
        for hdim in range(self.layers):
            self.conv_layers.append(WaveConv2dCwt(self.width, self.width, self.level, self.size,
                                                  self.wavelet1, self.wavelet2))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        
        self.fc1 = nn.Conv2d(self.width, 512, 1)
        self.fc2 = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        x = self.fc0(x)                     # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) # padding, if required
        
        for index, (convl, wl) in enumerate( zip(self.conv_layers, self.w_layers) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.gelu(x)                # Shape: Batch * Channel * x * y

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding] # removing padding, when applicable
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

# %%
""" Load the data """
TRAIN_PATH = '/Data/Nagumo_129_129_1000.mat'
reader = scipy.io.loadmat(directory + TRAIN_PATH)
x = torch.tensor(reader['x'][:,0], dtype = torch.float)[None,None,:]   
t = torch.tensor(reader['t'][:,0], dtype = torch.float)[None,None,:]   
x_data = torch.tensor(reader['mat_ics'], dtype = torch.float)[:,None,:,:] 
y_data = torch.tensor(reader['sol'], dtype = torch.float)[:,None,:,:] 

# Interpolate the data to rectangle domain
x_data = F.interpolate(x_data, size=[128,128], mode='bicubic', align_corners=True).squeeze(1)
y_data = F.interpolate(y_data, size=[128,128], mode='bicubic', align_corners=True).squeeze(1)
x = F.interpolate(x, size=[128], mode='linear', align_corners=True)[0,0,:]
t = F.interpolate(t, size=[128], mode='linear', align_corners=True)[0,0,:]

""" Model configurations """
ntrain = 800
ntest = 50

batch_size = 25
learning_rate = 0.001
epochs = 200
step_size = 20                          # weight-decay step size
gamma = 0.5                             # weight-decay rate

wavelet = ['near_sym_b', 'qshift_b']    # wavelet basis function
level = 4                               # lavel of wavelet decomposition
width = 64                              # uplifting dimension
layers = 5                              # no of wavelet layers

subx, subt = 2, 2                    # subsampling rate
h = int(((x_data.shape[-2] - 1)/subt) + 1) # total grid size divided by the subsampling rate
s = int(((x_data.shape[-1] - 1)/subx) + 1)
in_channel = 3               # a(x, t), x, t for this case

# %%
""" Read data """
xgrid, tgrid = torch.meshgrid((t, x), indexing='ij')   
xgrid, tgrid = xgrid[::subt,::subx], tgrid[::subt,::subx]
xgrid, tgrid = xgrid.tile(ntrain+ntest,1,1), tgrid.tile(ntrain+ntest,1,1)

x_train = x_data[:ntrain,::subt,::subx]               # Shape: Batch * x * y
y_train = y_data[:ntrain,::subt,::subx]              # Shape: Batch * x * y

x_test = x_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y
y_test = y_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = torch.stack(( x_train, xgrid[:ntrain], tgrid[:ntrain] ), dim=1) # Shape: Batch * Channel * x * y
x_test = torch.stack(( x_test, xgrid[:ntest], tgrid[:ntest] ), dim=1)     # Shape: Batch * Channel * x * y

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# %%
""" Prepare Gradient-free derivative """
lb = np.array([0, 0])           # Lower-bound of the grid
ub = np.array([1, 1])           # Upper-bound of the grid

gf = stp.gradientfree(lb, ub, N_f=h)      # Stochastic-Gradient free derivative

# %%
""" The model definition """
model = WNO2d(width=width, level=level, layers=layers, size=[h,s], wavelet=wavelet,
              in_channel=in_channel).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)

myloss = LpLoss(size_average=False)
y_normalizer.to(device)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_pl = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        physics_loss = 0
        optimizer.zero_grad()
        
        out = model(x).reshape(batch_size, h, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
                
        physics_loss = gf.loss(y_pred=out, y_dash=y, lambda_bc=40)
            
        loss =  physics_loss
        loss.backward()
        optimizer.step()
        
        train_pl += physics_loss.item()
    
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            out = model(x).reshape(batch_size, h, s)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.reshape(batch_size,-1), y.reshape(batch_size,-1)).item()
            
    train_pl/= ntrain
    test_l2 /= ntest
    
    train_loss[ep] = train_pl
    test_loss[ep] = test_l2
    
    t2 = default_timer()
    print('Epoch %d - Time %0.4f - Physics (MSE) %0.6f - Test (L2 relative) %0.6f'
          % (ep, t2-t1, train_pl, test_l2))
    
# %%
""" Prediction on the test data """
pred = []
test_e = []
y_normalizer.to(device)

with torch.no_grad():
    index = 0
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
        
        out = model(x).reshape(batch_size, h, s)
        out = y_normalizer.decode(out)
        test_l2 = myloss(out.reshape(batch_size,-1), y.reshape(batch_size,-1)).item()

        pred.append( out.cpu() )
        test_e.append( test_l2/batch_size )
        
        print("Batch-{}, Loss-{}".format(index, test_l2/batch_size) )
        index += 1

pred = torch.cat(( pred ))
test_e = torch.tensor(( test_e ))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 14

""" Plotting """  
figure7, ax = plt.subplots(nrows=3, ncols=4, figsize=(14, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
index = 0
for value in range(ntest):
    if value % 15 == 0:
        cmin = torch.min(y_test[value,:,:])
        cmax = torch.max(y_test[value,:,:])
        
        im = ax[0,index].imshow(y_test[value,:,:], cmap='gist_ncar', interpolation='Gaussian',
                                vmin=cmin, vmax=cmax)
        ax[0,index].set_title('Sample-{}'.format(value)); 
        plt.colorbar(im, ax=ax[0,index], fraction=0.045)
        
        im = ax[1,index].imshow(pred[value,:,:], cmap='gist_ncar', interpolation='Gaussian',
                                vmin=cmin, vmax=cmax)
        plt.colorbar(im, ax=ax[1,index], fraction=0.045)
        
        im = ax[2,index].imshow(np.abs(pred[value,:,:] - y_test[value,:,:]), cmap='jet',
                           interpolation='Gaussian', vmin=0, vmax=0.01)
        plt.colorbar(im, ax=ax[2,index], fraction=0.045)
        
        if index == 0:
            ax[0,index].set_ylabel('Truth'); 
            ax[1,index].set_ylabel('Identified'); 
            ax[2,index].set_ylabel('Absolure Error'); 
        
        index = index + 1
        
# %%
""" For saving the trained model and prediction data """
torch.save(model, 'model/model_phy_cwno_nagumo')
scipy.io.savemat('pred/prediction_pinn_wno_nagumo.mat', mdict={'pred': pred.cpu().numpy()})
