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
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# %%
""" Read data for sub-resolution """
subx, subt = 4, 4                    # subsampling rate
h = int(((x_data.shape[-2] - 1)/subt) + 1) # total grid size divided by the subsampling rate
s = int(((x_data.shape[-1] - 1)/subx) + 1)
in_channel = 3               # a(x, t), x, t for this case

xgrid, tgrid = torch.meshgrid((t, x), indexing='ij')   
xgrid, tgrid = xgrid[::subt,::subx], tgrid[::subt,::subx]
xgrid, tgrid = xgrid.tile(ntrain+ntest,1,1), tgrid.tile(ntrain+ntest,1,1)

x_train = x_data[:ntrain,::subt,::subx]               # Shape: Batch * x * y
y_train = y_data[:ntrain,::subt,::subx]              # Shape: Batch * x * y

x_test = x_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y
y_test_sub = y_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y

x_normalizer_sub = UnitGaussianNormalizer(x_train)
x_train = x_normalizer_sub.encode(x_train)
x_test = x_normalizer_sub.encode(x_test)

y_normalizer_sub = UnitGaussianNormalizer(y_train)
y_train = y_normalizer_sub.encode(y_train)

x_train = torch.stack(( x_train, xgrid[:ntrain], tgrid[:ntrain] ), dim=1) # Shape: Batch * Channel * x * y
x_test = torch.stack(( x_test, xgrid[:ntest], tgrid[:ntest] ), dim=1)     # Shape: Batch * Channel * x * y

train_loader_sub = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader_sub = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test_sub),
                                          batch_size=batch_size, shuffle=False)

# %%
""" Read data for train-resolution """
subx, subt = 2, 2                    # subsampling rate
h = int(((x_data.shape[-2] - 1)/subt) + 1) # total grid size divided by the subsampling rate
s = int(((x_data.shape[-1] - 1)/subx) + 1)
in_channel = 3               # a(x, t), x, t for this case

xgrid, tgrid = torch.meshgrid((t, x), indexing='ij')   
xgrid, tgrid = xgrid[::subt,::subx], tgrid[::subt,::subx]
xgrid, tgrid = xgrid.tile(ntrain+ntest,1,1), tgrid.tile(ntrain+ntest,1,1)

x_train = x_data[:ntrain,::subt,::subx]               # Shape: Batch * x * y
y_train = y_data[:ntrain,::subt,::subx]              # Shape: Batch * x * y

x_test = x_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y
y_test_train = y_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y

x_normalizer_train = UnitGaussianNormalizer(x_train)
x_train = x_normalizer_train.encode(x_train)
x_test = x_normalizer_train.encode(x_test)

y_normalizer_train = UnitGaussianNormalizer(y_train)
y_train = y_normalizer_train.encode(y_train)

x_train = torch.stack(( x_train, xgrid[:ntrain], tgrid[:ntrain] ), dim=1) # Shape: Batch * Channel * x * y
x_test = torch.stack(( x_test, xgrid[:ntest], tgrid[:ntest] ), dim=1)     # Shape: Batch * Channel * x * y

train_loader_actual = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader_actual = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test_train),
                                          batch_size=batch_size, shuffle=False)

# %%
""" Read data for Super-resolution """
subx, subt = 1, 1                    # subsampling rate
h = int(((x_data.shape[-2] - 1)/subt) + 1) # total grid size divided by the subsampling rate
s = int(((x_data.shape[-1] - 1)/subx) + 1)
in_channel = 3               # a(x, t), x, t for this case

xgrid, tgrid = torch.meshgrid((t, x), indexing='ij')   
xgrid, tgrid = xgrid[::subt,::subx], tgrid[::subt,::subx]
xgrid, tgrid = xgrid.tile(ntrain+ntest,1,1), tgrid.tile(ntrain+ntest,1,1)

x_train = x_data[:ntrain,::subt,::subx]               # Shape: Batch * x * y
y_train = y_data[:ntrain,::subt,::subx]              # Shape: Batch * x * y

x_test = x_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y
y_test_super = y_data[ntrain:ntrain + ntest,::subt,::subx]  # Shape: Batch * x * y

x_normalizer_super = UnitGaussianNormalizer(x_train)
x_train = x_normalizer_super.encode(x_train)
x_test = x_normalizer_super.encode(x_test)

y_normalizer_super = UnitGaussianNormalizer(y_train)
y_train = y_normalizer_super.encode(y_train)

x_train = torch.stack(( x_train, xgrid[:ntrain], tgrid[:ntrain] ), dim=1) # Shape: Batch * Channel * x * y
x_test = torch.stack(( x_test, xgrid[:ntest], tgrid[:ntest] ), dim=1)     # Shape: Batch * Channel * x * y

train_loader_super = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader_super = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test_super),
                                          batch_size=batch_size, shuffle=False)

# %%
y_test = [y_test_sub, y_test_train, y_test_super]
test_loader = [test_loader_sub, test_loader_actual, test_loader_super]
y_normalizer = [y_normalizer_sub, y_normalizer_train, y_normalizer_super]
cases = ['Sub-Resolution', 'Train-Resolution', 'Super-Resolution']
resol = [32, 64, 128]

# %%
""" The model definition """
model = torch.load('model/model_phy_cwno_nagumo', map_location=device)
print(count_params(model))

myloss = LpLoss(size_average=False)
    
# %%
""" Prediction on the test data """
pred_all = []
test_e_all = []

with torch.no_grad():
    for case in range(len(cases)):
        y_normalizer[case].to(device)
        index = 0
        pred = []
        test_e = []
        
        for x, y in test_loader[case]:
            test_l2 = 0
            x, y = x.to(device), y.to(device)
            
            t1 = default_timer()
            out = model(x).squeeze(1)
            out = y_normalizer[case].decode(out)
            t2 = default_timer()
            
            test_l2 = myloss(out.reshape(batch_size,-1), y.reshape(batch_size,-1)).item()
    
            pred.append( out.cpu() )
            test_e.append( test_l2/batch_size )
            
            print("Case-{}, Batch-{}, Time-{}, Loss-{}".format(cases[case], index, t2-t1, test_l2/batch_size) )
            index += 1
            
        pred = torch.cat(( pred ))
        test_e = torch.tensor(( test_e ))  
        print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
        
        pred_all.append(pred)
        test_e_all.append(test_e)

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 20

""" Zero-Shot Super-resolution, with sub == 1 in the data loading """  
figure8, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.35)
sample = 15

for case in range(len(cases)):
    cmin = torch.min(y_test[case][sample,:,:])
    cmax = torch.max(y_test[case][sample,:,:])
    
    im = ax[0,case].imshow(y_test[case][sample,:,:], cmap='gist_ncar', interpolation='Gaussian',
                               vmin=cmin, vmax=cmax, extent=[0,1,1,0])
    ax[0,case].set_title('{}\nTest on {} x {} grid'.format(cases[case], resol[case], resol[case])); 
    plt.colorbar(im, ax=ax[0,case], aspect=10)
    ax[0,case].set_xlabel('x', fontweight='bold')
    ax[0,case].set_ylabel('t', color='m', fontweight='bold')
        
    im = ax[1,case].imshow(pred_all[case][sample,:,:], cmap='gist_ncar', interpolation='Gaussian',
                            vmin=cmin, vmax=cmax, extent=[0,1,1,0])
    plt.colorbar(im, ax=ax[1,case], aspect=10)
    ax[1,case].set_xlabel('x', fontweight='bold')
    ax[1,case].set_ylabel('t', color='m', fontweight='bold')
        
    im = ax[2,case].imshow(np.abs(y_test[case][sample,:,:] - pred_all[case][sample,:,:]), cmap='jet',
                        interpolation='Gaussian', vmin=0, vmax=0.02, extent=[0,1,1,0])
    plt.colorbar(im, ax=ax[2,case], aspect=10)
    ax[2,case].set_xlabel('x', fontweight='bold')
    ax[2,case].set_ylabel('t', color='m', fontweight='bold')
        
figure8.text(0.05,0.75, 'Truth', color='r', rotation=90, fontweight='bold'); 
figure8.text(0.05,0.45, 'Prediction', color='g', rotation=90, fontweight='bold'); 
figure8.text(0.05,0.14, 'Absolute Error', color='purple', rotation=90, fontweight='bold'); 
    
figure8.suptitle('Nagumo Equation (Train on: 64 (space) x 64 (time) grid)',
                  fontweight='bold', fontsize=32, x=0.51, y=1.01)

# figure8.savefig('Nagumo_super.png', format='png', dpi=300, bbox_inches='tight')
