#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is for 2D Allen-Cahn equation,
    - The problem is a time-dependent problem.
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
import matplotlib.animation as animation

from wavelet_convolution import WaveConv2dCwt
from timeit import default_timer
from utilities import *
import stochatic_projection as stp
# import gradfree_fun 

torch.manual_seed(0)
np.random.seed(0)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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
TRAIN_PATH = directory + '/Data/Allen_Cahn_pde_65_65_1100.mat'
reader = MatReader(TRAIN_PATH)
x = reader.read_field('x')[:,0] 
y = reader.read_field('x')[:,0] 
t = reader.read_field('t')[:,0] 
data = reader.read_field('sol').permute(3,2,0,1)   # Shape: Batch * Time * x * y

""" Model configurations """
ntrain = 700
ntest = 60

batch_size = 60
learning_rate = 0.0001
epochs = 200
step_size = 10                          # weight-decay step size
gamma = 0.5                             # weight-decay rate

wavelet = ['near_sym_b', 'qshift_b']    # wavelet basis function
level = 4                               # lavel of wavelet decomposition
width = 32                              # uplifting dimension
layers = 3                              # no of wavelet layers

sub = 1                                   # subsampling rate
h = int(((data.shape[-1] - 1)/sub) + 1) # total grid size divided by the subsampling rate
s = int(((data.shape[-2] - 1)/sub) + 1)

T_in = 10
T = 10
step = 1
in_channel = T+2               # a(x, y), x, y for this case

# %%
""" Prepare the data-loader """
xgrid, ygrid = torch.meshgrid((x, y))   
xgrid, ygrid = xgrid[::sub,::sub], ygrid[::sub,::sub]
xgrid, ygrid = xgrid.tile(ntrain+ntest,1,1,1), ygrid.tile(ntrain+ntest,1,1,1)

train_a = data[:ntrain, :T_in, ::sub, ::sub]
train_u = data[:ntrain, T_in:T+T_in, ::sub, ::sub]

test_a = data[-ntest:, :T_in, ::sub, ::sub]
test_u = data[-ntest:, T_in:T+T_in, ::sub, ::sub]

train_a = torch.cat((train_a, xgrid[:ntrain], ygrid[:ntrain]), dim=1)
test_a = torch.cat((test_a, xgrid[:ntest], ygrid[:ntest]), dim=1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = torch.load('model/ns_wno_allencan', map_location=device)
print(count_params(model))

myloss = LpLoss(size_average=False)
    
# %%
""" Prediction """
prediction = []
test_e = []
with torch.no_grad():
    index = 0
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_batch = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        t1 = default_timer()
        for t in range(0, T, step):
            y = yy[:, t:t + step, :]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)
            xx = torch.cat((xx[:, step:, :], im), dim=1)
        
        t2 = default_timer()
        
        prediction.append( pred.cpu() )
        test_l2_step = loss.item()
        test_l2_batch = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        
        test_e.append( test_l2_step/ntest/(T/step) )
        index += 1
        
        print("Batch-{}, Time-{}, Test-loss-step-{:0.6f}, Test-loss-batch-{:0.6f}".format(
            index, t2-t1, test_l2_step/ntest/(T/step), test_l2_batch/ntest) )
        
prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 16

""" Plotting """  
figure7, ax = plt.subplots(nrows=3, ncols=5, figsize=(16, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

batch_no = 55
index = 0
for tvalue in range(T):
    if tvalue % 2 == 0:
        cmin = torch.min(test_u[batch_no,tvalue,:,:])
        cmax = torch.max(test_u[batch_no,tvalue,:,:])
        
        im = ax[0,index].imshow(test_u[batch_no,tvalue,:,:], cmap='gist_ncar', interpolation='Gaussian',
                                vmin=cmin, vmax=cmax)
        ax[0,index].set_title('Time step-{}'.format(tvalue)); 
        plt.colorbar(im, ax=ax[0,index], fraction=0.045)
        
        im = ax[1,index].imshow(prediction[batch_no,tvalue,:,:], cmap='gist_ncar', interpolation='Gaussian',
                                vmin=cmin, vmax=cmax)
        plt.colorbar(im, ax=ax[1,index], fraction=0.045)
        
        im = ax[2,index].imshow(np.abs(test_u[batch_no,tvalue,:,:] - prediction[batch_no,tvalue,:,:]), cmap='jet',
                           interpolation='Gaussian', vmin=0.0, vmax=0.2)
        plt.colorbar(im, ax=ax[2,index], fraction=0.045)
        
        if index == 0:
            ax[0,index].set_ylabel('Truth'); 
            ax[1,index].set_ylabel('Identified'); 
            ax[2,index].set_ylabel('Absolure Error'); 
        
        index = index + 1

gg
# %%
""" Animation of Navier-Stokes """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fig9, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6), dpi=100)
plt.subplots_adjust(wspace=0.3)
sample = 25

def update(x, y, k):
    print(k) 
    cmin = np.min(x)
    cmax = np.max(x)
    ax[0].cla()
    ax[0].imshow(x, origin='upper', extent=[0,1,0,1], cmap='jet', interpolation='Gaussian',
                 vmin=cmin, vmax=cmax) 
    ax[0].set_title('(a) Ground Truth', y=1.05, fontweight='bold')
    
    ax[1].cla()
    ax[1].imshow(y, origin='upper', extent=[0,1,0,1], cmap='jet', interpolation='Gaussian',
                 vmin=cmin, vmax=cmax) 
    ax[1].set_title('(b) WNO', y=1.05, fontweight='bold')
    plt.suptitle('Allen-Cahn: Phase-field modelling, Grid size 64x64', fontweight='bold', fontsize=24)

def animate(k):
    update(test_u[sample,k,:,:].numpy(), prediction[sample,k,:,:].numpy(), k)
    
anim = animation.FuncAnimation(fig=fig9, func=animate, interval=1, 
                                      frames=T, repeat=False)
anim.save("Allen_Cahn.gif", fps=5)
