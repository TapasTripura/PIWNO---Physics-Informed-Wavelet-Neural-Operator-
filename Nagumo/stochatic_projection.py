"""
This code defines the stochastic projection-based gradients to obtain the derivatives of the output

See the paper:
    Stochastic projection based approach for gradient free physics informed learning
    by N Navaneeth, Souvik Chakraborty
    Github link :: https://github.com/csccm-iitd/SP-PINN
    
"""

import torch
import torch.nn as nn
import numpy as np

class gradientfree(nn.Module):
    def __init__(self, lb, ub, N_f, nr=9, radius=0.024):
        super(gradientfree, self).__init__()  
        
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.nr = nr                        # Neighbourhood points
        self.radius = radius                # Projection radius
        
        self.xt1 = lb[0] + (ub[0]-lb[0]) * np.linspace(0,1,N_f)
        self.yt1 = lb[1] + (ub[1]-lb[1]) * np.linspace(0,1,N_f)
        self.Xt1, self.Yt1 = np.meshgrid(self.xt1, self.yt1)
        self.X_f_train = np.hstack([self.Xt1.reshape(N_f*N_f,1), self.Yt1.reshape(N_f*N_f,1)])
        self.x_f_train = torch.tensor(self.X_f_train, dtype=torch.float) 
        
        self.p_index = self.neighbour_index(self.x_f_train)
        self.invp_index = self.inverse_index(self.x_f_train)
        
    def neighbour_index(self,X_f_train):
        """
        Gets the indices of the neighbourhood points in the meshgrid
        
        Parameters
        ----------
        X_f_train : tensor, shape-(Nx * Ny, 2), meshgrid points.

        Returns
        -------
        zn : tensor, shape-(Nx * Ny, nr), neighbourhood index points.
        """
        device = X_f_train.device
        zn = torch.zeros([X_f_train.shape[0],self.nr], dtype=torch.int32).to(device)
        for j in range(X_f_train.shape[0]): 
            ngr1 = (torch.ones([1,self.nr], dtype=torch.int32)*j)[0].tolist()
            x = X_f_train[j,:][None,:]
            ngr1_d= torch.where(torch.linalg.norm(X_f_train-x,axis=1) < self.radius)[0].tolist()
            ngr1[0:len(ngr1_d)] = ngr1_d
            zn[j,:] = torch.tensor(ngr1)[0:self.nr]
        return zn 
    
    def inverse_index(self,X_f_train):
        """
        Estimates the denominator of the gradient
        
        Parameters
        ----------
        X_f_train : tensor, shape-(Nx * Ny, 2), meshgrid points.

        Returns
        -------
        zmd : tensor, shape-(Nx * Ny, 2, 2), indices for denominator of gradient.
        """
        device = X_f_train.device
        zn = torch.zeros([X_f_train.shape[0],self.nr],dtype=torch.int32).to(device)
        zmd = torch.zeros([X_f_train.shape[0],X_f_train.shape[1],X_f_train.shape[1]]).to(device)
        for j in range(X_f_train.shape[0]): 
                ngr1 = (torch.ones([1,self.nr],dtype=torch.int32)*j)[0].tolist()
                x = X_f_train[j,:][None,:]
                n = x.shape[1]
                md = torch.zeros((n,n)).to(device)
                ngr1_d= torch.where(torch.linalg.norm(X_f_train-x,axis=1)<self.radius)[0].tolist()
                ngr1[0:len(ngr1_d)] = ngr1_d
                zn[j,:] = torch.tensor(ngr1)[0:self.nr]
                xd = X_f_train[ngr1_d[0:self.nr],:][None,:]-x;
                md = torch.einsum('abi,abj->aij',xd,xd)
                md_inv = torch.inverse(md)
                zmd[j,:,:] = md_inv
        return zmd
    
    def loss_BC(self,up,usol):
        """
        Estimates the boundary/data loss
        
        Parameters
        ----------
        up : tensor, predicted boundary values.
        usol : tensor, true boundary values.

        Returns
        -------
        loss_u : scalar, boundary loss.
        """
        loss_u = self.loss_function(up,usol)
        return loss_u
    
    def grad1(self,xx,u_0,n_index,inv_mat):
        """

        Parameters
        ----------
        xx : tensor, shape-(Nx * Ny, 2), grid points.
        u_0 : tensor, shape-(Batch, Nx * Ny, 1), prediction field.
        n_index : tensor, shape-(Nx * Ny, nr), neighbourhood index points.
        inv_mat : tensor, shape-(Nx * Ny, 2, 2), indices for denominator of gradient.

        Returns
        -------
        zd : tensor, shape-(batch, Nx * Ny, 2), 
            first-order derivatives: u_x, u_y.
        """
        zd = torch.zeros(xx.shape, dtype=torch.float)
        m = n_index.shape[1]
        x_ngr = xx[n_index.tolist(),:]
        u_ngr = u_0[:,n_index.tolist(),:]
        x_d  = x_ngr-xx.unsqueeze(1).repeat(1,m,1)
        u_d = u_ngr-u_0.unsqueeze(2).repeat(1,1,m,1)
        u_ds = torch.sum(torch.einsum('lijp,ikp->likp',(u_d.permute(0,1,3,2)),(x_d.permute(0,2,1))),dim=3)
        zd = torch.einsum('qkj,kji->qki',u_ds,inv_mat)
        return zd
    
    def grad2(self,xx,u_x_t,n_index,inv_mat):
        """

        Parameters
        ----------
        xx : tensor, shape-(Nx * Ny, 2), grid points.
        u_x_t : tensor, shape-(Batch, Nx * Ny, 1), 
            first-order derivative of prediction field.
        n_index : tensor, shape-(Nx * Ny, nr), neighbourhood index points.
        inv_mat : tensor, shape-(Nx * Ny, 2, 2), indices for denominator of gradient.

        Returns
        -------
        zdd : tensor, shape-(batch, Nx * Ny, 4), 
            second-order derivatives: u_xx, u_xy, u_yx, u_yy.
        """
        zdd = torch.zeros(u_x_t.shape[0], 2*u_x_t.shape[1],dtype=torch.float)
        s =  u_x_t.shape[0]
        n = u_x_t.shape[1]
        m = n_index.shape[1]
        x_ngr = xx[n_index.tolist(),:]
        u_xngr = u_x_t[:,n_index.tolist(),:]
        x_d  = x_ngr-xx.unsqueeze(1).repeat(1,m,1)
        u_xd = u_xngr-u_x_t.unsqueeze(2).repeat(1,1,m,1)
        u_xds = torch.sum(torch.einsum('lijr,ipr->lijpr',(u_xd.permute(0,1,3,2)),(x_d.permute(0,2,1))),dim=4)
        zdd = torch.einsum('qkij,kjl->qkil',u_xds,inv_mat).reshape([s,n,4])
        return zdd
    
    def loss_PDE(self,x_to_train_f,u,n_index,inv_mat):
        """
        Returns the PDE loss on the prediction 
        
        Parameters
        ----------
        x_to_train_f : tensor, shape-(Nx * Ny, 2), grid points.
        u : tensor, shape-(Batch, Nx * Ny, 1), prediction field.
        n_index : tensor, shape-(Nx * Ny, 2), neighbourhood index points.
        inv_mat : tensor, shape-(Nx * Ny, 2, 2), index points for denominator.

        Returns
        -------
        loss_f : scalar, PDE loss.
        """
        nu = 0.08
        alpha = 0.5
        
        u_x_y = self.grad1(x_to_train_f,u,n_index,inv_mat)                                        
        u_xx_yy = self.grad2(x_to_train_f,u_x_y,n_index,inv_mat)     
                                                                                                             
        # u_x = u_x_y[:,:,[0]]
        u_y = u_x_y[:,:,[1]]
        
        u_xx = u_xx_yy[:,:,[0]]  
        # u_xy = u_xx_yy[:,:,[1]]  
        # u_yx = u_xx_yy[:,:,[2]]  
        # u_yy = u_xx_yy[:,:,[3]]  
        
        f = u_y - nu*u_xx - u*(1-u)*(u + alpha)
        
        f_hat =  torch.zeros(f.shape).to(x_to_train_f.device)
        loss_f = self.loss_function(f,f_hat)
        return  loss_f     

    def loss(self,y_pred,y_dash,lambda_bc=1,lambda_pde=1):
        """
        Returns the total loss on prediction

        Parameters
        ----------
        y_pred : tensor, Predicted field from WNO.
        y_dash : tensor, True solution field, only the boundary values are used.
        lambda_bc : scalar, optional
            Participation factor of boundary loss. The default is 1.
        lambda_pde : TYPE, optional
            Participation factor of PDE loss. The default is 1.

        Returns
        -------
        loss_val : scalar, Total MSE loss.
        """
        device = y_dash.device
        batch_size, N_f = y_pred.shape[:2]
        
        # bound condition 
        tp_u = y_pred[:,0,:][:,:,None]
        tp_usol = y_dash[:,0,:][:,:,None]       
        lt_u =  y_pred[:,:,0][:,:,None]  
        lt_usol = y_dash[:,:,0][:,:,None]  
        rt_u = y_pred[:,:,-1][:,:,None]
        rt_usol = y_dash[:,:,-1][:,:,None]
        
        up = torch.hstack([tp_u,lt_u,rt_u])
        usol = torch.hstack([tp_usol,lt_usol,rt_usol])
        
        # Physics including the boundary
        u = y_pred.reshape(batch_size, N_f*N_f, 1)
        
        loss_u = self.loss_BC(up,usol)
        loss_f = self.loss_PDE(self.x_f_train.to(device), u,
                               self.p_index.to(device), self.invp_index.to(device))
        
        loss_val = lambda_bc*loss_u + lambda_pde*loss_f
        return loss_val
   