# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 03:10:11 2018

@author: ljs41
"""

###################  main library imports #####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from numpy import random
from scipy import linalg
from scipy.io import wavfile

###################  main program #############################################

N=3;    #3-taps
n_it=500;
u_Newton=0.25;
################################ read #########################################

rate,d_n = wavfile.read('assign3_d.wav')

rate,x_n = wavfile.read('assign3_x.wav')

d_n=np.float32(d_n);
x_n=np.float32(x_n);
#%%########################## 1.theoretic ###############################################
a=0.95;
A=1/(1-a**2);

acrl_x=np.zeros([N]);
for i in range(N):
    acrl_x[i]=a**i*A;
print('acrl_x=',acrl_x);

R_th=linalg.toeplitz(acrl_x);
R_th=np.matrix(R_th);
R_th_inv=linalg.inv(R_th);

G=np.array([5,5/3]);
I=np.array([3,3/4]);

F=np.zeros([len(G)]);
for i in range(len(F)):
    F[i]=a**i;

acrl_s=np.array([1]);
acrl_v=np.array([1]);
acrl_d=signal.convolve(signal.convolve(acrl_v,G),G[::-1])+signal.convolve(signal.convolve(acrl_s,I),I[::-1]);
sigma_d2=acrl_d[int((len(acrl_d)-1)/2)];
print('acrl_d=',acrl_d);
ccrl_xd=signal.convolve(signal.convolve(acrl_v,G[::-1]),F);
print('ccrl_xd=',ccrl_xd);
p_th=np.pad(ccrl_xd[-2::-1],(0,N-len(G)), 'constant', constant_values=(0, 0));
p_th=np.matrix(p_th).T;

u_sd=0.25/N/acrl_x[0];

W_sd=np.zeros([N,n_it+1]);
W_Newton=np.zeros([N,n_it+1]);
MSE_sd=np.ones([n_it+1]);
MSE_Newton=np.ones([n_it+1]);

for i in range(1,n_it+1):
    ################## steepest descent #######################
    W_tmp=np.matrix(W_sd[:,i-1]).T;
    W_tmp2=W_tmp-u_sd*(R_th*W_tmp-p_th);
    W_sd[:,i]=W_tmp2.A1;
    MSE_temp=sigma_d2-2*W_tmp2.T*p_th+W_tmp2.T*R_th*W_tmp2;
    MSE_sd[i]=MSE_temp/sigma_d2;
    ################# Newton ###################################  
    W_tmp=np.matrix(W_Newton[:,i-1]).T;
    W_tmp2=W_tmp-u_Newton*R_th_inv*(R_th*W_tmp-p_th);
    W_Newton[:,i]=W_tmp2.A1;
    MSE_temp=sigma_d2-2*W_tmp2.T*p_th+W_tmp2.T*R_th*W_tmp2;
    MSE_Newton[i]=MSE_temp/sigma_d2;

    ######################## plot ####################################
fig, axarr = plt.subplots(2, sharex=False)
axarr[0].plot(MSE_sd);
axarr[0].set_title('MSE for SD')
axarr[0].set_ylabel('')
axarr[0].set_xlabel('')
axarr[0].grid()

axarr[1].plot(MSE_Newton)
axarr[1].set_title('MSE for Newton')
axarr[1].set_ylabel('')
axarr[1].set_xlabel('k')
axarr[1].grid()  
#%%########################### 2.measured ####################################    
sigma_d22=np.correlate(d_n,d_n);

acrl_x=np.zeros([N]);
ccrl_xd=np.zeros([N]);
for i in range(N):
    x_1=np.pad(x_n,(i,0), 'constant', constant_values=(0, 0));
    x_2=np.pad(x_n,(0,i), 'constant', constant_values=(0, 0));
    d_2=np.pad(d_n,(0,i), 'constant', constant_values=(0, 0));
    acrl_x[i]=np.correlate(x_1,x_2);
    ccrl_xd[i]=np.correlate(x_1,d_2);

R=linalg.toeplitz(acrl_x);
R=np.matrix(R);
R_inv=linalg.inv(R);
p=np.matrix(ccrl_xd).T;

u_sd2=0.25/N/acrl_x[0];

W_sd2=np.zeros([N,n_it+1]);
W_Newton2=np.zeros([N,n_it+1]);
MSE_sd2=np.ones([n_it+1]);
MSE_Newton2=np.ones([n_it+1]);

for i in range(1,n_it+1):
    ################## steepest descent #######################
    W_tmp=np.matrix(W_sd2[:,i-1]).T;
    W_tmp2=W_tmp-u_sd2*(R*W_tmp-p);
    W_sd2[:,i]=W_tmp2.A1;
    MSE_temp=sigma_d22-2*W_tmp2.T*p+W_tmp2.T*R*W_tmp2;
    MSE_sd2[i]=MSE_temp/sigma_d22;
    ################# Newton ###################################  
    W_tmp=np.matrix(W_Newton2[:,i-1]).T;
    W_tmp2=W_tmp-u_Newton*R_inv*(R*W_tmp-p);
    W_Newton2[:,i]=W_tmp2.A1;
    MSE_temp=sigma_d22-2*W_tmp2.T*p+W_tmp2.T*R*W_tmp2;
    MSE_Newton2[i]=MSE_temp/sigma_d22;
    
    ######################## plot ####################################

fig, axarr = plt.subplots(2, sharex=False)
axarr[0].plot(MSE_sd2);
axarr[0].set_title('MSE for SD')
axarr[0].set_ylabel('')
axarr[0].set_xlabel('')
axarr[0].grid()

axarr[1].plot(MSE_Newton2)
axarr[1].set_title('MSE for Newton')
axarr[1].set_ylabel('')
axarr[1].set_xlabel('k')
axarr[1].grid()

W_opt = W_Newton2[:,-1];    
y_n=np.convolve(x_n,W_opt);
e_n=d_n-y_n[:-2:];
e_refer=d_n-x_n;
wavfile.write('assign3_y.wav',rate,np.int16(y_n));
wavfile.write('assign3_e.wav',rate,np.int16(e_n));