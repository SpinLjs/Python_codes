# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:36:24 2018

@author: ljs41
"""

###################  main library imports #####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import random
from scipy import linalg
from scipy.io import wavfile

###################  main program #############################################

N=3;    #3-taps
n_it=400000;       # iteration number <= len(d)

L_lp=1000;
h_lp=np.ones([L_lp+1])/L_lp;
################################ read #########################################

rate,d_n = wavfile.read('assign3_d.wav')

rate,x_n = wavfile.read('assign3_x.wav')

d_n=np.float32(d_n);
x_n=np.float32(x_n);

Ds_n=np.convolve(d_n[0:n_it]**2,h_lp);
#%%############################ 1.Weiner ####################################    
sigma_d2=np.correlate(d_n,d_n)/len(x_n);

acrl_x=np.zeros([N]);
ccrl_xd=np.zeros([N]);
for i in range(N):
    x_1=np.pad(x_n,(i,0), 'constant', constant_values=(0, 0));
    x_2=np.pad(x_n,(0,i), 'constant', constant_values=(0, 0));
    d_2=np.pad(d_n,(0,i), 'constant', constant_values=(0, 0));
    acrl_x[i]=np.correlate(x_1,x_2)/len(x_n);
    ccrl_xd[i]=np.correlate(x_1,d_2)/len(x_n);

R=linalg.toeplitz(acrl_x);
R=np.matrix(R);
R_inv=linalg.inv(R);
p=np.matrix(ccrl_xd).T;

MMSE=sigma_d2-p.T*R_inv*p;
MMSE=MMSE/sigma_d2;

#%%############################# 2.SD/Newton ####################################   
u_Newton=1;
u_sd=1/N/acrl_x[0];

W_sd=np.zeros([N,n_it+1]);
W_Newton=np.zeros([N,n_it+1]);
Err_sd=np.ones([n_it]);
Err_Newton=np.ones([n_it]);

x_pad=np.pad(x_n,(N-1,0), 'constant', constant_values=(0, 0));

for i in range(0,n_it):
    d_tmp=d_n[i];
    x_tmp=np.matrix(np.flip(x_pad[i:i+N],0)).T
    
    ################## steepest descent #######################
    W_tmp=np.matrix(W_sd[:,i]).T;
    W_tmp2=W_tmp-u_sd*(R*W_tmp-p);
    W_sd[:,i+1]=W_tmp2.A1;
    
    e_tmp=d_tmp-x_tmp.T*W_tmp;
    
    Err_sd[i]=e_tmp;
    
    ################# Newton ###################################  
    W_tmp=np.matrix(W_Newton[:,i]).T;
    W_tmp2=W_tmp-u_Newton*R_inv*(R*W_tmp-p);
    W_Newton[:,i+1]=W_tmp2.A1;
    
    e_tmp=d_tmp-x_tmp.T*W_tmp;
    
    Err_Newton[i]=e_tmp;
    
Es_sd=np.convolve(Err_sd**2,h_lp);
Es_Newton=np.convolve(Err_Newton**2,h_lp);

#%%########################### 3.Stochastic ####################################

u_LMS=0.11/3/N/acrl_x[0];
u_NtLMS=0.11/3/N;
u_NLMS=0.018;

x_pad=np.pad(x_n,(N-1,0), 'constant', constant_values=(0, 0));

W_LMS=np.zeros([N,n_it+1]);
W_NtLMS=np.zeros([N,n_it+1]);
W_NLMS=np.zeros([N,n_it+1]);

Err_LMS=np.ones([n_it]);
Err_NtLMS=np.ones([n_it]);
Err_NLMS=np.ones([n_it]);


psi_NLMS=1000;

e_LMS=np.zeros([n_it]);
e_NtLMS=np.zeros([n_it]);
e_NLMS=np.zeros([n_it]);

for i in range(0,n_it):
    d_tmp=d_n[i];
    x_tmp=np.matrix(np.flip(x_pad[i:i+N],0)).T
    
    ################## LMS #######################
    W_tmp=np.matrix(W_LMS[:,i]).T;
    e_tmp=d_tmp-W_tmp.T*x_tmp;
    W_tmp2=W_tmp+u_LMS*x_tmp*e_tmp;
    W_LMS[:,i+1]=W_tmp2.A1;
    
    Err_LMS[i]=e_tmp;
    
    ################# Newton LMS ###################################  
    W_tmp=np.matrix(W_NtLMS[:,i]).T;
    e_tmp=d_tmp-W_tmp.T*x_tmp;
    W_tmp2=W_tmp+u_NtLMS*R_inv*x_tmp*e_tmp;
    W_NtLMS[:,i+1]=W_tmp2.A1;

    Err_NtLMS[i]=e_tmp;
    
    ################# Normalized LMS ###################################  
    W_tmp=np.matrix(W_NLMS[:,i]).T;
    e_tmp=d_tmp-W_tmp.T*x_tmp;
    scl_norm=1/(x_tmp.T*x_tmp+psi_NLMS);    
    W_tmp2=W_tmp+u_NLMS*x_tmp*e_tmp*scl_norm;
    W_NLMS[:,i+1]=W_tmp2.A1;
    
    Err_NLMS[i]=e_tmp;
    
Es_LMS=np.convolve(Err_LMS**2,h_lp);
Es_NtLMS=np.convolve(Err_NtLMS**2,h_lp);
Es_NLMS=np.convolve(Err_NLMS**2,h_lp);
#%%########################### 4.Affine Projection ####################################

u_AP=0.0015;
M=3;

d_pad_ap=np.pad(d_n,(M-1,0), 'constant', constant_values=(0, 0));
x_pad_ap=np.pad(x_n,(N+M-2,0), 'constant', constant_values=(0, 0));


X_matrix=np.zeros([N,n_it+M-1])
for i in range(0,n_it+M-1):
    X_matrix[:,i]=x_pad_ap[i:i+N];
X_matrix=np.flip(np.flip(X_matrix,0),1);

W_AP=np.zeros([N,n_it+1]);
Err_AP=np.ones([n_it]);

psi_ap=1000;
const_mtrx=np.matrix(np.diag(psi_ap*np.ones([M])));

for i in range(0,n_it):
    d_vect=np.matrix(np.flip(d_pad_ap[i:i+M],0));
    X_mtrx=np.matrix(X_matrix[:,n_it-i-1:n_it-i+M-1]);
    
    W_tmp=np.matrix(W_AP[:,i]).T;
    e_vect=d_vect-W_tmp.T*X_mtrx;
    M_inv=linalg.inv(X_mtrx.T*X_mtrx+const_mtrx);
    W_tmp2=W_tmp+u_AP*X_mtrx*M_inv*e_vect.T;
    W_AP[:,i+1]=W_tmp2.A1;
    
    Err_AP[i]=e_vect[0,0];

Es_AP=np.convolve(Err_AP**2,h_lp);

#%%########################## 5.Recursive Least Square ####################################

lambda_RLS=0.999;
delta=1e3;

Psi_inv_record = [];

x_pad_RLS=np.pad(x_n,(N-1,0), 'constant', constant_values=(0, 0));

W_RLS=np.zeros([N,n_it+1]);
Err_RLS=np.ones([n_it]);

Psi_inv_init=np.matrix(np.diag(delta*np.ones([N])));
Psi_inv=Psi_inv_init;

for i in range(0,n_it):
    d_tmp=d_n[i];
    x_tmp=np.matrix(np.flip(x_pad_RLS[i:i+N],0)).T
    
    W_tmp=np.matrix(W_RLS[:,i]).T;
    
    U_n=Psi_inv*x_tmp;
    e_n=d_tmp-x_tmp.T*W_tmp;
    k_n=U_n/(lambda_RLS+x_tmp.T*U_n);
    
    Psi_inv2=(Psi_inv-k_n*(x_tmp.T*Psi_inv))/lambda_RLS;    # symmetric in 400000 iters
#    Psi_inv2=(Psi_inv-k_n*U_n.T)/lambda_RLS;    # non-strict symmetric
    Psi_inv=Psi_inv2;
    Psi_inv_record.append(Psi_inv);
    
    W_tmp2=W_tmp+k_n*e_n;
    W_RLS[:,i+1]=W_tmp2.A1;

    Err_RLS[i]=e_n;

Es_RLS=np.convolve(Err_RLS**2,h_lp);    
#%%############################## plot & save #################################
n_dlt=np.int32(np.floor(L_lp/2));
################################### subplots ##########################
#fig, axarr = plt.subplots(2, sharex=False)
#axarr[0].plot(10*np.log10(Es_sd[0:-n_dlt]/Ds_n[0:-n_dlt]));
#axarr[0].set_title('Learning curve for SD')
#axarr[0].set_ylabel('')
#axarr[0].set_xlabel('')
#axarr[0].grid()
#
#axarr[1].plot(10*np.log10(Es_Newton[0:-n_dlt]/Ds_n[0:-n_dlt]))
#axarr[1].set_title('Learning curve for Newton')
#axarr[1].set_ylabel('')
#axarr[1].set_xlabel('n')
#axarr[1].grid()
#
#fig, axarr = plt.subplots(3, sharex=False)
#axarr[0].plot(10*np.log10(Es_LMS[0:-n_dlt]/Ds_n[0:-n_dlt]));
#axarr[0].set_title('Learning curve for LMS')
#axarr[0].set_ylabel('')
#axarr[0].set_xlabel('')
#axarr[0].grid()
#
#axarr[1].plot(10*np.log10(Es_NtLMS[0:-n_dlt]/Ds_n[0:-n_dlt]))
#axarr[1].set_title('Learning curve for Newton LMS')
#axarr[1].set_ylabel('')
#axarr[1].set_xlabel('')
#axarr[1].grid()
#
#axarr[2].plot(10*np.log10(Es_NLMS[0:-n_dlt]/Ds_n[0:-n_dlt]))
#axarr[2].set_title('Learning curve for NLMS')
#axarr[2].set_ylabel('')
#axarr[2].set_xlabel('n')
#axarr[2].grid()
#
#fig, axarr = plt.subplots(2, sharex=False)
#axarr[0].plot(10*np.log10(Es_AP[0:-n_dlt]/Ds_n[0:-n_dlt]));
#axarr[0].set_title('Learning curve for AP')
#axarr[0].set_ylabel('')
#axarr[0].set_xlabel('')
#axarr[0].grid()
#
#axarr[1].plot(10*np.log10(Es_RLS[0:-n_dlt]/Ds_n[0:-n_dlt]))
#axarr[1].set_title('Learning curve for Recursive LS')
#axarr[1].set_ylabel('')
#axarr[1].set_xlabel('n')
#axarr[1].grid()

#############################  save WAV. file ##########################
wavfile.write('assign4_e_SD.wav',rate,np.int16(Err_sd));
wavfile.write('assign4_e_Newton.wav',rate,np.int16(Err_Newton));
wavfile.write('assign4_e_LMS.wav',rate,np.int16(Err_LMS));
wavfile.write('assign4_e_NtLMS.wav',rate,np.int16(Err_NtLMS));
wavfile.write('assign4_e_NLMS.wav',rate,np.int16(Err_NLMS));
wavfile.write('assign4_e_AP.wav',rate,np.int16(Err_AP));
wavfile.write('assign4_e_RLS.wav',rate,np.int16(Err_RLS));

################################### Seperate plots ##########################
plt.figure();
plt.plot(10*np.log10(Es_sd[0:-n_dlt]/Ds_n[0:-n_dlt]));
plt.title('Learning curve for SD')
plt.ylabel('')
plt.xlabel('n')
plt.grid()

plt.figure();
plt.plot(10*np.log10(Es_Newton[0:-n_dlt]/Ds_n[0:-n_dlt]))
plt.title('Learning curve for Newton')
plt.ylabel('')
plt.xlabel('n')
plt.grid()

plt.figure();
plt.plot(10*np.log10(Es_LMS[0:-n_dlt]/Ds_n[0:-n_dlt]));
plt.title('Learning curve for LMS')
plt.ylabel('')
plt.xlabel('n')
plt.grid()

plt.figure();
plt.plot(10*np.log10(Es_NtLMS[0:-n_dlt]/Ds_n[0:-n_dlt]))
plt.title('Learning curve for Newton LMS')
plt.ylabel('')
plt.xlabel('n')
plt.grid()

plt.figure();
plt.plot(10*np.log10(Es_NLMS[0:-n_dlt]/Ds_n[0:-n_dlt]))
plt.title('Learning curve for NLMS')
plt.ylabel('')
plt.xlabel('n')
plt.grid()

plt.figure();
plt.plot(10*np.log10(Es_AP[0:-n_dlt]/Ds_n[0:-n_dlt]));
plt.title('Learning curve for AP')
plt.ylabel('')
plt.xlabel('n')
plt.grid()

plt.figure();
plt.plot(10*np.log10(Es_RLS[0:-n_dlt]/Ds_n[0:-n_dlt]))
plt.title('Learning curve for Recursive LS')
plt.ylabel('')
plt.xlabel('n')
plt.grid()
