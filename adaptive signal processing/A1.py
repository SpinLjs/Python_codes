# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:41:00 2018

@author: ljs41
"""

###################  main library imports #####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from numpy import random
from scipy import linalg

###################  main program #############################################

################################### 1-5 ########################################
N=10000;
stdd_s=10**(1/2);
stdd_v=1;
s_n=np.random.normal(loc=0.0,scale=stdd_s,size=N);
g=np.array([1,1]);
h=np.array([1,2,1]);
acrl_s=np.array([10]);
acrl_x=signal.convolve(signal.convolve(acrl_s,g),g[::-1]);
print('acrl_x=',acrl_x);
ccrl_dx=signal.convolve(acrl_x,h);  # origin at 2
print('ccrl_dx=',ccrl_dx);
acrl_d=signal.convolve(ccrl_dx,h[::-1]);
sigma_d2=acrl_d[3]+1;
print('sigma_d2=',sigma_d2);


P=np.reshape(ccrl_dx[1::1],(1,-1)).T;
teo_up=np.hstack((acrl_x[1::1],np.zeros(2)));
teo_lw=np.hstack((acrl_x[1::-1],np.zeros(2)));
R=linalg.toeplitz(teo_lw,teo_up);
R_inv=linalg.inv(R);
W_opt=R_inv.dot(P)[:-1]

MMSE=sigma_d2-(P.T.dot(R_inv)).dot(P);

W_opt=np.vstack((W_opt,np.zeros(1)));
MSE_4=sigma_d2-2*W_opt.T.dot(P)+W_opt.T.dot(R).dot(W_opt);

x_n=signal.convolve(s_n,g);
t_n=signal.convolve(x_n,h);
v_n=np.random.normal(loc=0.0,scale=stdd_v,size=len(t_n));
d_n=t_n+v_n;
y_n=signal.convolve(x_n,W_opt.ravel());
e_n=d_n-y_n[:-1];
MSE_5=np.sum(e_n**2)/len(e_n);
print('MSE_5=',MSE_5);

f, axarr = plt.subplots(2, sharex=False)
n=np.linspace(0, len(x_n)-1, len(x_n))
axarr[0].plot(n, x_n)
axarr[0].set_title('x(n)')
axarr[0].set_ylabel('')
axarr[0].set_xlabel('n')
axarr[0].grid()

n=np.linspace(0, len(d_n)-1, len(d_n))
axarr[1].plot(n, d_n)
axarr[1].set_title('d(n)')
axarr[1].set_ylabel('')
axarr[1].set_xlabel('n')
axarr[1].grid()

f, axarr = plt.subplots(2, sharex=False)
n=np.linspace(0, len(y_n)-1, len(y_n))
axarr[0].plot(n, y_n)
axarr[0].set_title('y(n)')
axarr[0].set_ylabel('')
axarr[0].set_xlabel('n')
axarr[0].grid()

n=np.linspace(0, len(e_n)-1, len(d_n))
axarr[1].plot(n, e_n)
axarr[1].set_title('e(n)')
axarr[1].set_ylabel('')
axarr[1].set_xlabel('n')
axarr[1].grid()

################################### 6-8 ########################################
acrl_x_est=np.correlate(x_n[:N],x_n[:N],"full ")

f, axarr = plt.subplots(2, sharex=False)
n=np.linspace(0, len(acrl_x_est)-1, len(acrl_x_est))
axarr[0].plot(n, acrl_x_est)
axarr[0].set_title(r'$\phi_{xx}$')
axarr[0].set_ylabel(r'$\phi_{xx}$')
axarr[0].set_xlabel('')
axarr[0].grid()

acrl_x_est=acrl_x_est[N-1:];
coef=np.arange(N,0,-1)
acrl_x_est=acrl_x_est/coef;

ccrl_dx_est=np.correlate(d_n[:N],x_n[:N],"full ")

n=np.linspace(0, len(ccrl_dx_est)-1, len(ccrl_dx_est))
axarr[1].plot(n, ccrl_dx_est)
axarr[1].set_title(r'$\phi_{dx}$')
axarr[1].set_ylabel(r'$\phi_{dx}$')
axarr[1].set_xlabel('n')
axarr[1].grid()

ccrl_dx_est=ccrl_dx_est[N-1:];
coef=np.arange(N,0,-1)
ccrl_dx_est=ccrl_dx_est/coef;

acrl_x_est_sym=np.hstack((acrl_x_est[:0:-1],acrl_x_est))

ccrl_dx_est_sym=np.hstack((ccrl_dx_est[:0:-1],ccrl_dx_est))

f, axarr = plt.subplots(2, sharex=False)
n=np.linspace(0, len(acrl_x_est_sym)-1, len(acrl_x_est_sym))
axarr[0].plot(n, acrl_x_est_sym)
axarr[0].set_title('Unbiased estimate of'+r'$\phi_{xx}$')
axarr[0].set_ylabel(r'$\phi_{xx}$')
axarr[0].set_xlabel('')
axarr[0].grid()

n=np.linspace(0, len(ccrl_dx_est_sym)-1, len(ccrl_dx_est_sym))
axarr[1].plot(n, ccrl_dx_est_sym)
axarr[1].set_title('Unbiased estimate of'+r'$\phi_{dx}$')
axarr[1].set_ylabel(r'$\phi_{dx}$')
axarr[1].set_xlabel('n')
axarr[1].grid()


P7=np.reshape(ccrl_dx_est[:100],(1,-1)).T;
teo_up=acrl_x_est[:100];
teo_lw=acrl_x_est[:100];
R7=linalg.toeplitz(teo_lw,teo_up);
R7_inv=linalg.inv(R7);

W_opt7=R7_inv.dot(P7)[:3];
W_opt7=np.vstack((W_opt7,0))            

#acrl_x_est=np.correlate(x_n,x_n,"full ")
#acrl_x_est=acrl_x_est[int(len(x_n)-1):];
#acrl_x_est=acrl_x_est/len(acrl_x_est);
#acrl_x_est_sym=np.hstack((acrl_x_est[:0:-1],acrl_x_est))
#
#ccrl_dx_est=np.correlate(d_n,x_n,"full ")
#ccrl_dx_est=ccrl_dx_est[int(len(x_n)-1):];
#ccrl_dx_est=ccrl_dx_est/len(ccrl_dx_est);
#
#P7=np.reshape(ccrl_dx_est,(1,-1)).T;
#teo_up=np.hstack((acrl_x_est,np.zeros(2)));
#teo_lw=np.hstack((acrl_x_est,np.zeros(2)));
#R7=linalg.toeplitz(teo_lw,teo_up);
#R7_inv=linalg.inv(R7);
#
#W_opt7=R7_inv.dot(P7);

MSE_8=sigma_d2-2*W_opt7.T.dot(P)+W_opt7.T.dot(R).dot(W_opt7);


#######################  9 -10   #############################################
x_n=x_n[:N]
d_n=d_n[:N]

nperseg=250;
overlap=125;
nfft=512;
freqtmp, Pxx = signal.csd(x_n, x_n, fs=1.0, window='hann', nperseg=nperseg, noverlap=overlap, nfft=nfft,
   detrend=False, return_onesided=False, scaling='density', axis=-1) # csd() normalizes window rms value

freqtmp, Pdx = signal.csd(d_n, x_n, fs=1.0, window='hann', nperseg=nperseg, noverlap=overlap, nfft=nfft,
   detrend=False, return_onesided=False, scaling='density', axis=-1) # csd() normalizes window rms value

W_optf=Pdx/Pxx;

f, axarr = plt.subplots(2, sharex=False)
k=np.linspace(0, len(Pxx)-1, len(Pxx))
axarr[0].plot(k, Pxx)
axarr[0].set_title('Weich method: '+r'$P_{xx}$')
axarr[0].set_ylabel(r'$P_{xx}$')
axarr[0].set_xlabel('')
axarr[0].grid()

axarr[1].plot(k, Pdx)
axarr[1].set_title('Weich method: '+r'$P_{dx}$')
axarr[1].set_ylabel(r'$P_{dx}$')
axarr[1].set_xlabel('k')
axarr[1].grid()

f, axarr = plt.subplots(2, sharex=False)
axarr[0].plot(k, Pxx)
axarr[0].set_title('Weich method: '+r'$W_{k}$')
axarr[0].set_ylabel(r'$W_{k}$')
axarr[0].set_xlabel('k')
axarr[0].grid()


W_opt9=np.real(fftpack.ifft(W_optf))

axarr[1].plot(k, W_opt9)
axarr[1].set_title('IFFT result: '+r'$W_{n}$')
axarr[1].set_ylabel('')
axarr[1].set_xlabel('n')
axarr[1].grid()


W_opt9=fftpack.ifftshift(W_opt9)[int(nfft/2-2):int(nfft/2+1)]
W_opt9=np.reshape(W_opt9,(1,-1)).T
W_opt9=np.vstack((W_opt9,0)) 



MSE_10=sigma_d2-2*W_opt9.T.dot(P)+W_opt9.T.dot(R).dot(W_opt9);






    