# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:26:25 2018

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

############### custom stft function ############################################
def my_stft(x , nperseg=1024, nshift=128, nseg=500):
    x_stft = np.zeros((nperseg,nseg),dtype='complex');
    for i in range(0,nseg):
        tmp = x[i*nshift:i*nshift+nperseg]*np.hanning(nperseg);
        tmp_ft = fftpack.fft(tmp,nperseg)
        x_stft[:,i]=tmp_ft.T;
    return  x_stft;



################################ read #########################################

rate,d_n = wavfile.read('assign2_d.wav')

rate,x_n = wavfile.read('assign2_x.wav')

############################# 1 ###############################################
n_seg=500;
nshift=128;
nperseg=1024;
noverlap=nperseg-nshift;

N_t = (n_seg-1)*nshift+nperseg;

d_stft = my_stft(d_n, nperseg=nperseg, nshift=nshift, nseg=n_seg);

x_stft = my_stft(x_n, nperseg=nperseg, nshift=nshift, nseg=n_seg);
#
#f,t,d_stft = signal.stft(d_n, fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap, 
#    nfft=None, detrend=False, return_onesided=False, boundary='zeros', padded=True, axis=-1)
#
#f,t,x_stft = signal.stft(x_n, fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap, 
#    nfft=None, detrend=False, return_onesided=False, boundary='zeros', padded=True, axis=-1)


N=5;

p=np.empty([N,1]);
W_opt=np.empty([N,1]);
E_ddH=np.empty([1,1]);

MMSE_1=np.empty([nperseg]);
MMSE_2=np.empty([nperseg]);
y_stft=np.empty([nperseg,n_seg],dtype=complex)

for k in range(0,nperseg):
    xdH_k=np.correlate(x_stft[k,:],d_stft[k,:],'full');
    xxH_k=np.correlate(x_stft[k,:],x_stft[k,:],'full');
    E_ddH_k = np.correlate(d_stft[k,:],d_stft[k,:],'valid')/n_seg;
    
    E_ddH=np.concatenate((E_ddH, np.reshape(E_ddH_k,(1,-1))), axis=1)
    
    p_k=xdH_k[n_seg-1:n_seg-N-1:-1]/n_seg;
    p=np.concatenate((p, np.reshape(p_k,(N,-1))), axis=1)
    
    r_kv=xxH_k[n_seg-1:n_seg-N-1:-1]/n_seg;
    R_k=linalg.toeplitz(r_kv);
    R_kinv=linalg.inv(R_k);
    W_kopt=R_kinv.dot(p_k);
    W_opt=np.concatenate((W_opt, np.reshape(W_kopt,(N,-1))), axis=1);
    
    p_k=np.asmatrix(p_k);
    W_kopt=np.asmatrix(W_kopt);
    MMSE_1k=E_ddH_k-p_k.conj()*W_kopt.T;
    MMSE_1k=MMSE_1k/E_ddH_k;
#    MMSE=np.concatenate((MMSE, MMSE_k), axis=1)
    MMSE_1[k]=np.asscalar(np.real(MMSE_1k))
      
    ################### 2 ########################################
    M_xknN=np.empty([1,n_seg]);
    for i in range(0,N):
        temp=np.pad(x_stft[k,:], (i,0), 'constant', constant_values=(0, 0))[0:n_seg];
        M_xknN=np.concatenate((M_xknN, np.reshape(temp,(1,-1))), axis=0)
    M_xknN = np.matrix(M_xknN[1:,:]);
    y_kn=W_kopt.conj()*M_xknN;
    e_kn=d_stft[k,:]-y_kn;
    e_kn=np.abs(e_kn.A);
    
    MMSE_2k = np.sum(e_kn**2)/n_seg/E_ddH_k;
    MMSE_2[k]=np.asscalar(np.real(MMSE_2k))
    
    #################### 3 ##########################################
    
    y_stft[k,:]=y_kn;

y_n = np.zeros([N_t])
d_n2 = np.zeros([N_t])
y_t = fftpack.ifft(y_stft,axis=-2);
y_t = np.real(y_t);
d_t2 = fftpack.ifft(d_stft,axis=-2);
d_t2 = np.real(d_t2);


for i_t in range(0,n_seg):
    temp = y_t[:,i_t];
    y_n[i_t*nshift:i_t*nshift+nperseg] += temp.flatten();
    temp = d_t2[:,i_t];
    d_n2[i_t*nshift:i_t*nshift+nperseg] += temp.flatten();

y_n=y_n/4;

#_,y_n=signal.istft(y_stft, fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap, 
#    nfft=None, input_onesided=False)

e_n = d_n - y_n;
MMSE_3 = np.sum(np.abs(e_n[noverlap:-noverlap])**2)/(N_t-2*noverlap);
E_dn=np.correlate(d_n.astype('float')[noverlap:-noverlap],d_n.astype('float')[noverlap:-noverlap],'valid')/(N_t-2*noverlap);
MMSE_3 = MMSE_3/E_dn;

MMSE_3_dB = 10*np.log10(MMSE_3)

print('MMSE_3 in dB:',MMSE_3_dB)

    
########################## plot ####################################
f = fftpack.fftfreq(nperseg);
f=fftpack.fftshift(f)

fig=plt.figure();
plt.pcolormesh(np.arange(0,n_seg), f, fftpack.fftshift(np.abs(d_stft),axes=(0,)))

fig, axarr = plt.subplots(2, sharex=False)
axarr[0].plot(f, fftpack.fftshift(MMSE_1))
axarr[0].set_title('MMSE(k) for Q1:')
axarr[0].set_ylabel('')
axarr[0].set_xlabel(r'$f$')
axarr[0].grid()

axarr[1].plot(f, 10*np.log10(fftpack.fftshift(MMSE_1)))
axarr[1].set_title('MMSE(k) for Q1 (in dB)')
axarr[1].set_ylabel('dB')
axarr[1].set_xlabel(r'$f$')
axarr[1].grid()
    
fig, axarr = plt.subplots(2, sharex=False)
axarr[0].plot(f, fftpack.fftshift(MMSE_2))
axarr[0].set_title('MMSE(k) for Q2:')
axarr[0].set_ylabel('')
axarr[0].set_xlabel(r'$f$')
axarr[0].grid()

axarr[1].plot(f, 10*np.log10(fftpack.fftshift(MMSE_2)))
axarr[1].set_title('MMSE(k) for Q2 (in dB)')
axarr[1].set_ylabel('dB')
axarr[1].set_xlabel(r'$f$')
axarr[1].grid()

fig=plt.figure()
plt.plot(e_n)
plt.title('Error in time serise: e(n)')
plt.xlabel('n')

wavfile.write('assign2_y.wav',rate,y_n);