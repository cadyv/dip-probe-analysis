import RingdownFunctions as ringdown
import matplotlib.pyplot as mpl
import numpy as np
import importlib
import time

mpl.close('all')

importlib.reload(ringdown)

bias_points =['b00','b01','b02','b03','b04','b05','b06']
phib_list = [5.9, 0, -9, -18, -34.06, -38.99 ,-47.26]

directory = 'Ringdown_6BP'
directory2 = 'Ringdown_6BP_72GHz'

samplerate = 10e6
injection_freq = 675500
amplitude_SNR_thresh = 0.5
mph = 0.5
savetimestreamPLTs = False
savePLTs = False
Qplot = False


# Parameters for fit using FFTs
binfactor = 10
overlap = 0.9

# Parameters for fit using Lockin
decimation = 10
useCarEdges = True
buffer = 0.002
max_expected_detuning = 15000

# Parameters for fit using decaying sine fit
startbuffer = 0.5
endbuffer = 0.1
fitbuffer = 5
fitfactor = 1.5

t1 = time.time()
FFT = ringdown.FFTRingdownDirectory(directory, bias_points,
                                                samplerate, injection_freq, binfactor, overlap,
                                                mph, buffer, amplitude_SNR_thresh, savePLTs,
                                                savetimestreamPLTs, Qplot,phib_list)
FFT2 = ringdown.FFTRingdownDirectory(directory2, bias_points,
                                                samplerate, injection_freq, binfactor, overlap,
                                                mph, buffer, amplitude_SNR_thresh, savePLTs,
                                                savetimestreamPLTs, Qplot,phib_list)
t2 = time.time()
LI = ringdown.LIRingdownDirectory(directory, bias_points,
                                                samplerate, injection_freq, decimation,
                                                useCarEdges, mph, buffer, amplitude_SNR_thresh,
                                                max_expected_detuning, savePLTs, Qplot,phib_list)
LI2 = ringdown.LIRingdownDirectory(directory2, bias_points,
                                                samplerate, injection_freq, decimation,
                                                useCarEdges, mph, buffer, amplitude_SNR_thresh,
                                                max_expected_detuning, savePLTs, Qplot,phib_list)

t3 = time.time()

SD = ringdown.SineRingdownDirectory(directory, bias_points,
                                                samplerate, injection_freq, startbuffer, endbuffer, fitbuffer,
                                                fitfactor, savePLTs, Qplot,phib_list)
SD2 = ringdown.SineRingdownDirectory(directory2, bias_points,
                                                samplerate, injection_freq, startbuffer, endbuffer, fitbuffer,
                                                fitfactor, savePLTs, Qplot,phib_list)

t4 = time.time()

print(20*'~')
print(20*'~')
print('FFT method took '+str(t2-t1) + ' seconds')
print('Lockin method took '+str(t3-t2)+ ' seconds')
print('Sine fit method took '+str(t4-t3)+' seconds')
print(20*'~')
print(20*'~')

fig1, ax1 = mpl.subplots()
ax1.errorbar(phib_list, FFT[0], yerr=FFT[1], fmt='o',capsize=10,c='thistle',label='FFT method, 4.3GHz GBP')
ax1.errorbar(phib_list, LI[0], yerr=LI[1], fmt='o',capsize=10, c='lavender',label='Lockin method, 4.3GHz GBP')
ax1.errorbar(phib_list, FFT2[0], yerr=FFT2[1], fmt='o',capsize=10,c='lightsteelblue',label='FFT method, 7.2GHz GBP')
ax1.errorbar(phib_list, LI2[0], yerr=LI2[1], fmt='o',capsize=10, c='paleturquoise',label='Lockin method, 7.2GHz GBP')
ax1.errorbar(phib_list, SD[0], yerr=SD[1], fmt='o',capsize=10, c='magenta',label='Sine decay method, 4.3GHz GBP')
ax1.errorbar(phib_list, SD2[0], yerr=SD2[1], fmt='o',capsize=10, c='blue',label='Sine decay method, 7.2GHz GBP')
ax1.set_xlabel('Phib')
ax1.set_ylabel('Q')
ax1.set_title('Quality factor across SQ1 VPhi')
ax1.legend()
fig1.show()

fig2, ax2 = mpl.subplots()
ax2.errorbar(phib_list, FFT[4], yerr=FFT[5], fmt='o',capsize=10,c='thistle',label='FFT method, 4.3GHz GBP')
ax2.errorbar(phib_list, LI[4], yerr=LI[5], fmt='o',capsize=10, c='lavender',label='Lockin method, 4.3GHz GBP')
ax2.errorbar(phib_list, FFT2[4], yerr=FFT2[5], fmt='o',capsize=10,c='lightsteelblue',label='FFT method, 7.2GHz GBP')
ax2.errorbar(phib_list, LI2[4], yerr=LI2[5], fmt='o',capsize=10, c='paleturquoise',label='Lockin method, 7.2GHz GBP')
ax2.errorbar(phib_list, SD[4], yerr=SD[5], fmt='o',capsize=10, c='magenta',label='Sine decay method, 4.3GHz GBP')
ax2.errorbar(phib_list, SD2[4], yerr=SD2[5], fmt='o',capsize=10, c='blue',label='Sine decay method, 7.2GHz GBP')
ax2.set_xlabel('Phib')
ax2.set_ylabel('Frequency')
ax2.set_title('Frequency across SQ1 VPhi')
ax2.legend()
fig2.show()

fig3, ax3 = mpl.subplots()
ax3.errorbar(phib_list, FFT[2], yerr=FFT[3], fmt='o',capsize=10,c='thistle',label='FFT method, 4.3GHz GBP')
ax3.errorbar(phib_list, LI[2], yerr=LI[3], fmt='o',capsize=10, c='lavender',label='Lockin method, 4.3GHz GBP')
ax3.errorbar(phib_list, FFT2[2], yerr=FFT2[3], fmt='o',capsize=10,c='lightsteelblue',label='FFT method, 7.2GHz GBP')
ax3.errorbar(phib_list, LI2[2], yerr=LI2[3], fmt='o',capsize=10, c='paleturquoise',label='Lockin method, 7.2GHz GBP')
ax3.errorbar(phib_list, SD[2], yerr=SD[3], fmt='o',capsize=10, c='magenta',label='Sine decay method, 4.3GHz GBP')
ax3.errorbar(phib_list, SD2[2], yerr=SD2[3], fmt='o',capsize=10, c='blue',label='Sine decay method, 7.2GHz GBP')
ax3.set_xlabel('Phib')
ax3.set_ylabel('Tau')
ax3.set_title('Tau across SQ1 VPhi')
ax3.legend()
fig3.show()
