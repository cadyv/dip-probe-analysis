import RingdownFunctions as ringdown
import os

# Load data
directory = '/Volumes/DMRADIO-E/run-in-progress/hiQ/20210915_HQ_dip/Ringdowns'
file = '20210915_174100_RINGDOWN_5_CYCLE_2_832_50kHz_150_mV_noFLL.csv'
filename = os.path.join(directory, file) # add the filename for your scope trace csv here
channels = [1,2] # pick what channels from the scope trace you want, currently assumes channel 3 is the func gen and 1 is SQUID
data = ringdown.getScopeTraceData(filename, channels, savePLTs=True)

samplerate = 2.5e6
f_approx = 832361.119

## ______ PICK WHICH METHOD(S) TO USE ________________
lockinFit = False
sineFit = False
FFTFit = True

savePLTs = True

## -------- IF DOING DIRECT SINE FIT ----------------
if sineFit:
    times = data[:,0]
    amp = data[:,1]
    trig = data[:,2] # this assumes that the trigger (the function gen output) is the third column

    p0 = None # initial conditions guess for fit (can leave this blank)

    ## these settings are described better in RingdownFunctions.py
    startbuffer = 0.5
    endbuffer = 0.1
    fitbuffer = 5
    fitfactor = 1.5

    Qav, Qstd, Fav, Fstd, tauav, taustd = ringdown.fitRingdownSine(times, amp,
                    trig, samplerate, f_approx, p0, startbuffer,
                    endbuffer, fitbuffer, fitfactor, savePLTs)

## --------- IF DOING LOCKIN FIT -----------------------
if lockinFit:
    dec = 10
    useCarEdges=True
    mph = 0.4
    buffer = 0.005
    amplitude_SNR_thresh=2
    max_expected_detuning=1500


    lidata = ringdown.getLockinDataFromTimestream(data, f_approx, dec, savePLTs=True)
    t = lidata[:,0]-lidata[0,0]
    r = lidata[:,1]
    phi = lidata[:,2]
    carr = lidata[:,3]
    carphi = lidata[:,4]
    Qav, Qstd, Fav, Fstd, tauav, taustd = ringdown.fitRingdownLI(t, r, phi, carr,
                    carphi, f_approx, useCarEdges, mph, buffer,
                    amplitude_SNR_thresh, max_expected_detuning,savePLTs)

## ----------- IF DOING FFT FIT -------------------------------
if FFTFit:
    
