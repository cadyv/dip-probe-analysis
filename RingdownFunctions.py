import os
import math
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.animation as ani
from scipy.optimize import curve_fit
from scipy import signal
import fitting as ft
import lockin_cady as lockin
from detect_peaks import detect_peaks as findpks

# ~~~~~~~~~~~~~ Direct fit with exponentially decaying sine method ~~~~~~~~~~~
def SineRingdownDirectory(directory, bias_points, samplerate, f_approx,
                            startbuffer, endbuffer, fitbuffer, fitfactor,
                            savePLTs=True, Qplot=False, phib = None):

    ###########################################################################
    #  Read all the files in the directory specified, and perform direct fits
    # of the timestreams with a exponentially decaying sine to extract the
    # quality factor and frequency of a resonator excited by near-resonant
    # pulses.
    ###########################################################################

    # directory: the relative path from the current process_directory
    # bias_points: a list of bias points (style: 'b00') to analyze. Eventually,
    #     this should be replaced with a path to the bias points json file, and
    #     extract both bias point names and properties from there
    # samplerate: sample rate, in Hz, of the raw data from the digitizer
    # f_approx: approximate frequency, in Hz, of the resonance
    # startbuffer: Value between 0 and 1. Sets fraction of samples (in a period
    #     between falling edge andnext rising edge) to discard before fitting
    #     data to find noise level
    # endbuffer: Value between 0 and 1, with startbuffer+endbuffer < 1. sets
    #     fraction of samples to discard at the end of a noise sample before
    #     fitting
    # fitbuffer: Number of samples to discard after a falling edge before
    #     fitting a decaying sine function
    # fitfactor: Value which, multiplied by the calculated noise level, sets
    #     the cutoff for minimum amplitude at which to fit the decaying sine
    #     function
    # savePLTs: whether to make plots of the data and fits
    # Qplot: Whether to show plots of Q, F, and Tau varying with bias points
    #     in the directory
    # Phib: list of phibs corresponding to each bias point, to be used to for
    #     plotting

    Qlist = []
    Qstdlist = []
    Taulist = []
    Taustdlist = []
    Flist = []
    Fstdlist = []

    N = len(bias_points)
    for ind in np.arange(N):
        bp = bias_points[ind]
        print(20*'~')
        print('Analyzing Bias Point '+bp)
        print(20*'~')

        filepath = directory + '/step_0-initial-'+bp+'_FLL-ringdown-FLL/raw-data'
        filename = filepath+'/'+os.listdir(filepath)[0]
        data = getUnscaledTimestreamData(filename, samplerate, savePLTs=False)
        times = data[:,0]
        amp = data[:,1]
        trig = data[:,2]

        p0 = [0, 1.48e-5, f_approx, 0]

        # Fit timestream with decaying sine wave fit and return parameters
        Qav, Qstd, Fav, Fstd, tauav, taustd = fitRingdownSine(times, amp, trig,
                                                    samplerate, f_approx, p0,
                                                    startbuffer, endbuffer,
                                                    fitbuffer, fitfactor,
                                                    savePLTs)

        Qlist.append(Qav)
        Qstdlist.append(Qstd)
        Taulist.append(tauav)
        Taustdlist.append(taustd)
        Flist.append(Fav)
        Fstdlist.append(Fstd)

    # Return values of Q for each bias point
    for ind in np.arange(N):
        print('Q for '+bias_points[ind]+': '+str(Qlist[ind]))

    # Plot values of Q, Frequency, and Tau across all bias points, with error
    # bars given by 1 std
    if Qplot:
        if phib is not None:
            xdata = phib
            xlabel = 'SQ1 Phib'
        else:
            xdata = np.arange(N)
            xlabel = 'Bias point'
        fig1, ax1 = mpl.subplots()
        ax1.errorbar(xdata, Qlist, yerr=Qstdlist, fmt='o',capsize=10)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Q (error bars = 1std)')
        ax1.set_title('Quality factor across SQ1 VPhi')
        fig1.show()

        fig2, ax2 = mpl.subplots()
        ax2.errorbar(xdata, Flist, yerr=Fstdlist, fmt='o',capsize=10)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Frequency (error bars = 1std)')
        ax2.set_title('Frequency across SQ1 VPhi')
        fig2.show()

        fig3, ax3 = mpl.subplots()
        ax3.errorbar(xdata, Taulist, yerr=Taustdlist, fmt='o',capsize=10)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Tau (error bars = 1std)')
        ax3.set_title('Tau across SQ1 VPhi')
        fig3.show()
    return Qlist, Qstdlist, Taulist, Taustdlist, Flist, Fstdlist

def fitRingdownSine(times, amp, trig, samplerate, f_approx, p0, startbuffer,
                    endbuffer, fitbuffer, fitfactor, savePLTs):

    ###########################################################################
    #  Takes in a timestream, amplitude, and trigger data and returns the
    # average parameters from an exponentially decaying sine fit directly to the
    # timestream data
    ###########################################################################

    # times: array of timestream data
    # amp: array of amplitude data for signal to be fit. Does not need to be
    #     scaled
    # trigger: array of data to be used for the trigger (usually, this is the
    #     function generator, which is channel2 from the digitizer)
    # samplerate: sample rate, in Hz, of the raw data from the digitizer
    # f_approx: approximate frequency, in Hz, of the resonance
    # p0: initial parameter guesses for exponentially decaying sine fit.
    #           p0[0] = amplitude, A
    #           p0[1] = time constant, Tau
    #           p0[2] = frequency in Hz, f
    #           p0[3] = phase offset in radians, phi
    # startbuffer: Value between 0 and 1. Sets fraction of samples (in a period
    #     between falling edge andnext rising edge) to discard before fitting
    #     data to find noise level
    # endbuffer: Value between 0 and 1, with startbuffer+endbuffer < 1. sets
    #     fraction of samples to discard at the end of a noise sample before
    #     fitting
    # fitbuffer: Number of samples to discard after a falling edge before
    #     fitting a decaying sine function
    # fitfactor: Value which, multiplied by the calculated noise level, sets
    #     the cutoff for minimum amplitude at which to fit the decaying sine
    #     function
    # savePLTs: whether to make plots of the data and fits

    # Perform a highpass butterworth filter to the amplitude data, to remove
    # any DC offset or drift
    hp_filt = signal.butter(2, f_approx/10, 'highpass',
                            fs = samplerate, output = 'sos')
    amp_hp = signal.sosfilt(hp_filt, amp)

    # Plot the original and filtered data
    if savePLTs:
        fig2, ax2 = mpl.subplots()
        ax2.plot(times, amp, label = 'data')
        ax2.plot(times, amp_hp, label='hp filtered data')
        ax2.legend(loc='lower right')
        fig2.show()

    # Find rising and falling edges to identify bursts
    tPos, tNeg = findEdges(times, trig, f_approx)
    numBursts = min(np.size(tPos), np.size(tNeg))

    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ax1.plot(times, amp_hp, label = 'hp filtered data')

    # Find the burst period and total acquisition length
    burstPeriod = np.median(np.diff(tPos)) # seconds
    acqLength = max(times)-min(times)  # acquisition length, in seconds

    # Report results from detections of bursts
    print('Detected rising edges at: '+str(tPos))
    print('Detected falling edges at: '+str(tNeg))
    print('Detected '+str(numBursts)+' full bursts')
    print('Burst period : '+'%.3f' % (burstPeriod)+' seconds')
    print('Total acquisition length: '+'%.2f' % (acqLength)+' seconds')

    loop_data = np.zeros((0, 3))

    # For each ringdown, fit an exponentially decaying sine wave to the
    # amplitude data
    for burst in range(numBursts):
        print(20*'~')
        print('Fitting Burst '+str(burst))
        tic = time.time()

        # Find the ringdown falling edge and corresponding index in the
        # timestream
        tN = tNeg[burst]
        minind = int(np.argwhere(times == tN))

        # Find the end of the region of interest, which is either the next
        # falling edge, or the end of the timestream
        try:
            maxind = int(np.argwhere(times == tNeg[burst+1]))
        except:
            maxind = -1

        # Truncate timestream data to the region of interest
        tsubset = times[minind:maxind]
        ampsubset = amp_hp[minind:maxind]

        # Take the RMS average of the amplitude between consecutive bursts to
        # find the noise level
        try:
            tP = tPos[min(np.argwhere(tPos > tN))]
            noisetime = tP-tN
            if noisetime < burstPeriod: # ensures only fitting noise and not burst
                NoiseLevel, startnoise, endnoise = findNoise(tN, tP, tsubset,
                                                    ampsubset, startbuffer,
                                                    endbuffer)
            else:
                NoiseLevel, startnoise, endnoise = findNoise(tN, tP, times,
                                                    amp_hp, 0.8, 0.09)
        except:
            print('using previous noise level')
        print('Noise level = '+'%.2f' % (NoiseLevel))
        fitthreshhold = NoiseLevel*fitfactor

        # Set bounds for fit
        minbounds = [0, 0, 0.95*f_approx, 0]
        maxbounds = [1.05*max(ampsubset), 1, 1.05*f_approx, 2*np.pi]
        bounds = (minbounds, maxbounds)

        # Fit ringdown
        try:
            popt, a, t, timesfit = fitBurstSineDecay(tN, tsubset, ampsubset, fitbuffer,
                                    fitthreshhold, bounds, p0, savePLTs=False)
        except:
            popt, a, t, timesfit = fitBurstSineDecay(tN, tsubset, ampsubset, fitbuffer,
                                    fitthreshhold, bounds=(-np.inf, np.inf), p0=p0, savePLTs=False)
        p0 = popt # set new initial conditions for the next fit

        # Extract fit parameters and calculate Q
        tau = popt[1]
        frequency = popt[2]
        Q = tau * frequency * np.pi
        if savePLTs:
            ax1.plot(timesfit, a, c='C1', lw=4, label = 'Data in fit')
            ax1.plot(timesfit,SineDecayFunc(t, *popt),
                    c='k', lw=2, label = 'Fit')
            ax1.plot(tsubset[startnoise:endnoise],
                    NoiseLevel*np.ones(len(tsubset[startnoise:endnoise])),
                    c='dimgray', lw=4, label = 'Noise Level')

        toc = time.time()
        print('Fitting took '+str(toc-tic)+' seconds')
        if burst == 0 and savePLTs:
            ax1.legend(loc='lower right')
            ax1.set_title('Fits using exponentially decaying sine')

        loop_data = np.vstack((loop_data, np.asarray((frequency, Q, tau))))

    if savePLTs:
        fig1.show()

    Fav = np.nanmean(loop_data[:,0])
    Fstd = np.nanstd(loop_data[:,0])
    Qav = np.nanmean(loop_data[:, 1])
    Qstd = np.nanstd(loop_data[:,1])
    tauav = np.nanmean(loop_data[:,2])
    taustd = np.nanstd(loop_data[:,2])

    print(20*'*')
    print(20*'*')
    print('Average ringdown Q: '+'%f' % (Qav)+' +/- '+'%f' % (Qstd))
    print('Average ringdown frequency: '+'%.3f' % (Fav)
            +' +/- '+'%.3f' % (Fstd)+' Hz')
    print('Average ringdown time constant: '+'%.3f' % (1000*tauav)
            +' +/- '+'%.3f' % (1000*taustd)+' ms')
    print(20*'*')
    print(20*'*')

    return Qav, Qstd, Fav, Fstd, tauav, taustd

def fitBurstSineDecay(tNeg, times, amp, fitbuffer, fitthreshhold,bounds, p0, savePLTs):

    ###########################################################################
    #  Takes in a timestream and amplitude data, as well as a falling edge and
    # returns the fit parameters for a single ringdown
    ###########################################################################

    # times: array of timestream data
    # amp: array of amplitude data for signal to be fit. Does not need to be
    #     scaled
    # fitbuffer: Number of samples to discard after a falling edge before
    #     fitting a decaying sine function
    # fitthreshhold: sets the cutoff for minimum amplitude at which to fit the
    #     decaying sine function

    # Takes the envelope of the signal to find the point at which the amplitude
    # falls below the fitthreshhold
    analytic_signal = signal.hilbert(amp)
    amplitude_envelope = np.abs(analytic_signal)

    # Sets start index to be maximum point in amplitude after the fitbuffer,
    # and sets end index to be the point where the amplitude falls below the
    # fit threshhold
    startind = int(min(np.argwhere(times > tNeg))+fitbuffer)
    end = int(min(np.argwhere((times > tNeg)&
                            (amplitude_envelope < fitthreshhold))))
    start = int(np.argmax(amp[startind:end]))+startind
    print('Fitting from '+'%.4f' % (times[start])+' to '+'%.4f' % (times[end]))
    a = amp[start:end]
    t = times[start:end]-times[start-1]

    # Fits data with an exponentially decaying sine function
    popt, pcov = curve_fit(SineDecayFunc, t, a, p0 = p0, bounds=bounds)
    tau = popt[1]
    frequency = popt[2]
    Q = tau * frequency * np.pi
    print('Frequency : '+str(frequency))
    print('Q : '+str(Q))
    print('Starting amplitude: '+str(popt[0]))

    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ax1.plot(times, amp, label='Data')
        ax1.plot(times[start:end],a, c='C1', lw=4, label = 'Data in fit')
        ax1.plot(times[start:end],fitthreshhold*np.ones(end-start),
                c='red', lw=2, label='Fit threshhold')
        ax1.plot(times[start:end],amplitude_envelope[start:end],
                c = 'dimgray', label = 'Envelope of fitted data')
        ax1.plot(times[start:end],SineDecayFunc(t, *popt),
                c='k', lw=2, label = 'Fit')
        ax1.legend(loc='lower right')
        fig1.show()

    return popt, a, t, times[start:end]

def findNoise(tNeg, tPos, times, amp, startbuffer, endbuffer):
    ###########################################################################
    #  Takes in a timestream and amplitude data, as well as a start and stopping
    # point. Uses buffers to throw out data on either side before finding the
    # RMS value of the noise and returning it.
    ###########################################################################

    # times: array of timestream data
    # amp: array of amplitude data for signal to be fit. Does not need to be
    #     scaled
    # startbuffer: Value between 0 and 1. Sets fraction of samples (in a period
    #     between falling edge andnext rising edge) to discard before fitting
    #     data to find noise level
    # endbuffer: Value between 0 and 1, with startbuffer+endbuffer < 1. sets
    #     fraction of samples to discard at the end of a noise sample before
    #     fitting

    start = int(min(np.argwhere(times > tNeg)))
    end = int(min(np.argwhere(times > tPos)))
    numSamples = end-start
    startnoise = int(start + startbuffer*numSamples)
    endnoise = int(end - endbuffer*numSamples)
    noisedata = amp[startnoise:endnoise]
    NoiseLevel = np.sqrt(2*np.mean(np.square(noisedata)))

    return NoiseLevel, startnoise, endnoise

def findEdges(times, trigger, freq):
    ###########################################################################
    #  Takes in a timestream and trigger amplitude data, performs a lockin at
    # the selected frequency and returns the times of the rising and falling
    # edges that are detected.
    ###########################################################################

    # times: array of timestream data
    # trigger: the array of amplitude data used for edge detection
    # freq: frequency at which to perform software lockin

    decimation = 10
    XX = np.sin(2*np.pi*freq*times)
    YY = np.sin(2*np.pi*freq*times+np.pi/2)
    # unsmoothed multiplied signals
    xx = np.multiply(trigger, XX)
    yy = np.multiply(trigger, YY)
    # smoothed signals
    x = signal.decimate(xx, decimation, axis=0, ftype='fir')
    y = signal.decimate(yy, decimation, axis=0, ftype='fir')
    # polar coordinates
    r = 2*np.sqrt(x**2 + y**2) # check that this is where we want this
    t = times[::decimation]

    posEdges, negEdges = ft.softwareTrigger(t, r, mph=0.5)

    return t[posEdges], t[negEdges]

def SineDecayFunc(x, A, tau, f, phi):
    ###########################################################################
    #  Exponentially decaying sine function
    ###########################################################################

    # A: initial amplitude
    # tau: exponential decay constant
    # f: frequency, in Hz
    # phi: phase offset, in radians [0,2pi]

    return A*np.sin(2*np.pi*f*x+phi)*np.exp(-x/tau)

# ~~~~~~~~~~~~~ Fit with FFTs of timestream data method ~~~~~~~~~~~
def FFTRingdownDirectory(directory, bias_points, samplerate, f_approx,
                        binfactor = 10, overlap = 0.9, mph = 0.5, buffer = 0.002,
                        amplitude_SNR_thresh = 0.5, savePLTs=True,
                        savetimestreamPLTs=False, Qplot=True, phib = None):
    ###########################################################################
    #  Read all the files in the directory specified, and for each bias point
    # timestream extract ringdown parameters by taking FFTs of the data and
    # fitting the frequency and decay of the maximum value
    ###########################################################################

    # directory: the relative path from the current process_directory
    # bias_points: a list of bias points (style: 'b00') to analyze. Eventually,
    #     this should be replaced with a path to the bias points json file, and
    #     extract both bias point names and properties from there
    # samplerate: sample rate, in Hz, of the raw data from the digitizer
    # f_approx: approximate frequency, in Hz, of the resonance
    # binfactor: Number of periods to include in each FFT
    # overlap: fractional overlap between each FFT (value takes 0-1)
    # mph: minimum normalized peak height (value takes 0-1) for identifying
    #     edges
    # buffer: sets the number of samples to discard in the vicinity of
    #       trigger edges/phase jumps/SNR thresholds, in fraction of the burst
    #       period (e.g. buffer=0.01 discards 1% of the burst period after
    #       each trigger and before each phase jump/SNR threshold).
    # amplitude_SNR_thresh: the required SNR for data to be considered in the
    #       amplitde fits, in e-foldings (i.e. log(signal/noise), with signal
    #       and noise both in Volts).
    # savePLTs: Whether to show plots of ringdown fits for each timestream
    # savetimestreamPLTs: whether to show plots of timestream data without fits
    # Qplot: Whether to show plots of Q, F, and Tau varying with bias points
    #     in the directory
    # Phib: list of phibs corresponding to each bias point, to be used to for
    #     plotting

    Qlist = []
    Qstdlist = []
    Taulist = []
    Taustdlist = []
    Flist = []
    Fstdlist = []

    N = len(bias_points)
    for ind in np.arange(N):
        bp = bias_points[ind]
        print(20*'~')
        print('Analyzing Bias Point '+bp)
        print(20*'~')
        filepath = directory + '/step_0-initial-'+bp+'_FLL-ringdown-FLL/raw-data'
        filename = filepath+'/'+os.listdir(filepath)[0]
        timestream = getUnscaledTimestreamData(filename, samplerate,
                        savetimestreamPLTs)

        times = np.array(timestream[:,0])
        amp = np.array(timestream[:,1])

        # Fit timestream using FFT method
        Qav, Qstd, Fav, Fstd, tauav, taustd = fitRingdownFFT(amp, times,
                                                samplerate, f_approx, binfactor,
                                                overlap, mph, buffer,
                                                amplitude_SNR_thresh, savePLTs)

        Qlist.append(Qav)
        Qstdlist.append(Qstd)
        Taulist.append(tauav)
        Taustdlist.append(taustd)
        Flist.append(Fav)
        Fstdlist.append(Fstd)


    for ind in np.arange(N):
        print('Q for '+bias_points[ind]+': '+str(Qlist[ind]))

    if Qplot:
        if phib is not None:
            xdata = phib
            xlabel = 'SQ1 Phib'
        else:
            xdata = np.arange(N)
            xlabel = 'Bias point'
        fig1, ax1 = mpl.subplots()
        ax1.errorbar(xdata, Qlist, yerr=Qstdlist, fmt='o',capsize=10)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Q (error bars = 1std)')
        ax1.set_title('Quality factor across SQ1 VPhi')
        fig1.show()

        fig2, ax2 = mpl.subplots()
        ax2.errorbar(xdata, Flist, yerr=Fstdlist, fmt='o',capsize=10)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Frequency (error bars = 1std)')
        ax2.set_title('Frequency across SQ1 VPhi')
        fig2.show()

        fig3, ax3 = mpl.subplots()
        ax3.errorbar(xdata, Taulist, yerr=Taustdlist, fmt='o',capsize=10)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Tau (error bars = 1std)')
        ax3.set_title('Tau across SQ1 VPhi')
        fig3.show()

    return Qlist, Qstdlist, Taulist, Taustdlist, Flist, Fstdlist

def fitRingdownFFT(amp, times, samplerate, f_approx, binfactor=10,
                    overlap=0.9, mph = 0.5, buffer = 0.002,
                    amplitude_SNR_thresh = 0.5, savePLTs=True):

    hp_filt = signal.butter(2, f_approx/10, 'highpass',
                            fs = samplerate, output = 'sos')
    amp_hp = signal.sosfilt(hp_filt, amp)

    print(20*'~')
    t, freqs, maxvals, binlength = findFFTmax(amp_hp, times, binfactor, overlap,
                                        samplerate, f_approx)
    print('Done taking FFTs!')

    print(20*'~')
    print('Finding bursts.....')
    posEdges, negEdges = ft.softwareTrigger(t, maxvals, mph=mph)

    numBursts = min(np.size(posEdges), np.size(negEdges))
    burstPeriod = np.median(np.diff(times[posEdges])) # seconds
    acqLength = times[-1]  # acquisition length, in seconds

    samplesPerPeriod = np.mean(np.diff(posEdges))
    ptmBuffer = 0.4*samplesPerPeriod

    print('Detected rising edges at: '+str(times[posEdges]))
    print('Detected falling edges at: '+str(times[negEdges]))
    print('Detected '+str(numBursts)+' full bursts')
    print('Burst period : '+'%.3f' % (burstPeriod)+' seconds')
    print('Total acquisition length: '+'%.2f' % (acqLength)+' seconds')


    ind = np.arange(len(t))

    loop_data = np.zeros((0, 3))

    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ax1.plot(t, np.log(maxvals), label='All Data')

    for burst in range(numBursts):
        print(20*'~')
        print('Fitting burst '+str(burst))

        # wait after a falling edge to start fitting
        first = negEdges[burst]

        try:
            # the next burst rising edge is available, so use it
            # if it's not, just use the previous burst's value
            # average over a window of width bufferSamples.
            ptmFirst = int(posEdges[burst+1] - ptmBuffer)
            ptmLast = int(posEdges[burst+1] - 0.1*ptmBuffer)
        except:
            print('Using previous pre-trigger mean')
        ptm = np.mean(maxvals[ptmFirst:ptmLast])
        ptmLog = np.log(ptm)
        print('Pre-trigger mean = ' + str(ptmLog))
        SNR = np.log(maxvals) - ptmLog
        notEnoughSNR = np.where(SNR < amplitude_SNR_thresh, True, False)
        notEnoughSNRindices = ind[notEnoughSNR]
        goodRSamples = (np.min(notEnoughSNRindices[notEnoughSNRindices > first])) - first
        lastr = first + goodRSamples
        fit_samples = lastr - first
        pltFirst = int(first - 0.1 * fit_samples)
        pltLastr = int(lastr + 0.1 * fit_samples)

        rcoeffs = np.polyfit(t[first:lastr], np.log(maxvals[first:lastr]), 1)
        tau = -1/rcoeffs[0]
        frequency = np.mean(freqs[first:lastr])
        Q = tau * frequency * np.pi

        if savePLTs:
            ax1.plot(t[first:lastr], np.log(maxvals[first:lastr]),
                        label='Data in Fits', c='C1', lw=4)
            ax1.plot(t[pltFirst:pltLastr],
                        t[pltFirst:pltLastr]*rcoeffs[0] + rcoeffs[1],
                        label='Fits', c='k', lw=2)
            ax1.plot(t[ptmFirst:ptmLast], ptmLog*np.ones(ptmLast-ptmFirst),
                        label='Pre-Trigger Mean', c='dimgray', lw=4)
        if burst == 0 and savePLTs:
            ax1.legend(loc='lower right')
            ax1.set_title('Amplitude fits using FFT method')

        print('Frequency : '+str(frequency))
        print('Q : '+str(Q))

        loop_data = np.vstack((loop_data, np.asarray((frequency, Q, tau))))

    Fav = np.nanmean(loop_data[:,0])
    Fstd = np.nanstd(loop_data[:,0])
    Qav = np.nanmean(loop_data[:, 1])
    Qstd = np.nanstd(loop_data[:,1])
    tauav = np.nanmean(loop_data[:,2])
    taustd = np.nanstd(loop_data[:,2])

    print(20*'*')
    print(20*'*')
    print('Average ringdown Q: '+'%f' % (Qav)+' +/- '+'%f' % (Qstd))
    print('Average ringdown frequency: '
            +'%.3f' % (Fav)+' +/- '+'%.3f' % (Fstd)+' Hz')
    print('Average ringdown time constant: '
            +'%.3f' % (1000*tauav)+' +/- '+'%.3f' % (1000*taustd)+' ms')
    print(20*'*')
    print(20*'*')

    if savePLTs:
        fig1.show()
    return Qav, Qstd, Fav, Fstd, tauav, taustd

def findFFTmax(amp, times, binfactor, overlap, samplerate, f_approx):
    Ncycle = int(samplerate/f_approx) # number of samples expected per period
    binfactor = int(binfactor)
    binsize = 2**(binfactor*Ncycle-1).bit_length()
    binlength = binsize/samplerate
    maxbins = math.floor((len(amp)/binsize-1)/(1-overlap))+1
    print('Taking '+str(maxbins)+' FFTs of '
            +'%.3f' % (1000*binlength)+ ' ms each')

    f = samplerate/2*np.linspace(0,1,int(binsize/2)+1)

    maxvals = np.zeros(maxbins)
    t = np.zeros(maxbins)
    freqs = np.zeros(maxbins)

    twin = signal.windows.flattop(binsize)
    S1 = sum(twin)

    for i in np.arange(maxbins):
        start = math.floor(i*(1-overlap)*binsize)
        end = start+binsize
        Y = np.fft.fft(amp[start:end]*twin, binsize)
        Yh = 2*abs(Y[0:int(binsize/2)+1])/S1
        maxind = np.argmax(Yh)
        maxvals[i] = Yh[maxind]
        freqs[i] = f[maxind]
        t[i] = np.mean(times[start:end])

    return t, freqs, maxvals, binlength

# ~~~~~~~~~~~~~ Fit with software lockin method ~~~~~~~~~~~
def LIRingdownDirectory(directory, bias_points, samplerate=10000000, f_approx=675500,
                            decimation=10, useCarEdges=True, mph = 0.5, buffer = 0.002,
                            amplitude_SNR_thresh = 0.5, max_expected_detuning=15000,
                            savePLTs = True, Qplot = True, phib=None):
    Qlist = []
    Qstdlist = []
    Taulist = []
    Taustdlist = []
    Flist = []
    Fstdlist = []

    N = len(bias_points)
    for ind in np.arange(N):
        print(20*'~')
        print('Analyzing Bias Point '+bias_points[ind])
        print(20*'~')
        filepath = directory + '/step_0-initial-'+bias_points[ind]+'_FLL-ringdown-FLL/raw-data'
        filename = filepath+'/'+os.listdir(filepath)[0]
        timestream = getUnscaledTimestreamData(filename, samplerate, savePLTs = False)
        d = getLockinDataFromTimestream(timestream, f_approx, decimation, savePLTs = False)
        t = d[:,0]
        r = d[:,1]
        phi = d[:,2]
        carr = d[:,3]
        carphi = d[:,4]
        Qav, Qstd, sumAv, sumStd, tauAv, tauStd = fitRingdownLI(t, r, phi, carr, carphi,
                                                f_approx, useCarEdges, mph, buffer,
                                                amplitude_SNR_thresh, max_expected_detuning,
                                                savePLTs)
        Qlist.append(Qav)
        Qstdlist.append(Qstd)
        Taulist.append(tauAv)
        Taustdlist.append(tauStd)
        Flist.append(sumAv)
        Fstdlist.append(sumStd)


    for ind in np.arange(N):
        print('Q for '+bias_points[ind]+': '+str(Qlist[ind]))

    if Qplot:
        if phib is not None:
            xdata = phib
            xlabel = 'SQ1 Phib'
        else:
            xdata = np.arange(N)
            xlabel = 'Bias point'
        fig1, ax1 = mpl.subplots()
        ax1.errorbar(xdata, Qlist, yerr=Qstdlist, fmt='o',capsize=10)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Q (error bars = 1std)')
        ax1.set_title('Quality factor across SQ1 VPhi')
        fig1.show()

        fig2, ax2 = mpl.subplots()
        ax2.errorbar(xdata, Flist, yerr=Fstdlist, fmt='o',capsize=10)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Frequency (error bars = 1std)')
        ax2.set_title('Frequency across SQ1 VPhi')
        fig2.show()

        fig3, ax3 = mpl.subplots()
        ax3.errorbar(xdata, Taulist, yerr=Taustdlist, fmt='o',capsize=10)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Tau (error bars = 1std)')
        ax3.set_title('Tau across SQ1 VPhi')
        fig3.show()
    return Qlist, Qstdlist, Taulist, Taustdlist, Flist, Fstdlist

def fitRingdownLI(t, r, phi, carr, carphi, f_approx,
                useCarEdges=True, mph = 0.4, buffer = 0.002,
                amplitude_SNR_thresh = 0.5, max_expected_detuning=15000,
                savePLTs=True):

    if useCarEdges:
        posEdges, negEdges = ft.softwareTrigger(t, carr, mph=mph)
    else:
        posEdges, negEdges = ft.softwareTrigger(t, r, mph=mph)

    posEdge, negEdge = ft.firstFullBurst(posEdges, negEdges)
    numBursts = min(np.size(posEdges), np.size(negEdges))
    samplesPerPeriod = np.mean(np.diff(posEdges))  # samples
    burstPeriod = np.median(np.diff(t[posEdges]))  # seconds
    bufferSamples = int(buffer * samplesPerPeriod)  # samples
    dutyCycle = (t[negEdge] - t[posEdge])/burstPeriod
    acqLength = t[-1]  # acquisition length, in seconds
    N = len(t)
    inds = np.arange((N))
    decimated_sample_rate = 1/t[1]
    print('Detected rising edges at: '+str(t[posEdges]))
    print('Detected falling edges at: '+str(t[negEdges]))
    print('Discarding data within '+str(bufferSamples)+' samples of trigger edges/phase jumps')
    print('Detected '+str(numBursts)+' full bursts')
    print('Burst period : '+'%.3f' % (burstPeriod)+' seconds')
    print('Burst duty cycle: '+'%.2f' % (dutyCycle))
    print('Total acquisition length: '+'%.2f' % (acqLength)+' seconds')
    print('Downsampled data with sampling rate: '+'%d' % (decimated_sample_rate)+' Hz')
    df = np.diff(phi)/360 * decimated_sample_rate  # Hz
    jumps = np.where(np.abs(df) > max_expected_detuning, True, False)
    jumpIndices = inds[0:-1][jumps]
    print('Jump indices: ' + str(jumpIndices))

    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ax1.plot(t, np.log(r), label='All Data')
        fig2, ax2 = mpl.subplots()
        ax2.plot(t, phi, label='All Data')
        fig3, ax3 = mpl.subplots()
        ax3.plot(t[1:], df, label='All Data')
        ax3.scatter(t[1:][jumps], df[jumps], label='Detected jumps', c='r')
        fig1.show()
        fig2.show()
        fig3.show()

    loop_data = np.zeros((0, 4))
    # within each acquisition, loop through the detected bursts and
    # fit to extract the Q, f, tau data.
    for burst in range(numBursts):
        print(20*'~')
        print('Fitting burst '+str(burst))

        # wait after a falling edge to start fitting either the
        # or amplitude data.
        first = negEdges[burst] + bufferSamples

        # compute the average amplitude right before the rising
        # edge to estimeate the lockin amplitude noise floor:
        # we call this the 'pre-trigger mean' = ptm
        try:
            # the next burst rising edge is available, so use it
            # if it's not, just use the previous burst's value
            # average over a window of width bufferSamples.
            ptmFirst = posEdges[burst+1] - 100*bufferSamples
            ptmLast = posEdges[burst+1] - 10*bufferSamples
        except:
            print('Using previous pre-trigger mean')
        ptm = np.mean(r[ptmFirst:ptmLast])
        ptmLog = np.log(ptm)
        print('Pre-trigger mean = ' + str(ptmLog))

        # Evaluate which amplitude data have enough SNR to be used
        # for amplitude fitting.
        SNR = np.log(r) - ptmLog
        notEnoughSNR = np.where(SNR < amplitude_SNR_thresh, True, False)
        notEnoughSNRindices = inds[notEnoughSNR]
        goodRSamples = (np.min(notEnoughSNRindices[notEnoughSNRindices > first])) - first
        enoughRSamples = goodRSamples > bufferSamples
        lastr = first + goodRSamples - bufferSamples

        # In order to make the amplitude plots somewhat easier to
        # read, extend the time range over which to plot the fits.
        fit_samples = lastr - first
        pltFirst = int(first - 0.1 * fit_samples)
        pltLastr = int(lastr + 0.1 * fit_samples)

        # Evaluate which phase data are far enough from a phase
        # jump (where phase wraps through +/- 180 degrees in our
        # definition of phase).
        goodPhiSamples = (np.min(jumpIndices[jumpIndices > first])) - first
        # ensure there will be at least 10 points in the fit
        enoughPhiSamples = goodPhiSamples > bufferSamples
        lastphi = first + goodPhiSamples - bufferSamples
        pltLastphi = int(lastphi + 0.1 * fit_samples)

        # fit to the log of the amplitude data to get the
        # exponential time decay constant (a.k.a tau)
        print('Fitting amplitude using ' + str(goodRSamples)+ ' points')
        if enoughRSamples:
            print('Using amplitude data between '+'%.3f' % (t[first])+' and ''%.3f' % (t[lastr])+' seconds')
            rcoeffs = np.polyfit(t[first:lastr], np.log(r[first:lastr]), 1)
        else:
            rcoeffs = np.asarray((np.nan, np.nan))
            print('Not enough samples to fit r')
        # amplitude decay time in seconds/e-folding
        tau = -1/rcoeffs[0]

        # fit to the phase data to get the rate of change of phase
        # of the SQUID signal with respect to the lockin reference
        # signal, which gives the detuning in Hz between the SQUID
        # signal and the reference.
        print('Fitting phase using ' + str(goodPhiSamples)+ ' points')
        if enoughPhiSamples:
            print('Using phase data between '+'%.3f' % (t[first])+' and ''%.3f' % (t[lastphi])+' seconds')
            phicoeffs = np.polyfit(t[first:lastphi], phi[first:lastphi], 1)
        else:
            phicoeffs = np.asarray((np.nan, np.nan))
            print('Not enough samples to fit phase')
        # measured frequency detuning
        detuning = phicoeffs[0]/360
        # measured resonant frequency
        sum = f_approx + detuning

        # combine tau and sum to get ringdown Q
        Q = tau * sum * np.pi

        # print fit outputs for each burst
        print('burst '+str(burst)+' ringdown Q: '+'%f' % (Q))
        print('burst '+str(burst)+' ringdown frequency: '+'%.3f' % (sum)+' Hz')
        # add data to the array
        loop_data = np.vstack((loop_data, np.asarray((sum, Q, tau, detuning))))
        # plot data used in the fits, and the fits themselves
        if enoughRSamples and savePLTs:
            ax1.plot(t[first:lastr], np.log(r[first:lastr]), label='Data in Fits', c='C1', lw=4)
            ax1.plot(t[pltFirst:pltLastr], t[pltFirst:pltLastr]*rcoeffs[0] + rcoeffs[1], label='Fits', c='k', lw=2)
            ax1.plot(t[ptmFirst:ptmLast], ptmLog*np.ones(ptmLast-ptmFirst), label='Pre-Trigger Mean', c='dimgray', lw=4)
        if enoughPhiSamples and savePLTs:
            ax2.plot(t[first:lastphi], phi[first:lastphi], label='Data in Fit', c='C1', lw=4)
            ax2.plot(t[pltFirst:pltLastphi], t[pltFirst:pltLastphi]*phicoeffs[0] + phicoeffs[1], label='Fit', c='k', lw=2)
        if burst == 0 and savePLTs:
            ax1.legend(loc='lower right')
            ax1.set_title('Amplitude fits using lockin method')
            ax2.legend(loc='lower right')
            ax2.set_title('Phase fits using lockin method')



    # all the detected bursts have been fit, calculate summary
    # statistics for this acquisition.
    sumAv = np.nanmean(loop_data[:, 0])
    sumStd = np.nanstd(loop_data[:, 0])
    Qav = np.nanmean(loop_data[:, 1])
    Qstd = np.nanstd(loop_data[:, 1])
    tauAv = np.nanmean(loop_data[:, 2])
    tauStd = np.nanstd(loop_data[:, 2])
    detuningAv = np.nanmean(loop_data[:, 3])
    detuningStd = np.nanstd(loop_data[:, 3])
    # add the data from this acquistion to the data for the whole
    # directory and print to the terminal.
    print(20*'*')
    print(20*'*')
    print('Average ringdown Q: '+'%f' % (Qav)+' +/- '+'%f' % (Qstd))
    print('Average ringdown frequency: '+'%.3f' % (sumAv)+' +/- '+'%.3f' % (sumStd)+' Hz')
    print('Average ringdown time constant: '+'%.3f' % (1000*tauAv)+' +/- '+'%.3f' % (1000*tauStd)+' ms')
    print(20*'*')
    print(20*'*')

    if savePLTs:
        ax1.grid(True)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Log(SQUID amplitude/1V) (dimensionless)')
        ax1.set_title('Average ringdown Q: '+'%.2f' % (Qav)+' +/- '+'%.f' % (Qstd))
        fig1.tight_layout()

        ax2.grid(True)
        ax2.set_ylim(bottom=-500, top=500)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('SQUID Phase (degrees)')
        ax2.set_title('Average ringdown frequency: '+'%.3f' % (sumAv)+' +/- '+'%.3f' % (sumStd)+' Hz'
                      +'\nAverage detuning from reference: '+'%.3f' % (detuningAv)+' +/- '+'%.3f' % (detuningStd)+' Hz')
        fig2.tight_layout()

        ax3.grid(True)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Detuning from reference (Hz)')
        ax3.set_title('Detuning threshold: '+'%d' % (max_expected_detuning)+' Hz')
        ax3.legend()
        fig3.tight_layout()

    return Qav, Qstd, sumAv, sumStd, tauAv, tauStd

def getLockinDataFromTimestream(timestreamData, frequency, dec, savePLTs=True):
    times = timestreamData[:,0]
    signals = timestreamData[:,1:]
    lockindata = lockin.streamlinedLockIn(times, signals, frequency, decimation=dec)
    lockintimes = lockindata[:,0]
    lockinSQr = lockindata[:,3]
    lockinSQphi = lockindata[:,4]
    lockinCarR = lockindata[:,11]
    lockinCarPhi = lockindata[:,12]
    if savePLTs:
        fig, axs = mpl.subplots(2)
        #ig.suptitle('Signal injection frequency: {:.2}'.format(frequency))
        axs[0].plot(lockintimes, np.log(lockinSQr))
        axs[0].set_xlabel('Time (Seconds)')
        axs[0].set_ylabel('log(SQ radius)')
        axs[1].plot(lockintimes, lockinSQphi)
        axs[1].set_xlabel('Time (Seconds)')
        axs[1].set_ylabel('SQ phase')
        fig.tight_layout()
        fig.show()
    return np.column_stack((lockintimes, lockinSQr, lockinSQphi, lockinCarR, lockinCarPhi))

# ~~~~~~~~~~~~~ Functions to load data ~~~~~~~~~~~
def getUnscaledTimestreamData(file, samplerate=10e6, savePLTs=True):
    dig_data = h5py.File(file,'r')
    chan1 = np.array(dig_data['channel_1_data/data_unscaled'][0])
    chan2 = np.array(dig_data['channel_2_data/data_unscaled'][0])
    times = np.linspace(0, len(chan1)/samplerate, num=len(chan1))
    if savePLTs:
        mpl.plot(times, chan1)
        mpl.plot(times, chan2)
        mpl.xlabel('time')
        mpl.legend(["Chan1","Chan2"], loc = 'lower right')
        mpl.show()
    return np.column_stack((times, chan1, chan2))

def getScopeTraceData(file, channels, savePLTs=True):
    print('Loading file...............')
    data = np.loadtxt(file, delimiter=',')
    print('File loaded!')
    print(20*'~')
    times = data[:,0]
    signals = data[:,channels]
    if savePLTs:
        for ind in range(len(channels)):
            mpl.plot(times, signals[:,ind],label='Channel '+str(channels[ind]))
        mpl.xlabel('time')
        mpl.legend(loc = 'lower right')
        mpl.show()
    return np.column_stack((times, signals))
