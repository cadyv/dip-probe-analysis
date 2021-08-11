import numpy as np
import matplotlib.pyplot as mpl
import math
from scipy.optimize import curve_fit as curve_fit
from scipy.signal import periodogram as PSD
from detect_peaks import detect_peaks as findpks
from scipy.signal import decimate


def returnSine(times, amplitude, frequency, phase):
    # amplitude in your favorite units, frequency in Hz, phase in degrees
    return amplitude*np.sin(2*np.pi*frequency*times+phase*np.pi/180)


def returnSineOffset(times, amplitude, frequency, phase, offset):
    # amplitude in your favorite units, frequency in Hz, phase in degrees
    return offset + amplitude*np.sin(2*np.pi*frequency*times+phase*np.pi/180)


def fitSine(times, voltages, **kwargs):
    fitDC = kwargs.pop('fitDC', 0)
    if not fitDC:
        defaultBounds = [[0, 0, -180], [np.inf, np.inf, 180]]
    elif fitDC:
        defaultBounds = [[0, 0, -180, -np.inf], [np.inf, np.inf, 180, np.inf]]
    bounds = kwargs.pop('bounds', defaultBounds)
    plot = kwargs.pop('plot', 0)
    overPlot = kwargs.pop('overPlot', 0)
    prnt = kwargs.pop('prnt', 0)
    # guess the sine amplitude, guess sine freq from zero crossings
    ampGuess = (np.max(voltages)-np.min(voltages))/2
    freqGuess = kwargs.pop('nominalFreq', None)
    if freqGuess is None:
        zeroCrossings = voltages[0:-2]*voltages[1:-1]
        zeroCrossings[zeroCrossings > 0] = 0
        crossIndices = np.argwhere(zeroCrossings)
        crossTimes = times[crossIndices].flatten()
        periodGuess = 2*np.mean(np.diff(crossTimes))
        freqGuess = 1/periodGuess
    if not fitDC:
        guess = [ampGuess, freqGuess, 0]
        params, covariance = curve_fit(returnSine, times, voltages, p0=guess,
                                       bounds=bounds)
    elif fitDC:
        offsetGuess = np.mean(voltages)
        guess = [ampGuess, freqGuess, 0, offsetGuess]
        params, covariance = curve_fit(returnSineOffset, times, voltages,
                                       p0=guess, bounds=bounds)
    if plot:
        if not overPlot:
            mpl.clf()
        mpl.plot(times, voltages, label='data')
        if not fitDC:
            mpl.plot(times, returnSine(times, params[0], params[1],
                                       params[2]),
                                       label='fit')
        elif fitDC:
            mpl.plot(times, returnSineOffset(times, params[0], params[1],
                                             params[2], params[3]),
                                             label='fit')
        mpl.xlabel("Time (seconds)")
        mpl.ylabel("Signal (Volts)")
        mpl.legend()
        mpl.show()
    if prnt:
        print("Guessed amplitude: "+str(ampGuess))
        print("Guessed frequency: "+str(freqGuess))
        print("Fitted amplitude, frequency, phase, (DC Level):")
        print(params)
    return params, covariance


def returnSpectrumPeaks(times, voltages, **kwargs):
    # units are Volts/rt(Hz) before normalization, and 1/rt(Hz) after
    # variables with "Mag" in the name are the absolute value of a Fourier bin
    # variables with "Amp" in the name are complex Fourier amplitudes
    plot = kwargs.pop('plot', 0)
    overPlot = kwargs.pop('overPlot', 0)
    prnt = kwargs.pop('prnt', 0)
    numPks = kwargs.pop('numPks', 25)
    minPkDist = kwargs.pop('minPkDist', 1)  # 1 Hz
    # look for peaks with amplitudes above 10% of largest peak's amplitude
    minPkHt = kwargs.pop('minPkHt', 0.1)
    sampleFreq = kwargs.pop('sampleFreq', None)
    if sampleFreq is None:
        sampleFreq = 1/np.mean(np.diff(times))
    numPts = times.shape[0]
    # discard negative frequency components, which are redundant for the DFT
    # of a purely real signal. For numpy, the largest positive frequency is at
    # index floor(numPts/2)
    lastIndex = math.floor(numPts/2)
    freqs = sampleFreq*np.fft.fftfreq(numPts)
    fftAmps = np.fft.ifft(voltages)
    fftMags = 2*np.abs(fftAmps)
    fftArgs = np.angle(fftAmps, deg=True)
    normMags = fftMags/np.max(fftMags)
    pkIndices = findpks(normMags[0:lastIndex], mpd=minPkDist, mph=minPkHt)
    pkFreqs = freqs[pkIndices]
    pkMags = fftMags[pkIndices]
    pkArgs = fftArgs[pkIndices]
    pkParams = np.stack((pkMags, pkFreqs, pkArgs), axis=1)
    # sort peaks in descending order by amplitude
    pkParams = np.flip(pkParams[pkParams[:, 0].argsort()], axis=0)
    # pad the output with zeros to always be numPks long
    pkOutputs = np.zeros((numPks, 3))
    pkOutputs = pkParams[0:numPks, :]
    if plot:
        if not overPlot:
            mpl.clf()
        mpl.plot(freqs[0:lastIndex], fftMags[0:lastIndex])
        mpl.plot(pkFreqs, pkMags, 'r.')
        mpl.xlabel("Frequency (Hz)")
        mpl.ylabel("Estimated signal amplitude (Volts)")
    if prnt:
        print("Found peaks at: (Hz)")
        print(pkFreqs)
        print("with magnitudes: (Volts))")
        print(pkMags)
        print("with phases: (degrees)")
        print(pkArgs)
    return pkOutputs


def lockIn(times, carrier, signal, **kwargs):
    # provide a reasonable frequency guess to initialize the fit
    nominalFreq = kwargs.pop('nominalFreq', None)
    # fit to the carrier signal in the first fraction of an acquisition
    # fit window = 1 --> fit all data
    # fit window = 0.1 --> fit to first 10% of data
    fitWindow = kwargs.pop('fitWindow', 1)
    # sample index of first sample to begin fitting at
    firstSample = kwargs.pop('firstSample', 0)
    # which harmonic of the carrier to lock in to
    harmonic = kwargs.pop('harmonic', 1)
    # decimation factor in the output timestreams
    decimation = kwargs.pop('decimation', None)
    N = np.size(times)
    if fitWindow is not None:
        lastSample = int(fitWindow * N + firstSample)
        params, covariance = fitSine(times[firstSample:lastSample],
                                     carrier[firstSample:lastSample],
                                     fitDC=True, nominalFreq=nominalFreq)
    elif fitWindow is None:
        params = np.asarray((np.nan, nominalFreq, 0, np.nan))
    # generate a quadrature pair at the carrier frequency
    # amplitude 1V, frequency is carrier frequency times harmonic index,
    # phase is fit phase, offset is 0V
    XX = returnSineOffset(times, 1, harmonic*params[1], params[2], 0)
    firstX, lastX = returnIntegerNumberOfPeriods(times, XX)
    YY = returnSineOffset(times, 1, harmonic*params[1], params[2]+90, 0)
    # unsmoothed multiplied signal
    xx = XX[firstX:lastX]*signal[firstX:lastX]
    yy = YY[firstX:lastX]*signal[firstX:lastX]
    # homodyne the actual carrier voltage against the fitted version of it
    carxx = XX[firstX:lastX]*carrier[firstX:lastX]
    caryy = YY[firstX:lastX]*carrier[firstX:lastX]
    # pick a number of periods of the carrier to average over
    periods = kwargs.pop('periods', 10)
    T = periods * 1/params[1]  # T is averaging time in seconds
    t = times[1] - times[0]  # t is sampling time in seconds
    numSamples = int(T/t)  # number of samples in T seconds
    window = np.ones(numSamples)  # square window, for now
    # smooth and evaluate parameters
    x = 2*t/T*np.convolve(xx, window, mode='valid')
    y = 2*t/T*np.convolve(yy, window, mode='valid')
    r = np.sqrt(x**2 + y**2)
    phi = 180/np.pi*np.arctan2(y, x)
    # for the carrier as well
    carx = 2*t/T*np.convolve(carxx, window, mode='valid')
    cary = 2*t/T*np.convolve(caryy, window, mode='valid')
    carr = np.sqrt(carx**2 + cary**2)
    carphi = 180/np.pi*np.arctan2(cary, carx)
    # return the vector of times after smoothing
    tt = times[0:np.shape(x)[0]]
    if type(decimation) == int:
        tt = tt[0:-1:decimation]
        x = decimate(x, decimation, ftype='fir')
        y = decimate(y, decimation, ftype='fir')
        r = decimate(r, decimation, ftype='fir')
        phi = decimate(phi, decimation, ftype='fir')
        carx = decimate(carx, decimation, ftype='fir')
        cary = decimate(cary, decimation, ftype='fir')
        carr = decimate(carr, decimation, ftype='fir')
        carphi = decimate(carphi, decimation, ftype='fir')
    return tt, x, y, r, phi, carx, cary, carr, carphi, params


def lockInNoCarrier(times, signal, **kwargs):
    # provide a reasonable frequency guess to initialize the fit
    nominalFreq = kwargs.pop('nominalFreq', None)
    # which harmonic of the carrier to lock in to
    harmonic = kwargs.pop('harmonic', 1)
    # generate a quadrature pair at the carrier frequency
    # amplitude 1V, frequency is carrier frequency times harmonic index,
    # phase is fit phase, offset is 0V
    XX = returnSineOffset(times, 1, harmonic*nominalFreq, 0, 0)
    firstX, lastX = returnIntegerNumberOfPeriods(times, XX)
    YY = returnSineOffset(times, 1, harmonic*nominalFreq, 90, 0)
    # unsmoothed multiplied signal
    xx = XX[firstX:lastX]*signal[firstX:lastX]
    yy = YY[firstX:lastX]*signal[firstX:lastX]
    # pick a number of periods of the carrier to average over
    periods = kwargs.pop('periods', 10)
    T = periods * 1/nominalFreq  # T is averaging time in seconds
    t = times[1] - times[0]  # t is sampling time in seconds
    numSamples = int(T/t)  # number of samples in T seconds
    window = np.ones(numSamples)  # square window, for now
    # smooth and evaluate parameters
    x = 2*t/T*np.convolve(xx, window, mode='valid')
    y = 2*t/T*np.convolve(yy, window, mode='valid')
    r = np.sqrt(x**2 + y**2)
    phi = 180/np.pi*np.arctan2(y, x)
    # return the vector of times after smoothing
    tt = times[0:np.shape(x)[0]]
    return tt, x, y, r, phi


def returnIntegerNumberOfPeriods(times, voltages):
    voltages = voltages - np.mean(voltages)
    zeroCrossings = voltages[0:-2]*voltages[1:-1]
    zeroCrossings[zeroCrossings > 0] = 0
    crossIndices = np.argwhere(zeroCrossings)
    firstCrossing = np.min(crossIndices)
    lastCrossing = np.max(crossIndices)
    return firstCrossing, lastCrossing


def softwareTrigger(times, trigger, **kwargs):
    mph = kwargs.pop('mph', 0.5)
    trigdf = np.diff(trigger)
    normalized = trigdf/np.max(np.abs(trigdf))
    # find all edges with a maximum instantaneous time derivative at least
    # mph times the fastest edge
    risingEdges = findpks(normalized, mph=mph)
    fallingEdges = findpks(-normalized, mph=mph)
    return risingEdges, fallingEdges


# assume the acquisition is longer than 2 burst periods,
# as this gaurantees there is a full, unitinterrupted burst
# somewhere in the data
def firstFullBurst(risingEdges, fallingEdges):
    # the first edge was falling,
    # so the acquisition started during a burst:
    # discard the first burst for fitting purposes
    # we want the first rising edge and the second falling edge
    if fallingEdges[0] < risingEdges[0]:
        risingEdge = risingEdges[0]
        fallingEdge = fallingEdges[1]
    # otherwise, the first burst is fully contained in the acquisition
    else:
        risingEdge = risingEdges[0]
        fallingEdge = fallingEdges[0]
    return risingEdge, fallingEdge


def parse_timestamp(fname, year):
    # assume format is:
    # yyyymmdd_hhmmss, local time
    # calculate seconds since first of the month, current year
    # leap seconds, lengths of months, etc not dealt with
    start = fname.find('_'+str(year)) + 1
    stamp = fname[start:start+15]
    if stamp[8] != '_':
        print('Using wrong date format! Returning NaN.')
        return np.NaN
    else:
        dd = int(stamp[6:8])
        hh = int(stamp[9:11])
        mm = int(stamp[11:13])
        ss = int(stamp[13:15])
        return 24 * 3600 * dd + 3600 * hh + 60 * mm + ss
