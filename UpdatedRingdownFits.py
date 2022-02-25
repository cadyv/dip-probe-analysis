import numpy as np
import matplotlib.pyplot as mpl
from detect_peaks import detect_peaks as findpks
from scipy import signal

def findBurstsTrigSig(times, trigger, **kwargs):
    ########################################
    #  Finds edges of bursts using trigger signal from function gen
    ########################################

    mph = kwargs.pop('mph', 0.5)
    savePLTs = kwargs.pop('savePLTs', True)
    printThings = kwargs.pop('printThings',True)

    trigdf = np.diff(trigger)
    normalized = trigdf/np.max(np.abs(trigdf))

    posEdges = findpks(normalized, mph=mph)
    negEdges = findpks(-normalized, mph=mph)
    posTimes = times[posEdges]
    negTimes = times[negEdges]
    if printThings:
        print("Found "+str(len(posEdges))+ " positive edges")
        print("Found "+str(len(negEdges))+ " negative edges")

    # convert from times back to indices of undecimated timestream
    risingEdges = np.searchsorted(times, posTimes)
    fallingEdges = np.searchsorted(times, negTimes)

    if fallingEdges[0] < risingEdges[0]:
        fallingEdges = fallingEdges[1:]
    risingEdges = risingEdges[1:]
    if len(risingEdges) == len(fallingEdges):
        starts = fallingEdges.tolist()
        ends = risingEdges.tolist()
    elif len(risingEdges) == len(fallingEdges)-1:
        starts = fallingEdges.tolist()
        ends = risingEdges.tolist()
        ends.append(-1)
    else:
        starts = fallingEdges.tolist()
        ends = fallingEdges[1:].tolist()
        ends.append(-1)
    numbursts = min(np.size(risingEdges), np.size(fallingEdges))
    if printThings:
        print(20*'~')
        print('Detected '+str(numbursts)+' bursts')
        print(20*'~')
    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ymax = max(trigger)*1.1
        ax1.plot(times, trigger, zorder = 1, c='mediumpurple', label = 'trigger data')
        ax1.plot(times[1:], normalized, zorder = 1, c='yellow', label = 'np diff of trigger')
        ax1.vlines(times[starts],-ymax, ymax, colors='green',zorder=2, label = 'burst starts')
        ax1.vlines(times[ends],-ymax, ymax, colors='red',zorder=2, label = 'burst ends')
        ax1.legend(loc='lower left')
        fig1.show()
    return starts, ends

def findBurstsFuncGen(times, trigger, f_approx, **kwargs):
    ########################################
    #  Finds edges of bursts by taking lockin of trigger
    ########################################
    # - f_approx sets the lockin frequency

    mph = kwargs.pop('mph', 0.5)
    savePLTs = kwargs.pop('savePLTs', True)
    printThings = kwargs.pop('printThings',True)

    decimation = kwargs.pop('decimation', 10)

    XX = np.sin(2*np.pi*f_approx*times)
    YY = np.sin(2*np.pi*f_approx*times+np.pi/2)
    # unsmoothed multiplied signals
    xx = np.multiply(trigger, XX)
    yy = np.multiply(trigger, YY)
    # smoothed signals
    x = signal.decimate(xx, decimation, axis=0, ftype='fir')
    y = signal.decimate(yy, decimation, axis=0, ftype='fir')
    # polar coordinates
    r = 2*np.sqrt(x**2 + y**2) # check that this is where we want this
    t = times[::decimation]
    trigdf = np.diff(r)
    normalized = trigdf/np.max(np.abs(trigdf))

    posEdges = findpks(normalized, mph=mph)
    negEdges = findpks(-normalized, mph=mph)
    posTimes = t[posEdges]
    negTimes = t[negEdges]

    if printThings:
        print("Found "+str(len(posEdges))+ " positive edges")
        print("Found "+str(len(negEdges))+ " negative edges")

    # convert from times back to indices of undecimated timestream
    risingEdges = np.searchsorted(times, posTimes)
    fallingEdges = np.searchsorted(times, negTimes)

    if fallingEdges[0] < risingEdges[0]:
        fallingEdges = fallingEdges[1:]
    risingEdges = risingEdges[1:]
    if len(risingEdges) == len(fallingEdges):
        starts = fallingEdges.tolist()
        ends = risingEdges.tolist()
    elif len(risingEdges) == len(fallingEdges)-1:
        starts = fallingEdges.tolist()
        ends = risingEdges.tolist()
        ends.append(-1)
    else:
        starts = fallingEdges.tolist()
        ends = fallingEdges[1:].tolist()
        ends.append(-1)
    numbursts = min(np.size(risingEdges), np.size(fallingEdges))
    if printThings:
        print(20*'~')
        print('Detected '+str(numbursts)+' bursts')
        print(20*'~')
    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ymax = max(trigger)*1.5
        ax1.plot(times, trigger, zorder = 1, c='mediumpurple', label = 'trigger data')
        ax1.plot(t, r, zorder = 1, c='indigo', label = 'lockin of trigger data')
        ax1.plot(t[1:], normalized, zorder = 1, c='yellow', label = 'np diff of lockin')
        ax1.vlines(times[starts],-ymax, ymax, colors='green',zorder=2, label = 'burst starts')
        ax1.vlines(times[ends],-ymax, ymax, colors='red',zorder=2, label = 'burst ends')
        ax1.legend(loc='lower left')
        fig1.show()
    return starts, ends

def fitBurstLockin(t_burst, amp_burst,f_approx,  **kwargs):
    ########################################
    #  Fits a single ringdown using lockin measurement
    ########################################

    max_expected_detuning = kwargs.pop('max_expected_detuning', 2000) # this changes how picky the phase fit is
    decimation = kwargs.pop('decimation', 100) # decrease this if you want to fit more points, increase if the data looks too noisy
    savePLTs = kwargs.pop('savePLTs', True)
    buffer = kwargs.pop('buffer',0.001) # throws out points at beginning/end of fit
    amplitude_SNR_thresh = kwargs.pop('amplitude_SNR_thresh', 0.5) # increase this if you want the fit to stop higher above the calculated noise floor
    threshhold = kwargs.pop('threshhold', -1) # the squid amplitude must decay to below this threshhold before fit begins
    minphisample = kwargs.pop('minphisample', 10) # minimum number of phi samples to fit
    minrsample = kwargs.pop('minrsample', 10) # minimum number of r samples to fit

    bufferSamples = int(buffer * len(amp_burst))


    XX = np.sin(2*np.pi*f_approx*t_burst)
    YY = np.sin(2*np.pi*f_approx*t_burst+np.pi/2)
    # unsmoothed multiplied signals
    xx = np.multiply(amp_burst, XX)
    yy = np.multiply(amp_burst, YY)
    # smoothed signals
    x = signal.decimate(xx, decimation, axis=0, ftype='fir')
    y = signal.decimate(yy, decimation, axis=0, ftype='fir')
    # polar coordinates
    r = 2*np.sqrt(x**2 + y**2) # check that this is where we want this
    phi = 180/np.pi*np.arctan2(y, x)
    t = t_burst[::decimation]
    inds = np.arange(len(t))
    decimated_sample_rate = 1/(t[0]-t[1])

    df = np.diff(phi)/360 * decimated_sample_rate
    jumps = np.where(np.abs(df) > max_expected_detuning)
    jumpIndices = inds[0:-1][jumps]

    ptm = np.mean(r[-100*bufferSamples:-10*bufferSamples])
    ptmLog=np.log(ptm)

    #### set noise floor
    NoiseFloor = kwargs.pop('NoiseFloor',ptmLog)

    print('Pre-trigger mean = ' + str(ptmLog))

    SQmax = np.argmax(np.log(r))
    SQLowEnough = np.where(np.log(r) < threshhold, True, False)
    SQLowEnoughindices=inds[SQLowEnough]
    first = np.min(SQLowEnoughindices[SQLowEnoughindices > SQmax])

    # Evaluate which amplitude data have enough SNR to be used
    # for amplitude fitting.
    SNR = np.log(r) - NoiseFloor
    notEnoughSNR = np.where(SNR < amplitude_SNR_thresh, True, False)
    notEnoughSNRindices = inds[notEnoughSNR]
    goodRSamples = (np.min(notEnoughSNRindices[notEnoughSNRindices > first])) - first
    enoughRSamples = goodRSamples > minrsample
    lastr = first + goodRSamples

    # In order to make the amplitude plots somewhat easier to
    # read, extend the time range over which to plot the fits.
    fit_samples = lastr - first
    pltFirstr = int(first - 0.1 * fit_samples)
    pltLastr = int(lastr + 0.1 * fit_samples)

    # Evaluate which phase data are far enough from a phase
    # jump (where phase wraps through +/- 180 degrees in our
    # definition of phase).
    firstphi = bufferSamples

    try:
        goodPhiSamples = (np.min(jumpIndices[jumpIndices > firstphi])) - firstphi
    except:
        goodPhiSamples = len(inds)-2*bufferSamples
    # ensure there will be at least 10 points in the fit
    enoughPhiSamples = goodPhiSamples > bufferSamples
    lastphi = firstphi+goodPhiSamples-bufferSamples
    pltFirstphi = int(firstphi-0.1*fit_samples)
    pltLastphi = int(lastphi + 0.1 * fit_samples)

    # ensure there will be at least
    fit_samples_phi = lastphi-firstphi

    # fit to the log of the amplitude data to get the
    # exponential time decay constant (a.k.a tau)
    print('Fitting amplitude using ' + str(fit_samples)+ ' points')
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
    print('Fitting phase using ' + str(fit_samples_phi)+ ' points')
    if enoughPhiSamples:
        print('Using phase data between '+'%.3f' % (t[first])+' and ''%.3f' % (t[lastphi])+' seconds')
        phicoeffs = np.polyfit(t[firstphi:lastphi], phi[firstphi:lastphi], 1)
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
    print('burst  ringdown Q: '+'%f' % (Q))
    print('burst  ringdown frequency: '+'%.3f' % (sum)+' Hz')
    if savePLTs:
        fig1, ax1 = mpl.subplots()
        ax1.plot(t, np.log(r), label='All Data')
        ax1.legend(loc='lower right')
        ax1.set_title('Amplitude fits using lockin method')
        fig2, ax2 = mpl.subplots()
        ax2.plot(t, phi, label='All Data')
        ax2.legend(loc='lower right')
        ax2.set_title('Phase fits using lockin method')
        ax2.plot(t[firstphi:lastphi], phi[firstphi:lastphi], label='Data in phi fit')
        fig3, ax3 = mpl.subplots()
        ax3.plot(t[1:], df, label='All Data')
        ax3.scatter(t[1:][jumps], df[jumps], label='Detected jumps', c='r')

    if enoughRSamples and savePLTs:
        ax1.plot(t[first:lastr], np.log(r[first:lastr]), label='Data in Fits', c='C1', lw=4)
        ax1.plot(t[first:lastr], t[first:lastr]*rcoeffs[0] + rcoeffs[1], label='Fits', c='k', lw=2)
        ax1.grid()
        #ax1.plot(t[ptmFirst:ptmLast], ptmLog*np.ones(ptmLast-ptmFirst), label='Pre-Trigger Mean', c='dimgray', lw=4)
    if enoughPhiSamples and savePLTs:
        ax2.plot(t[firstphi:lastphi], phi[firstphi:lastphi], label='Data in Fit', c='C1', lw=4)
        ax2.plot(t[firstphi:lastphi], t[firstphi:lastphi]*phicoeffs[0] + phicoeffs[1], label='Fit', c='k', lw=2)
        ax2.grid()
    if savePLTs:
        fig1.show()
        fig2.show()
        fig3.show()


    return Q, sum, tau

##### Things to set ######

samplerate = 500e3
f_approx = 191280
buffer = 0.0005
decimation = 300
max_expected_detuning=1000
savePLTs=True
threshhold = -2
NoiseFloor = -4

file = 'example_ringdown_joe.csv'
print('Loading file.....')
#data = np.loadtxt(file, delimiter=',')
#np.save(data, 'example_ringdown_joe.npy')
data = np.load('example_ringdown_joe.npy')
print('File loaded!')
times = data[:,0]
sq = data[:,1] # make sure this is the right channel
trig = data[:,2] # make sure this is the right channel

starts, ends = findBurstsFuncGen(times, trig, f_approx, savePLTs=True)

num_pulses = len(starts)
samples_per_burst = int(np.mean(np.array(ends)-np.array(starts)))
samples_per_fit = int(0.90*samples_per_burst) # doesn't fit the points right before the next burst

Qlist = []
Flist = []
taulist = []

for ind in np.arange(num_pulses):
    start = starts[ind]
    end = ends[ind]
    t_to_fit = times[start:end]
    sq_to_fit = sq[start:end]
    # fig1, ax1 = mpl.subplots()
    # ax1.plot(times, sq)
    # ax1.plot(t_to_fit, sq_to_fit)
    # fig1.show()
    Q, F, tau = fitBurstLockin(t_to_fit, sq_to_fit, f_approx, savePLTs = savePLTs,
                buffer = buffer, decimation=decimation, threshhold=threshhold,
                max_expected_detuning=max_expected_detuning, NoiseFloor=-4)
    Qlist.append(Q)
    Flist.append(F)
    taulist.append(tau)

print(50*'*')
print(50*'*')
print(f'Average Q: {np.nanmean(Qlist):.2f}')
print(f'Std. Dev. Q: {np.nanstd(Qlist):.2f}')
print(f'Average F: {np.nanmean(Flist):.3f} Hz')
print(f'Std. Dev. F: {np.nanstd(Flist):.3f} Hz')
