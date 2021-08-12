import RingdownFunctions as ringdown
import os

# 1) Load data
directory = '/Volumes/DMRADIO-E/run-in-progress/hiQ/20210810_HQ_dip/Ringdowns'
file = '20210810_140809ringdown5_FLL_multiple_356_khz.csv'
filename = os.path.join(directory, file) # add the filename for your scope trace csv here
channels = [1,3] # pick what channels from the scope trace you want, currently assumes channel 3 is the func gen and 1 is SQUID
data = ringdown.getScopeTraceData(filename, channels, savePLTs=True)

# 2) set ringdown settings
times = data[:,0]
amp = data[:,1]
trig = data[:,2] # this assumes that the trigger (the function gen output) is the third column
samplerate = 5e6
f_approx = 356000
p0 = None # initial conditions guess for fit (can leave this blank)

## these settings are described better in RingdownFunctions.py
startbuffer = 0.5
endbuffer = 0.1
fitbuffer = 5
fitfactor = 1.5

savePLTs = True

# 3) fit the ringdown data
Qav, Qstd, Fav, Fstd, tauav, taustd= ringdown.fitRingdownSine(times, amp, trig, samplerate, f_approx, p0, startbuffer,
                    endbuffer, fitbuffer, fitfactor, savePLTs)
