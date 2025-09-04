import numpy as np
import scipy.signal as sig

# add the top directory to the python system path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import src.general.distributions as dist
from src.radar.radarUtils import radarEquation

class pulseCompRadar():
    
    def __init__(self,
                 pulseTime: int | float,
                 pulseStartFreq: int | float,
                 pulseRepFreq: int | float,
                 bandwidth: int | float,
                 sampleRate: int | float,
                 maxRange: int | float | None = None,
                 speedOfLight: int | float = 3e8):
        """ Pulse compression radar.
        
        #TODO

        Parameters
        ----------
        pulseTime : int | float
            Time of the pulse [s].
        pulseStartFreq : int | float
            Starting frequency of chrip [Hz]. The end frequency is that of the start frequency plus the bandwidth of 
            the signal.
        pulseRepFreq : int | float
            Pulse repetition frequency [Hz].
        bandwidth : int | float
            Bandwidth of pulse [Hz].
        sampleRate : int | float
            Sample rate [Hz].
        maxRange : int | float | None, optional
            Maximum range of the radar [m]. Note the that full TX signal consists of the maximum distace plus the
            length of the pulse to ensure that the full pulse is recieved in a window of the RX signal, by default 
            None.
        speedOfLight : int | float, optional
            Speed of light [m/s], by default 3e8
        """
        
        # initialise the parameters
        self.pulseTime = pulseTime
        self.pulseStartFreq = pulseStartFreq
        self.pulseRepFreq = pulseRepFreq
        self.bandwidth = bandwidth
        self.sampleRate = sampleRate
        self.speedOfLight = speedOfLight
        
        # calculate the pulse repetition time
        self.pulseRepTime = 1/self.pulseRepFreq
        
        if maxRange:
            # calculate the maximum time limit dependant on specified max radar range
            maxTime = self.__rangeToTime(maxRange)
            maxRangePoss = self.__timeToRange(self.pulseRepTime)
            
            # if the max time limit is greater than pulse repetition time limit to that, else if it is less than the 
            # pulse time limit to 4 times the pulse repetition time
            if maxTime > self.pulseRepTime:
                self.maxTime = self.pulseRepTime
                self.maxRange = self.__timeToRange(self.maxTime)
                print(f"The specified max range of {maxRange}m is greater than the max range of radar {maxRangePoss}m, limiting to a range of {self.maxRange}m")
            else:
                if maxTime > self.pulseTime:
                    self.maxTime = maxTime
                    self.maxRange = self.__timeToRange(self.maxTime)
                else:
                    self.maxTime = 4 * self.pulseRepTime
                    self.maxRange = self.__timeToRange(self.maxTime)
                    print(f"The specified max range of {maxRange}m is less than the duration of the pulse, setting to 4 times the length of the pulse {self.maxRange}m")
        else:
            self.maxTime = self.pulseRepTime
            self.maxRange = self.__timeToRange(self.maxTime)
            
        # calculate the maximum allowable range of a target so that he chirp signal is within the specified target 
        # range
        self.maxTargetRange = self.maxRange - self.__timeToRange(pulseTime)
        
        # calculate the minimmum detectable range
        self.minTargetRange = self.__timeToRange(pulseTime)
              
        # create the tx signal
        self.__createTxSignal()
        
        # determine the signal length
        self.signalLength = len(self.txSignal)
        
    def __rangeToTime(self,
                      rangeVal: int | float
                      ) -> float:
        """ Converts target range to time delay value.

        Parameters
        ----------
        rangeVal : int | float
            Range of target [m].

        Returns
        -------
        float
            Target time delay [s].
        """
        
        return 2 * rangeVal / self.speedOfLight
    
    def __timeToRange(self,
                      timeVal: int | float
                      ) -> float:
        """ Calculates the target range from the target time delay.

        Parameters
        ----------
        timeVal : int | float
            Target time delay [s].

        Returns
        -------
        float
            Range of target [m].
        """
        
        return timeVal * self.speedOfLight / 2
        
    def __createTxSignal(self):
        """ Create the transmitted chirp signal
        
        """
        
        # create tx time vector
        self.timeVector = np.arange(0, self.maxTime, 1/self.sampleRate)
        
        # create time vector of the rx signal
        chirpTimeVec = np.arange(0, self.pulseTime, 1/self.sampleRate)
        
        # create the chirp signal
        self.chirpSignal = sig.chirp(chirpTimeVec, 
                                     f0=self.pulseStartFreq, 
                                     f1=self.pulseStartFreq + self.bandwidth, 
                                     t1=self.pulseTime, 
                                     method='linear',
                                     complex=True)
        
        # add the chirp to the tx signal
        self.txSignal = np.zeros(len(self.timeVector)).astype(np.complex128)
        self.txSignal[0:len(self.chirpSignal)] = self.chirpSignal
        
    def info(self):
        """ Prints the key pulse compression radar parameters.
        """
        
        print("\nRadar Info\n")
        print(f"   Pulse Time: {self.pulseTime*1e6} us")
        print(f"   Pulse Start Frequency: {self.pulseStartFreq} s")
        print(f"   Pulse Repetition Frequency: {self.pulseRepFreq} Hz")
        print(f"   Pulse Repetition Time: {self.pulseRepTime*1e6} us")
        print(f"   Bandwidth: {self.bandwidth*1e-6:.3f} MHz")
        print(f"   Min Target Range: {self.minTargetRange*1e-3:.1f} km")
        print(f"   Max Range: {self.maxRange*1e-3:.1f} km")
        print(f"   Max Target Range: {self.maxTargetRange*1e-3:.1f} km")
        print(f"   Sample Rate: {self.sampleRate*1e-6:.3f} MHz")
        print(f"   Signal Length: {self.signalLength} samples")
        print(f"   Speed of Light: {self.speedOfLight:.0f} m/s")
        
    def getTxSignal(self) -> np.ndarray:
        """ Gets the transmitted signal.

        Returns
        -------
        np.ndarray
            Transmitted signal.
        """
        
        return self.txSignal
    
    def getTimeVector(self) -> np.ndarray:
        """ Gets the time vector associated with the transmitted and received signals.

        Returns
        -------
        np.ndarray
            Time vector [s].
        """
        
        return self.timeVector
    
    def getRangeVector(self) -> np.ndarray:
        """ Gets the range vector associated with the transmitted and received signals.

        Returns
        -------
        np.ndarray
            Range vector [m].
        """
        
        return self.speedOfLight * self.timeVector / 2

    def createRxSignal(self,
                       targetRange: int | float,
                       snr: int | float | str,
                       maxRangeSnr: int | float | None = None,
                       targetPresent: bool = True,
                       noisePresent: bool = True,
                       noiseDist: str = "complex gaussian"
                       ) -> np.ndarray:
        """ Create the received signal

        Parameters
        ----------
        targetRange : int | float
            Target range [m].
        snr : int | float
            Signal-to-noise-ratio [dB].
        maxRangeSnr : int | float | None, optional
            The signal-to-noise ratio at the maximum range of the radar [dB], by default None.
        targetPresent : bool, optional
            Specify if a target should be present of not, by default True.
        noisePresent : bool, optional
            Specify if the noise is present or not, by default True.
        noiseDist : str, optional
            Noise distribution type, by default "complex gaussian". Only required if 'noisePresent' is set to True.

        Returns
        -------
        np.ndarray
            Receieved signal.

        Raises
        ------
        ValueError
            If the target range is greater than the maximum range possible.
        ValueError
            If the target range is less than the minimum range possible.
        ValueError
            If signal-to-noise is specified as varying and no maximum range snr value is input.
        """
        
        # check target range value
        if targetRange > self.maxTargetRange:
            raise ValueError(f"the target range must less than the maximum radar range of {self.maxTargetRange}")
        elif targetRange < self.minTargetRange:
            raise ValueError(f"the target range must greater than the minumum radar range of {self.minTargetRange}")
        
        # raise an error if signal-to-noise is specified as varying and no maximum range snr value is input
        # TODO - make this better
        if isinstance(snr, str) and (maxRangeSnr is None):
            raise ValueError(f"if snr is set to varying, maxRangeSnr must have a value")
        
        # create the empty RX signal
        self.rxSignal = np.zeros(self.signalLength).astype(np.complex128)
        
        # calculate the delay to the target in samples
        targetDelay = 2 * targetRange / self.speedOfLight
        targetDelaySample = int(np.floor(targetDelay * self.sampleRate))
        
        # add the target to the signal
        if targetPresent:
            # if snr is set to 'varying' then calculate the snr based upon the target range
            # TODO - make this better
            if isinstance(snr, str):
                recPowerMin = radarEquation(self.maxTargetRange)
                recPower = radarEquation(targetRange)
                snr = 10*np.log10(recPower/recPowerMin) + maxRangeSnr
            # calculate the snr
            snrMag = 10**(snr/10) # type: ignore
            # add the chirp signal to the rx signal at the appropriate sample corresponding to the target delay
            self.rxSignal[targetDelaySample:targetDelaySample+len(self.chirpSignal)] = snrMag * self.chirpSignal
        
        # add noise to the signal
        if noisePresent:
            noise = dist.samples(distType=noiseDist,
                                 nSamples=self.signalLength)
            self.rxSignal += noise
        
        return self.rxSignal

    def pulseCompress(self,
                      applyWindow: bool = True,
                      ) -> np.ndarray:
        """ Apply pulse compression to the received signal.

        Parameters
        ----------
        applyWindow : bool, optional
            Select if to apply the window to the chirp signal or not, by default True.

        Returns
        -------
        np.ndarray
            Pulse compressed signal.

        Raises
        ------
        Exception
            If no received signal has been created.
        """
        
        # determine if the received signal has be been created first
        try:
            temp = self.rxSignal
        except NameError:
            raise Exception("no RX signal has been created, please run the 'createRxSignal' method before runnning this one")
        
        if applyWindow:
            # apply to chirp signal
            winChirp = self.chirpSignal * np.hamming(len(self.chirpSignal))
            
            # match filter signal
            pcSignal = np.correlate(self.rxSignal, winChirp, "full")
        else:
            # match filter signal
            pcSignal = np.correlate(self.rxSignal, self.chirpSignal, "full")

        # remove negative lag values
        pcSignal = pcSignal[len(self.chirpSignal)-1:]
        
        return pcSignal

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # parameters    
    pulseTime = 20e-6
    pulseStartFreq = 0
    pulseRepFreq = 5e2
    bandwidth = 10e6
    maxRange = 60e3
    fs = 20e6
    
    # create instance of the class
    pcr = pulseCompRadar(pulseTime=pulseTime,
                         pulseStartFreq=pulseStartFreq,
                         pulseRepFreq=pulseRepFreq,
                         bandwidth=bandwidth,
                         sampleRate=fs,
                         maxRange=maxRange)
    
    # print the radar info
    pcr.info()
    
    # get the tx signal and time vector
    txSignal = pcr.getTxSignal()
    timeVector = pcr.getTimeVector() * 1e3
    rangeVector = pcr.getRangeVector()
    
    # process rx signal
    targetRange = 20e3
    snr = 10
    rxSignal = pcr.createRxSignal(targetRange=targetRange,
                                  snr=snr,
                                  noisePresent=True)
    
    # pulse compress the rx signal
    pcSignal = pcr.pulseCompress()
    pcSignalNoWin = pcr.pulseCompress(applyWindow=False)
    
    # process rx signal
    targetRange = 20e3
    snr = "varying"
    rxSignalVaryingSnr = pcr.createRxSignal(targetRange=targetRange,
                                            snr=snr,
                                            maxRangeSnr=-10,
                                            noisePresent=True)
    
    # plot the main signals
    plt.figure(figsize=(12, 12))
    
    plt.subplot(321)
    plt.plot(timeVector, np.real(txSignal))
    plt.grid()
    plt.title("TX Signal")
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [V]")
    
    plt.subplot(322)
    plt.plot(rangeVector, np.real(txSignal))
    plt.grid()
    plt.title("TX Signal")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    
    plt.subplot(323)
    plt.plot(timeVector, np.real(rxSignal))
    plt.grid()
    plt.title("RX Signal")
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [V]")
    
    plt.subplot(324)
    plt.plot(rangeVector, np.real(rxSignal))
    plt.grid()
    plt.title("RX Signal")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    
    plt.subplot(325)
    plt.plot(timeVector, np.abs(pcSignal))
    plt.grid()
    plt.title("PC Signal")
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [V]")
    
    plt.subplot(326)
    plt.plot(rangeVector, np.abs(pcSignal))
    plt.grid()
    plt.title("PC Signal")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    
    # plot the rx signal with a static snr versus an varying snr
    plt.figure(figsize=(12, 12))
    
    plt.subplot(211)
    plt.plot(rangeVector, np.real(rxSignal), label="static")
    plt.grid()
    plt.title("RX Signal - Static SNR")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    
    plt.subplot(212)
    plt.plot(rangeVector, np.real(rxSignalVaryingSnr), label="varying")
    plt.grid()
    plt.title("RX Signal - Varying SNR")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    
    # plot the pulse compression signal versus the windowed pulse compression signal and same again but with the 
    # windowed normalised to the original pulse compressed signal 
    plt.figure(figsize=(12, 12))
    
    plt.subplot(211)
    plt.plot(rangeVector, np.abs(pcSignal), label="win")
    plt.plot(rangeVector, np.abs(pcSignalNoWin), label="no win")
    plt.grid()
    plt.legend()
    plt.title("PC Signal - Windowing vs. No Windowing")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    plt.xlim([targetRange-500, targetRange+500])
    
    plt.subplot(212)
    plt.plot(rangeVector, np.abs(pcSignal), label="win")
    plt.plot(rangeVector, np.abs(pcSignalNoWin) * np.max(np.abs(pcSignal)) / np.max(np.abs(pcSignalNoWin)), label="no win")
    plt.grid()
    plt.legend()
    plt.title("PC Signal - Windowing vs. No Windowing - Normalised")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    plt.xlim([targetRange-500, targetRange+500])
    
    plt.show()