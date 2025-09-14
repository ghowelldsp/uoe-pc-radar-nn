import numpy as np
import os, sys, inspect
import matplotlib.pyplot as plt
from typing import TypedDict, List

from src.general.utils import createDirectory, removeDirectory, setSeeds
from src.radar.pulseCompressionRadar import pulseCompRadar
from src.general.distributions import distShortName

class RadarParamsFull(TypedDict):
    """ Definition of radar parameters structure

    In the form:
        pulseTime: float
            The pulses time [s].
        pulseStartFreq: int | float
            Pulse start frequency [Hz].
        pulseRepFreq: int | float
            Pulse repetition frequency [Hz].
        bandwidth: int | float
            Bandwidth [Hz].
        maxRange: int | float
            Max range [m].
        sampleRate: int | float
            Sample rate [Hz].
        targetRange: int | float
            Target range [m].
        snr: List[int | float]
            List of SNR's to create data for, in dB.
        maxRangeSnr: int | float
            SNR at maximum range of radar in dB, specified by 'maxRange'.
        targetPresent: bool
            Creates a target signal if set to true.
        noisePresent: bool
            Creates noise signal if set to true.
        noiseDist: List[str]
            List of noise distribution types to create data for.
    """
    pulseTime: float
    pulseStartFreq: int | float
    pulseRepFreq: int | float
    bandwidth: int | float
    maxRange: int | float
    sampleRate: int | float
    targetRange: int | float
    snr: List[int | float | str]
    maxRangeSnr: int | float
    targetPresent: bool
    noisePresent: bool
    noiseDist: List[str]
    
class RadarParamsSingle(TypedDict):
    """ Definition of radar parameters structure

    In the form:
        pulseTime: float
            The pulses time [s].
        pulseStartFreq: int | float
            Pulse start frequency [Hz].
        pulseRepFreq: int | float
            Pulse repetition frequency [Hz].
        bandwidth: int | float
            Bandwidth [Hz].
        maxRange: int | float
            Max range [m].
        sampleRate: int | float
            Sample rate [Hz].
        targetRange: int | float
            Target range [m].
        snr: int | float
            SNR [dB].
        maxRangeSnr: int | float
            SNR at maximum range of radar in dB, specified by 'maxRange'.
        targetPresent: bool
            Creates a target signal if set to true.
        noisePresent: bool
            Creates noise signal if set to true.
        noiseDist: List[str]
            Noise distribution type.
    """
    pulseTime: float
    pulseStartFreq: int | float
    pulseRepFreq: int | float
    bandwidth: int | float
    maxRange: int | float
    sampleRate: int | float
    targetRange: int | float
    snr: int | float | str
    maxRangeSnr: int | float
    targetPresent: bool
    noisePresent: bool
    noiseDist: str

def createExampleSignal(radarParams: RadarParamsSingle) -> None:
    """ Create an example signal using the parameters set in the radar params.

    Parameters
    ----------
    radarParams : RadarParamsSingle
        Dictionary of radar parameters. Full specification can be found in 'RadarParamsSingle' class definition.
    """
    
    # create instance of the class
    pcr = pulseCompRadar(pulseTime=radarParams["pulseTime"],
                         pulseStartFreq=radarParams["pulseStartFreq"],
                         pulseRepFreq=radarParams["pulseRepFreq"],
                         bandwidth=radarParams["bandwidth"],
                         sampleRate=radarParams["sampleRate"],
                         maxRange=radarParams["maxRange"])
    
    # print the radar info
    pcr.info()
    
    # process rx signal
    rxSignal = pcr.createRxSignal(targetRange=radarParams["targetRange"],
                                  snr=radarParams["snr"],
                                  maxRangeSnr=radarParams["maxRangeSnr"],
                                  targetPresent=radarParams["targetPresent"],
                                  noisePresent=radarParams["noisePresent"],
                                  noiseDist=radarParams["noiseDist"])
    
    # pulse compress signal
    pcSignal = pcr.pulseCompress()
    
    # get the range vector
    rangeVector = pcr.getRangeVector()

    # plot the radar signal
    plt.figure()
    
    plt.subplot(121) 
    plt.plot(rangeVector, np.real(rxSignal))
    plt.grid()
    plt.title("Radar RX Signal")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    plt.xlim([rangeVector[0], rangeVector[-1]])
    
    plt.subplot(122)
    plt.plot(rangeVector, np.abs(pcSignal))
    plt.grid()
    plt.title("Radar PC Signal")
    plt.xlabel("Range [m]")
    plt.ylabel("Amplitude [V]")
    plt.xlim([rangeVector[0], rangeVector[-1]])
        
    plt.show()
        
def __createDirectories(dataPath: str,
                        verbose: bool = False
                        ) -> None:
    """ Creates all the directories required for the data.

    Parameters
    ----------
    dataPath : str
        Path to the main directory folder.
    verbose : bool
        Print debug information, defaults to False.
    """
    
    # remove and create specified folders in path directory
    removeDirectory(dataPath, verbose=verbose)
    createDirectory(dataPath, verbose=verbose)
    
    for signalType in ["rx", "pc"]:
        createDirectory(f"{dataPath}/{signalType}", verbose=verbose)
        for dirName in ["test", "train", "eval"]:
            createDirectory(f"{dataPath}/{signalType}/{dirName}", verbose=verbose)
            for className in ["noise", "target"]:
                createDirectory(f"{dataPath}/{signalType}/{dirName}/{className}", verbose=verbose)
                
def saveInfo(dataPath: str,
             radarParams: RadarParamsFull,
             sigLen: int,
             nTrainData: int,
             nTestData: int,
             nEvalData: int,
             ) -> None:
    """ Save an information file of the key data about the model.
    
    # TODO - might be better to get the radar info from the class info method and use this instead of the radar params

    Parameters
    ----------
    dataPath : str
        Path to the model directory.
    radarParams : RadarParamsFull
        Dictionary of radar parameters. Full specification can be found in 'RadarParamsFull' class definition.
    sigLen: int
        Length of the signal.
    nTrainData : int
        Number of training data samples.
    nTestData : int
        Number of testing data samples.
    nEvalData : int
        Number of evaluation data samples.
    """
    
    fileName = f"{dataPath}/info.txt"
    
    try:
        with open(fileName, "w") as f:  # "x" = create, fail if exists
            # write the params
            f.write("SAMPLES\n")
            f.write(f"Training Samples: {nTrainData}\n")
            f.write(f"Testing Samples: {nTestData}\n")
            f.write(f"Evaulation Samples: {nEvalData}\n")
            
            f.write("\nRADAR PARAMS\n")
            f.write(f"Pulse Time: {radarParams['pulseTime']} s\n")
            f.write(f"Pulse Start Frequency: {radarParams['pulseStartFreq']} Hz\n")
            f.write(f"Pulse Repetition Frequency: {radarParams['pulseRepFreq']} Hz\n")
            f.write(f"Bandwidth: {radarParams['bandwidth']} Hz\n")
            f.write(f"Max Range: {radarParams['maxRange']} m\n")
            f.write(f"Sample Rate: {radarParams['sampleRate']} Hz\n")
            f.write(f"Signal Length: {sigLen} samples\n")
            
            f.close()
    except PermissionError:
        print(f"Permission denied: Unable to create '{fileName}'.")
        raise
    except Exception as error:
        print(f"An error occurred: {error}")
        raise

def createSignalData(nDataSamples: int,
                     dataPath: str,
                     dataType: str,
                     pcr: pulseCompRadar,
                     radarParams: RadarParamsSingle
                     ) -> None:
    """ Creates the received and pulse compressed signal data

    Parameters
    ----------
    nDataSamples : int
        Number of data samples.
    dataPath : str
        Path to data folder.
    dataType : str
        Type of data, i.e. training, testing or evaluation data.
    pcr : pulseCompRadar
        Handle to the pulse compression radar class.
    radarParams : RadarParamsSingle
        Dictionary of radar parameters. Full specification can be found in 'RadarParamsSingle' class definition.
    """
    
    # create random the target ranges
    minRange = np.ceil(pcr.minTargetRange)
    maxRange = np.floor(pcr.maxTargetRange)
    targetRanges = np.random.randint(minRange, maxRange, nDataSamples)
    
    for i in range(nDataSamples):
        
        if not(i % int(nDataSamples/10)):
            print(f"  Creating samples {i}/{nDataSamples}")
        
        # create the rx target signal
        rxSignal = pcr.createRxSignal(targetRange=targetRanges[i],
                                      snr=radarParams["snr"],
                                      maxRangeSnr=radarParams["maxRangeSnr"],
                                      targetPresent=True,
                                      noisePresent=True,
                                      noiseDist=radarParams["noiseDist"])
        
        # create the pc target signal
        pcSignal = pcr.pulseCompress()
        
        # save target data to file
        np.save(f"{dataPath}/rx/{dataType}/target/{i}.npy", np.real(rxSignal).astype(np.float32))
        np.save(f"{dataPath}/pc/{dataType}/target/{i}.npy", np.abs(pcSignal).astype(np.float32))

        # create the rx noise data
        rxSignal = pcr.createRxSignal(targetRange=targetRanges[i],
                                      snr=radarParams["snr"],
                                      maxRangeSnr=radarParams["maxRangeSnr"],
                                      targetPresent=False,
                                      noisePresent=True,
                                      noiseDist=radarParams["noiseDist"])
        
        # create the pc noise data
        pcSignal = pcr.pulseCompress()
        
        # save noise data to file
        np.save(f"{dataPath}/rx/{dataType}/noise/{i}.npy", np.real(rxSignal).astype(np.float32))
        np.save(f"{dataPath}/pc/{dataType}/noise/{i}.npy", np.abs(pcSignal).astype(np.float32))

def createRadarData(dataPath: str,
                    radarParams: RadarParamsSingle,
                    nTrainData: int,
                    nTestData: int,
                    nEvalData: int,
                    verbose: bool = False,
                    ) -> int:
    """ Creates all the signal data for the model.

    Parameters
    ----------
    dataFolder : str
        Path to the main data directory.
    radarParams : RadarParamsSingle
        Dictionary of radar parameters. Full specification can be found in 'RadarParamsSingle' class definition.
    nTrainData : int
        Number of training data samples.
    nTestData : int
        Number of testing data samples.
    nEvalData : int
        Number of evaluation data samples.
    verbose : bool
        Print debug information, defaults to False.

    Raises
    ------
    ValueError
        If selected noise distribution is not one of the options.
    """
    
    print("\nDIRECTORY CREATION\n")
    
    # get short name of noise distribution
    noiseDist = distShortName(radarParams["noiseDist"])
        
    # create data paths
    folderName = f"noise_{noiseDist}_snr_{radarParams['snr']}"
    fullPath = f"../data/{dataPath}/{folderName}"
    
    # remove and create the directories
    __createDirectories(fullPath, verbose=verbose)
    
    print("\nDATA CREATION\n")
    
    # create instance of the class
    pcr = pulseCompRadar(pulseTime=radarParams["pulseTime"],
                         pulseStartFreq=radarParams["pulseStartFreq"],
                         pulseRepFreq=radarParams["pulseRepFreq"],
                         bandwidth=radarParams["bandwidth"],
                         sampleRate=radarParams["sampleRate"],
                         maxRange=radarParams["maxRange"])
    
    # get the signal length
    sigLen = pcr.signalLength
    
    print("Training Data")
    
    createSignalData(nDataSamples=nTrainData,
                     dataPath=fullPath,
                     dataType="train",
                     pcr=pcr,
                     radarParams=radarParams)
    
    print("Testing Data")
    
    createSignalData(nDataSamples=nTestData,
                     dataPath=fullPath,
                     dataType="test",
                     pcr=pcr,
                     radarParams=radarParams)
        
    print("Evaluation Data")
    
    createSignalData(nDataSamples=nEvalData,
                     dataPath=fullPath,
                     dataType="eval",
                     pcr=pcr,
                     radarParams=radarParams)
    
    return sigLen

def createModelData(radarParams: RadarParamsFull,
                    nTrainData: int,
                    nTestData: int,
                    nEvalData: int,
                    exampleSignal: bool,
                    createData: bool,
                    verbose: bool = False
                    ) -> None:
    """ Creates all the model data

    Parameters
    ----------
    radarParams : RadarParamsFull
        Dictionary of radar parameters. Full specification can be found in 'RadarParamsFull' class definition.
    nTrainData : int
        Number of training data samples.
    nTestData : int
        Number of testing data samples.
    nEvalData : int
        Number of evaluation data samples.
    exampleSignal : bool
        Create an example signal if set to true.
    createData : bool
        Creates the model data if set to true.
    verbose : bool, optional
        Prints of debug information if true, by default False.
    """
    
    # get the name of the file that called it
    callerFrame = inspect.stack()[1]
    dataPath, _ = os.path.splitext(os.path.basename(callerFrame.filename))
    
    # change to the current working directory of the script
    scriptPath = os.path.abspath(sys.argv[0])
    os.chdir(os.path.dirname(scriptPath))
    
    # create the data folder if not present
    print("creating the main data folder ...")
    createDirectory("../data", verbose=verbose)
    
    # create the model data folder
    print("creating the model data folder ...")
    createDirectory(f"../data/{dataPath}", verbose=verbose)
    
    # set random seeds
    setSeeds()

    for dist in radarParams["noiseDist"]:
        for snr in radarParams["snr"]:

            # radar params
            radarParamsSingle : RadarParamsSingle = {
                "pulseTime": radarParams["pulseTime"],
                "pulseStartFreq": radarParams["pulseStartFreq"],
                "pulseRepFreq": radarParams["pulseRepFreq"],
                "bandwidth": radarParams["bandwidth"],
                "maxRange": radarParams["maxRange"],
                "sampleRate": radarParams["sampleRate"],
                "targetRange" : radarParams["targetRange"],
                "snr": snr,
                "maxRangeSnr" : radarParams["maxRangeSnr"],
                "targetPresent" : radarParams["targetPresent"],
                "noisePresent" : radarParams["noisePresent"],
                "noiseDist" : dist
                }
                
            # create an example signal, prints the info and plots
            if exampleSignal:
                createExampleSignal(radarParamsSingle)

            # create model data
            if createData:
                sigLen = createRadarData(dataPath=dataPath,
                                         radarParams=radarParamsSingle,
                                         nTrainData=nTrainData, 
                                         nTestData=nTestData, 
                                         nEvalData=nEvalData,
                                         verbose=verbose)
    
    # save the info
    if createData:
        saveInfo(dataPath=f"../data/{dataPath}",
                 radarParams=radarParams,
                 sigLen=sigLen, # type: ignore
                 nTrainData=nTrainData,
                 nTestData=nTestData,
                 nEvalData=nEvalData)
    
    print("\nCOMPLETED\n")