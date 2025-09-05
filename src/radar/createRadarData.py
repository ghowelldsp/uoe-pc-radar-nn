import numpy as np
import os, sys, inspect, shutil
import matplotlib.pyplot as plt

from src.general.utils import createDirectory, removeDirectory
from src.radar.pulseCompressionRadar import pulseCompRadar
from src.general.distributions import distShortName

def createExampleSignal(radarParams: dict) -> None:
    """ Create an example signal using the parameters set in the radar params.
    
    # TODO

    Parameters
    ----------
    radarParams : dict
        Dictionary of radar parameters.
            {
             pulseTime, # TODO
             pulseStartFreq, 
             pulseRepFreq, 
             bandwidth, 
             maxRange, 
             sampleRate, 
             targetRange, 
             snr,
             maxRangeSnr, 
             targetPresent, 
             noisePresent,
             noiseDist
             }
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
             radarParams: dict,
             nTrainData: int,
             nTestData: int,
             nEvalData: int,
             ) -> None:
    """ Save an information file of the key data about the model.

    Parameters
    ----------
    dataPath : str
        Path to the model directory.
    radarParams : dict
        Dictionary of radar parameters.
            {
             pulseTime, # TODO
             pulseStartFreq, 
             pulseRepFreq, 
             bandwidth, 
             maxRange, 
             sampleRate, 
             targetRange, 
             snr,
             maxRangeSnr, 
             targetPresent, 
             noisePresent,
             noiseDist
             }
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
                     radarParams: dict
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
    radarParams : dict
        Dictionary of radar parameters.
            {
             pulseTime, # TODO
             pulseStartFreq, 
             pulseRepFreq, 
             bandwidth, 
             maxRange, 
             sampleRate, 
             targetRange, 
             snr,
             maxRangeSnr, 
             targetPresent, 
             noisePresent,
             noiseDist
             }
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
                    radarParams: dict,
                    nTrainData: int,
                    nTestData: int,
                    nEvalData: int,
                    verbose: bool = False,
                    ) -> None:
    """ Creates all the signal data for the model.

    Parameters
    ----------
    dataFolder : str
        Path to the main data directory.
    radarParams : dict
        Dictionary of radar parameters.
            {
             pulseTime, # TODO
             pulseStartFreq, 
             pulseRepFreq, 
             bandwidth, 
             maxRange, 
             sampleRate, 
             targetRange, 
             snr,
             maxRangeSnr, 
             targetPresent, 
             noisePresent,
             noiseDist
             }
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

def createModelData(radarParams: dict,
                    nTrainData: int,
                    nTestData: int,
                    nEvalData: int,
                    exampleSignal: bool,
                    createData: bool,
                    verbose: bool = False
                    ) -> None:
    
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
    
    # save the info
    saveInfo(dataPath=f"../data/{dataPath}",
             radarParams=radarParams,
             nTrainData=nTrainData,
             nTestData=nTestData,
             nEvalData=nEvalData)

    for dist in radarParams["noiseDist"]:
        for snr in radarParams["snr"]:

            # radar params
            radarParamsSingle = {
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
                createExampleSignal(radarParams)

            # create model data
            if createData:
                createRadarData(dataPath=dataPath,
                                radarParams=radarParamsSingle,
                                nTrainData=nTrainData, 
                                nTestData=nTestData, 
                                nEvalData=nEvalData,
                                verbose=verbose)
            
    print("\nCOMPLETED\n")