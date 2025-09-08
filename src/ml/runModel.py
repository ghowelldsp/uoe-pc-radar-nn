from pathlib import Path
import numpy as np
import torch
from torch import nn
from typing import Dict

# add the top directory to the python system path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import src.ml.dataloader as dl
import src.ml.models as mdl
from src.ml.utils import availableDevices, setSeeds
import src.ml.trainTest as tt
from src.ml.saveResults import modelInfoDict, saveAllResults
from src.ml.resultsEval import plotLossCurves
from src.general.distributions import distShortName
    
def checkDatasetsExist(dataDirPath: str,
                       datasets: Dict,
                       ) -> None:
    """ Checks if all datasets exist

    Parameters
    ----------
    dataDirPath : str
        The path to the data directory.
    datasets : Dict
        Dictionary of the datasets. 
        
        In the form:
            {"datasetDirectory":
                [
                    ["noiseDist1", "noiseDist2"],
                    ["snr1", "snr2"],
                    ["signalType1", "signalType2"],
                ]
            }
            
        For example:
            {"syntheticData": 
                [
                    ["complex gaussian", "complex weibull"], 
                    ["10", "5"], 
                    ["pc", "rx"]
                ]
            }

    Raises
    ------
    ValueError
        If data directory does not exist
    ValueError
        If dataset directory does not exist
    ValueError
        If dataset does not exist
    """
    
    # check if data path exists
    if not os.path.isdir(dataDirPath):
        raise ValueError(f"data directory does not exist")
    
    # create a list of dataset directories (i.e. the keys of the datasets dictionary)
    datasetDirs = list(datasets.keys())
    
    # check all datasets
    for datasetDir in datasetDirs:
        
        # check if dataset directory exist
        datasetDirPath = f"{dataDirPath}/{datasetDir}"
        if not os.path.isdir(datasetDirPath):
            raise ValueError(f"dataset directory {datasetDir} does not exist")
        
        # get noise types, snrs and signal types from dataset dictionary
        noiseTypes = datasets[datasetDir][0]
        snrs = datasets[datasetDir][1]
        signalTypes = datasets[datasetDir][2]
        
        # check if datasets exist
        for noiseType in noiseTypes:
            
            # get the short name of the noise dis
            noiseDist = distShortName(noiseType)
            
            for snr in snrs:
                for signalType in signalTypes:
                    
                    # create path to dataset
                    datasetPath = f"{datasetDirPath}/noise_{noiseDist}_snr_{snr}/{signalType}"
                    
                    # check if dataset exists
                    if not os.path.isdir(datasetPath):
                        raise ValueError(f"dataset directory {datasetPath} does not exist")
                    
def trainModels(datasets: Dict,
                modelParams: Dict,
                evalModel: bool = True,
                plotResults: bool = True,
                verboseOutput: bool = True,
                ) -> None:
    """ Trains model all datasets

    Parameters
    ----------
    datasets : Dict
        Dictionary of the datasets. 
            In the form:
                {"datasetDirectory":
                    [
                        ["noiseDist1", "noiseDist2"],
                        ["snr1", "snr2"],
                        ["signalType1", "signalType2"],
                    ]
                }
                
            For example:
                {"syntheticData": 
                    [
                        ["complex gaussian", "complex weibull"], 
                        ["10", "5"], 
                        ["pc", "rx"]
                    ]
                }
    modelParams: Dict
        Dictionary of Pytorch model parameters.
            In the form:
                {"nEpochs": int,
                    Number of epochs.
                 "batchSize": int,
                    Batch size to create.
                 "modelType": str,
                    Model to train on.
                 "loss function": str,
                    Loss function.
                 "optimiser": "Adam",
                    Optimiser.
                 "optimiser learn rate": int | float}
                    Optimister learn rate.
                }
            For example:
                {"nEpochs": 2,
                 "batchSize": 2,
                 "modelType": "dynamicCNN1",
                 "loss function": "Cross Entropy Loss",
                 "optimiser": "Adam",
                 "optimiser learn rate": 1e-4
                 }
    evalModel: bool
        Evaluate trained model, defaults to True.
    plotResults : bool
        Plot output results after training, defaults to True.
    verboseOutput : bool
        Verbose output, defaults to True.
    """
    
    # change to the current working directory of the script
    scriptPath = os.path.abspath(sys.argv[0])
    os.chdir(os.path.dirname(scriptPath))
    
    # data directory path
    dataDirPath = "../data"
    
    # check if all datasets exist
    checkDatasetsExist(dataDirPath,
                       datasets)
    
    # create a list of dataset directories (i.e. the keys of the datasets dictionary)
    datasetDirs = list(datasets.keys())
    
    # run for all datasets
    for datasetDir in datasetDirs:
        
        # create dataset directory path
        datasetDirPath = f"{dataDirPath}/{datasetDir}"
        
        # get noise types, snrs and signal types from dataset dictionary
        noiseTypes = datasets[datasetDir][0]
        snrs = datasets[datasetDir][1]
        signalTypes = datasets[datasetDir][2]
        
        for noiseType in noiseTypes:
        
            # get the short name of the noise dis
            noiseDist = distShortName(noiseType)
            
            for snr in snrs:
                for signalType in signalTypes:
                    
                    # create path to dataset
                    datasetPath = f"{datasetDirPath}/noise_{noiseDist}_snr_{snr}/{signalType}"
        
                    # run model
                    trainModel(datasetPath=datasetPath, 
                               modelParams=modelParams,
                               evalModel=evalModel,
                               plotResults=plotResults, 
                               verboseOutput=verboseOutput)

def trainModel(datasetPath: str,
               modelParams: Dict,
               evalModel: bool, 
               plotResults: bool,
               verboseOutput: bool,
               ) -> None:
    """ Trains model on a dataset

    Parameters
    ----------
    datasetPath : str
        Path to dataset to train on.
    modelParams: Dict
        Dictionary of Pytorch model parameters.
            In the form:
                {"nEpochs": int,
                    Number of epochs.
                 "batchSize": int,
                    Batch size to create.
                 "modelType": str,
                    Model to train on.
                 "loss function": str,
                    Loss function.
                 "optimiser": "Adam",
                    Optimiser.
                 "optimiser learn rate": int | float}
                    Optimister learn rate.
                }
            For example:
                {"nEpochs": 2,
                 "batchSize": 2,
                 "modelType": "dynamicCNN1",
                 "loss function": "Cross Entropy Loss",
                 "optimiser": "Adam",
                 "optimiser learn rate": 1e-4
                 }
    evalModel: bool
        Evaluate trained model.
    plotResults : bool
        Plot output results after training.
    verboseOutput : bool
        Verbose output.
    """
    
    print(f"\nRunning on dataset: {datasetPath}")
    
    # set random seed 
    setSeeds()
    
    # create dataloaders
    trainDL, testDL, evalDL = dl.createDataloaders(datasetPath=datasetPath,
                                                   batchSize=modelParams["batchSize"],
                                                   verboseOutput=verboseOutput)

    # check availible devices
    device = availableDevices()
    
    # load one data sample to get the input feature length
    # TODO - make this better
    signalLen = len(np.load(f"{datasetPath}/train/target/0.npy"))
    
    # load model
    model = mdl.loadModel(modelType=modelParams["modelType"],
                          device=device,
                          signalLen=signalLen,
                          verboseOutput=verboseOutput)

    print("\nTraining and Testing Model\n")

    # select loss function
    match modelParams["loss function"]:
        case "Cross Entropy Loss":
            lossFn = nn.CrossEntropyLoss()
        case "Binary Cross Entropy":
            lossFn = nn.BCELoss()
        case _:
            raise ValueError(f"loss function {modelParams['loss function']} is invalid")
        
    # select optimiser
    match modelParams["optimiser"]:
        case "Adam":
            optimizer = torch.optim.Adam(params=model.parameters(), 
                                         lr=modelParams['optimiser learn rate'])
        case _:
            raise ValueError(f"optimizer {modelParams['optimiser']} is invalid")

    # train and test function
    resultsTT = tt.train(model=model,
                         trainDL=trainDL,
                         testDL=testDL,
                         lossFn=lossFn,
                         optimizer=optimizer,
                         epochs=modelParams["nEpochs"],
                         device=device)
    
    if evalModel:
        print("\nEvaluating Model\n")
        # run evaluation on trained model
        resultsPT = tt.eval(model=model,
                            dataloader=evalDL,
                            lossFn=lossFn,
                            device=device)
    else:
        resultsPT = None
    
    print("\nResults\n")
    
    # create dictionary of model information
    modelInfo : modelInfoDict = {
        "model": model,
        "signalLength": signalLen,
        "modelType": modelParams['modelType'],
        "device": device,
        "lossFn": modelParams['loss function'],
        "optimiser": modelParams['optimiser'],
        "optimiserLR": modelParams['optimiser learn rate'],
        "nEpochs": modelParams['nEpochs'],
        "batchSize":  modelParams['batchSize'],
        "nTrainBatchs": len(trainDL),
        "nTrainSamples": len(trainDL.dataset), # type: ignore
        "nTestBatchs": len(testDL),
        "nTestSamples": len(testDL.dataset), # type: ignore
        "nEvalBatchs": len(evalDL),
        "nEvalSamples": len(evalDL.dataset), # type: ignore
    }
    
    # save results
    saveAllResults(datasetPath=datasetPath,
                   modelInfo=modelInfo,
                   resultsTT=resultsTT,
                   resultsPT=resultsPT)
    
    # plot the loss curves
    if plotResults:
        plotLossCurves(resultsTT)
        
    print("\nCompleted\n")
