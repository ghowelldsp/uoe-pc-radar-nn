import torch
from torchinfo import summary
from torch import nn
import numpy as np
import datetime
import inspect
import os
from pathlib import Path
from typing import TypedDict, Dict, List

from src.general.utils import createDirectory

class modelInfoDict(TypedDict):
    """ Model information dictionary definition.

    The dictionary contains all the model information that is saved in the results file.
    
    In the form:
        model: nn.Module
            Pytorch model used.
        signalLength: int
            Length of the signal in the training, testing and evaluation.
        modelType: str
            The model used.
        device: torch.device
            The device used.
        lossFn: str
            The loss function.
        optimiser: str
            The optimiser
        nEpochs: int
            The number of epochs.
        batchSize: int
            The batch size.
        nTrainBatchs: int
            The number of training batches.
        nTrainSamples: int
            The number of training samples.
        nTestBatchs: int
            The number of testing batches.
        nTestSamples: int
            The number of testing samples.
        nEvalBatchs: int
            The number of evaluation batches.
        nEvalSamples: int
            The number of evaluation samples.
        
    """
    model: nn.Module
    signalLength: int
    modelType: bool
    device: torch.device
    lossFn: str
    optimiser: str
    optimiserLR: float
    nEpochs: int
    batchSize: int
    nTrainBatchs: int
    nTrainSamples: int
    nTestBatchs: int
    nTestSamples: int
    nEvalBatchs: int
    nEvalSamples: int

def saveTrainTestResults(datasetPathResults: Path,
                         fileName: str,
                         results: Dict[str, List]
                         ) -> None:
    """ Saves the training and testing loss and accuracy results.

    Parameters
    ----------
    datasetResultsPath : Path
        Path to dataset.
    fileName : str
        Name of the results file.
    results : Dict[str, list]
        A dictionary of training and testing loss as well as training and testing accuracy metrics. Each metric has a 
        value in a list for each epoch.     
            In the form: 
                {trainLoss: [...],
                trainAcc: [...],
                testLoss: [...],
                testAcc: [...]} 
            For example if training for nEpochs = 2: 
                {trainLoss: [2.0616, 1.0537],
                trainAcc: [0.3945, 0.3945],
                testLoss: [1.2641, 1.5706],
                testAcc: [0.3400, 0.2973]}
    """
    
    # save results
    np.save(f"{datasetPathResults}/{fileName}_TT_data.npy", results) # type: ignore
    
def saveProbTargetResults(datasetPathResults: Path,
                          fileName: str,
                          results: Dict[str, np.ndarray] | None
                          ) -> None:
    """ Saves the probability and target results.
    
    This is generated from when evaluating the model and used to compute the ROC curves.

    Parameters
    ----------
    datasetResultsPath : Path
        Path to dataset.
    fileName : str
        Name of the results file.
    results : Dict[str, np.ndarray] | None
        # TODO - define
    """
    
    # save results
    np.save(f"{datasetPathResults}/{fileName}_PT_data.npy", results) # type: ignore
    
def saveInfoResults(datasetPathResults: Path,
                    fileName: str,
                    modelInfo: modelInfoDict,
                    results: Dict[str, List],
                    ) -> None:
    """ Save information file.

    Parameters
    ----------
    datasetPathResults : Path
        Path to dataset.
    fileName : str
        Name of the results file.
    modelInfo : modelInfoDict
        Dictionary of model information parameters. Full specification can be found in modelInfoDict class definition.
    results : Dict[str, List]
        A dictionary of training and testing loss as well as training and testing accuracy metrics. Each metric has a 
        value in a list for each epoch.     
            In the form: 
                {trainLoss: [...],
                trainAcc: [...],
                testLoss: [...],
                testAcc: [...]} 
            For example if training for nEpochs = 2: 
                {trainLoss: [2.0616, 1.0537],
                trainAcc: [0.3945, 0.3945],
                testLoss: [1.2641, 1.5706],
                testAcc: [0.3400, 0.2973]}
    """
    
    # create and open the file
    f = open(f"{datasetPathResults}/{fileName}_info.txt", "w")
    
    # get the current date and time
    dt = datetime.datetime.now()
    
    # write basic params
    f.write("TEST INFO\n")
    f.write(f"Date: {dt.day}/{dt.month}/{dt.year}\n")
    f.write(f"Time: {dt.hour}:{dt.minute}\n")
    
    # write basic params
    f.write("\nMODEL INFO\n")
    f.write(f"Model Type: {modelInfo['modelType']}\n")
    f.write(f"Device: {modelInfo['device']}\n")
    f.write(f"Loss Function: {modelInfo['lossFn']}\n")
    f.write(f"Optimiser: {modelInfo['optimiser']}\n")
    f.write(f"Optimiser Loss Rate: {modelInfo['optimiserLR']}\n")
    f.write(f"No. of Epochs: {modelInfo['nEpochs']}\n")
    f.write(f"Batch Size: {modelInfo['batchSize']}\n")
    f.write(f"Train Batchs: {modelInfo['nTrainBatchs']}\n")
    f.write(f"Train Samples: {modelInfo['nTrainSamples']}\n")
    f.write(f"Test Batchs: {modelInfo['nTestBatchs']}\n")
    f.write(f"Test Samples: {modelInfo['nTestSamples']}\n")
    f.write(f"Eval Batchs: {modelInfo['nEvalBatchs']}\n")
    f.write(f"Eval Samples: {modelInfo['nEvalSamples']}\n")
    
    # save model summary
    f.write("\nMODEL SUMMARY\n")
    summaryResults = summary(modelInfo["model"], input_size=[modelInfo["batchSize"], modelInfo["signalLength"]])
    f.write(f"{summaryResults}\n")
    
    # write test results
    f.write("\nRESULTS\n")
    for i in range(len(results["trainLoss"])):
        
        # get values for epoch
        trainLoss = results["trainLoss"][i]
        trainAcc = results["trainAcc"][i] * 100
        testLoss = results["testLoss"][i]
        testAcc = results["testAcc"][i] * 100
        
        # print out summary
        dataLine = (f"Epoch {i} | "
                    f"trainLoss: {trainLoss:.4f} | "
                    f"trainAcc: {trainAcc:.2f} | "
                    f"testLoss: {testLoss:.4f} | "
                    f"testAcc: {testAcc:.2f}")
        
        # save to file
        f.write(f"{dataLine}\n")
        
        # print out summary
        print(dataLine)
            
    f.close()
    
def saveAllResults(datasetPath: str,
                   modelInfo: modelInfoDict,
                   resultsTT: Dict[str, List],
                   resultsPT: Dict[str, np.ndarray] | None,
                   ) -> None:
    """ Saves all the results

    Parameters
    ----------
    datasetPath : str
        Path to dataset.
    modelInfo : modelInfoDict
        Dictionary of model information parameters. Full specification can be found in modelInfoDict class definition.
    resultsTT : Dict[str, List]
        A dictionary of training and testing loss as well as training and testing accuracy metrics. Each metric has a 
        value in a list for each epoch.     
            In the form: 
                {trainLoss: [...],
                trainAcc: [...],
                testLoss: [...],
                testAcc: [...]} 
            For example if training for nEpochs = 2: 
                {trainLoss: [2.0616, 1.0537],
                trainAcc: [0.3945, 0.3945],
                testLoss: [1.2641, 1.5706],
                testAcc: [0.3400, 0.2973]}
    resultsPT : Dict[str, np.ndarray] | None
        # TODO - define
    """
    
    # create the results folder
    print("creating the results directory ...")
    createDirectory(f"../results", verbose=True)
    
    # change the data directory to results in dataset path
    datasetPathOld = Path(datasetPath) # type: ignore
    datasetPathParts = list(datasetPathOld.parts)
    datasetPathParts[1] = "results"
    datasetPathResults = Path(*datasetPathParts)
    
    # create the dataset results directory
    datasetPathParts = datasetPathResults.parts
    createDirectory(Path(*datasetPathParts[:3]), verbose=True)
    createDirectory(Path(*datasetPathParts[:4]), verbose=True)
    createDirectory(datasetPathResults, verbose=True)
    
    # get the name of the cnn script that called it
    callerFrame = inspect.stack()[3] # TODO - not sure I like hardcoding this, but works for now
    fileName, _ = os.path.splitext(os.path.basename(callerFrame.filename))
    
    # save info results
    saveInfoResults(datasetPathResults=datasetPathResults,
                    fileName=fileName,
                    modelInfo=modelInfo,
                    results=resultsTT)
    
    # save train and test accuracies
    saveTrainTestResults(datasetPathResults=datasetPathResults,
                         fileName=fileName,
                         results=resultsTT)
    
    if resultsPT is None:
        # save probability and target results
        saveProbTargetResults(datasetPathResults=datasetPathResults,
                              fileName=fileName,
                              results=resultsPT)
    