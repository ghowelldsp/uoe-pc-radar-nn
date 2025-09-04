import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from typing import Tuple

from src.ml.utils import findClasses

class radarSignalDataloader(Dataset):
    
    def __init__(self, 
                 dataDir : str,
                 transform : torchvision.transforms.transforms.Compose | None = None 
                 ) -> None:
        """ Initialise the data loader class

        Parameters
        ----------
        dataDir : Path
            Path to data directory
        transform : torchvision.transforms.transforms.Compose | None, optional
            List of transforms to compose, by default None
        """
        
        # get all paths to each data
        self.paths = list(Path(dataDir).glob("*/*.npy"))
        
        # load the transforms
        self.transform = transform
        
        # create the classes and class-to-index variables
        self.classes, self.classToIndex = findClasses(dataDir)
        
    def loadData(self,
                 index: int
                 ) -> np.ndarray:
        """ Loads the data

        Parameters
        ----------
        index : int
            Index of the data to load.

        Returns
        -------
        np.ndarray
            Signal data.
        """
        
        # get the path of the data to load
        dataPath = self.paths[index]
        
        # load and return the data
        return np.load(dataPath)  
        
    def __len__(self) -> int:
        """ Gets the len of all the data

        Returns
        -------
        int
            Length of data.
        """
        
        return len(self.paths)
    
    def __getitem__(self,
                    index: int
                    ) -> Tuple[torch.Tensor, int]:
        """ Gets an item of data

        Parameters
        ----------
        index : int
            Index of the data to get.

        Returns
        -------
        Tuple[np.ndarray, int]
            Tuple containing the signal data and class index
        """
        
        # load the data
        data = self.loadData(index)
        
        # determine the class and class index of the data
        className = self.paths[index].parent.name
        classIdx = self.classToIndex[className]
        
        # apply transforms if required
        if self.transform:
            # TODO - fix warning
            return self.transform(data), classIdx
        else:
            data = torch.from_numpy(data).type(torch.float)
            return data, classIdx

def createDataloaders(datasetPath: str,
                      batchSize: int,
                      verboseOutput: bool):

    # create the transforms
    # TODO - see why the transforms are not working
    simpleTransform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # training and testing data directories
    trainPath = f"{datasetPath}/train"
    testPath = f"{datasetPath}/test"
    evalPath = f"{datasetPath}/eval"

    # load the training data with the dataloader
    trainingDataCustom = radarSignalDataloader(dataDir=trainPath,
                                               transform=None)

    testingDataCustom = radarSignalDataloader(dataDir=testPath,
                                              transform=None)
    
    evalDataCustom = radarSignalDataloader(dataDir=evalPath,
                                           transform=None)

    if verboseOutput:
        print("\nCUSTOM DATALOADERS\n")

        print("Training Data")
        print(f"   Length: {len(trainingDataCustom)}")
        print(f"   Class to index's: {trainingDataCustom.classToIndex}")
        print("Testing Data")
        print(f"   Length: {len(testingDataCustom)}")
        print(f"   Class to index's: {testingDataCustom.classToIndex}")
        print("Evaluation Data")
        print(f"   Length: {len(evalDataCustom)}")
        print(f"   Class to index's: {evalDataCustom.classToIndex}")

    # get the number of cpu's in the system
    noWorkers = os.cpu_count()
    if noWorkers is None:
        raise ValueError("No CPU cores availible")

    # load the data as a dataloader
    trainDL = DataLoader(dataset=trainingDataCustom,
                         batch_size=batchSize,
                         num_workers=noWorkers,
                         shuffle=True)

    testDL = DataLoader(dataset=testingDataCustom,
                        batch_size=batchSize,
                        num_workers=noWorkers,
                        shuffle=False)
    
    evalDL = DataLoader(dataset=evalDataCustom,
                        batch_size=batchSize,
                        num_workers=noWorkers,
                        shuffle=False)

    if verboseOutput:
        print("\nDATALOADERS (BATCH CREATION)\n")

        print(f"   Batch Size: {batchSize}")
        print("Training DL")
        print(f"   No. of Batches: {len(trainDL)}")
        print(f"   Total Sample Length: {len(trainDL.dataset)}") # type: ignore
        print("Testing DL")
        print(f"   No. of Batches: {len(testDL)}")
        print(f"   Total Sample Length: {len(testDL.dataset)}") # type: ignore
        print("Evaluation DL")
        print(f"   No. of Batches: {len(evalDL)}")
        print(f"   Total Sample Length: {len(evalDL.dataset)}") # type: ignore
    
    return trainDL, testDL, evalDL