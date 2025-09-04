import torch, random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import os

def availableDevices() -> torch.device:
    """ Available Devices
    
    Check what GPU devices are avalible on the systems. If no GPU devices are available then defaults to the CPU.
    
    Returns
    -------
    device : torch.device
        Return the GPU device avalible on the system. If no GPU is available, returns cpu.
    """
    if torch.cuda.is_available():
        # NVIDIA GPU
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU
        device = torch.device("mps")
    else:
        # Default to CPU if no GPU is available
        device = torch.device("cpu")
        
    return device

def setSeeds(seed: int = 42) -> None:
    """ Sets random seeds for all random processes and torch operations.

    Parameters
    ----------
    seed : int, optional
        Random seed to set, by default 42.
    """
    
    # set the seed for random and numpy random processes
    random.seed(seed)
    np.random.seed(seed)  

    # Set the seed for general torch operations
    torch.manual_seed(seed)
    
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    
    # set the seed for mpu
    torch.mps.manual_seed(seed)
    
    # set the seed for cpu
    torch.manual_seed(seed)
    
def findClasses(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """ Finds the classes of the data

    Parameters
    ----------
    directory : Path
        Directory path

    Returns
    -------
    Tuple[List[str], Dict[str, int]]
        (list_of_class_names, dict(class_name: idx...))

    Raises
    ------
    FileNotFoundError
        If not file found
    """
    
    # get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    return classes, class_to_idx
