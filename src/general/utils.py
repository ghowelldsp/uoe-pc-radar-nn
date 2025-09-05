import os
import shutil
import random
import numpy as np

def setSeeds(seed: int = 42) -> None:
    """ Set seeds for random and numpy random.

    Parameters
    ----------
    seed : int, optional
        Random seed to set, by default 42.
    """
    
    # set the seed for random and numpy random processes
    random.seed(seed)
    np.random.seed(seed) 

def createDirectory(dirPath: str,
                    verbose: bool = False
                    ) -> None:
    """ Creates the directory specified by the directory path

    Parameters
    ----------
    dirPath : str
        Path to directory to create.
    """
    try:
        os.mkdir(dirPath)
        if verbose:
            print(f"Directory '{dirPath}' created successfully.")
    except FileExistsError:
        if verbose:
            print(f"Directory '{dirPath}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dirPath}'.")
        raise
    except Exception as error:
        print(f"An error occurred: {error}")
        raise
        
def removeDirectory(dirPath: str,
                    verbose: bool = False
                    ) -> None:
    """ Removes the directory specified by the directory path

    Parameters
    ----------
    dirPath : str
        Path to directory to remove.
    """
    try:
        shutil.rmtree(dirPath, ignore_errors=False)
        if verbose:
            print(f"Removed '{dirPath}'")
    except FileNotFoundError as error:
        if verbose:
            print(f"No '{dirPath}' to remove")
    except PermissionError:
        print(f"Permission denied: Unable to remove '{dirPath}'.")
    except Exception as error:
        print(f"{error} error occurred when trying to remove '{dirPath}'")
        
if __name__ == "__main__":
    
    createDirectory("test", verbose=True)
    createDirectory("test", verbose=True)
    removeDirectory("test", verbose=True)
    removeDirectory("test", verbose=True)