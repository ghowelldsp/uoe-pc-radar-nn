import os
import shutil

def createDirectory(dirPath: str) -> None:
    """ Creates the directory specified by the directory path

    Parameters
    ----------
    dirPath : str
        Path to directory to create.
    """
    try:
        os.mkdir(dirPath)
        print(f"Directory '{dirPath}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dirPath}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dirPath}'.")
    except Exception as error:
        print(f"An error occurred: {error}")
        
def removeDirectory(dirPath: str) -> None:
    """ Removes the directory specified by the directory path

    Parameters
    ----------
    dirPath : str
        Path to directory to remove.
    """
    try:
        shutil.rmtree(dirPath, ignore_errors=False)
        print(f"Removed '{dirPath}'")
    except FileNotFoundError as error:
        print(f"No '{dirPath}' to remove")
    except Exception as error:
        print(f"{error} error occurred when trying to remove '{dirPath}'")