import matplotlib.pyplot as plt
import numpy as np
import glob, os
from sklearn.metrics import roc_curve, auc
from typing import Dict, List

# TODO - add comments

def compareResults(dirPaths: list[str]) -> None:
    # TODO - check if this function is needed
    
    # create figure
    plt.figure()
    plotData = True

    # set the words that we want to remove from filename string
    wordToRemove = ["RESULTS", "date", "time"]

    # go through all directories in the list
    for currDir in dirPaths:
        
        # find all .npy data in the current directory
        files = glob.glob(f"{currDir}/*.npy")
        filesPT = glob.glob(f"{currDir}/*_PT.npy")
        for filePT in filesPT:
            files.remove(filePT)
        print(files)
        
        # check if any files are found
        if files:
            plotData = True
            print("Found files")
            for file in files:
                print(f"   {os.path.basename(file)}")
        else:
            print(f"No files found in path {currDir}")
            
        # open file data and plot
        for file in files:
            # load the file data
            fileData = np.load(file, allow_pickle=True)
            
            # get the data
            trainLoss = np.array(fileData.item().get('trainLoss'))
            testLoss = np.array(fileData.item().get('testLoss'))
            
            # get the filename
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0]
            
            # remove the words from filename string
            plotLabel = "_".join([word for word in filename.split("_") if word not in wordToRemove])
            
            # plot the data
            line1, = plt.plot(trainLoss, label={plotLabel})
            plt.plot(testLoss, color=line1.get_color(), linestyle="--")
    
    if plotData:
        plt.title("Training (solid) and Testing (dashed) Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()   
        plt.show()
        
def createMainResultsFile(dataDirPath,
                          dataTypePath):
    
    # create data path
    dataPath = dataDirPath / dataTypePath
    
    # read current info files in dir
    files = glob.glob(f"{dataPath}/resultsInfo/*.txt")
    
    # check if there any results present
    if not files:
        newResultsNum = 1
    else:
        # create empty results numbers array
        resultsNums = np.array([]).astype(np.int32)
        
        # go through each file and get result number
        for file in files:

            # get the file name
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0]

            # append result number to array
            resultsNums = np.append(resultsNums, int(filename.split('_')[1]))

        # new result number
        newResultsNum = np.max(resultsNums) + 1
        
    # create new results file
    # TODO - create results info folder
    f = open(f"{dataPath}/resultsInfo/RESULTS_{newResultsNum}.txt", "x")
    
    f.write(f"\nDATASETS TESTED\n")
    
    return f, newResultsNum

def appendMainResultsFile(file,
                          datasetName,
                          signalType):

    # append dataset to results file
    file.write(f"{datasetName}/{signalType}\n")
    
def plotLossCurves(results: Dict) -> None:
    """ Plots training and testing loss and accuracy curves.

    Parameters
    ----------
    results : Dict
        Dictionary containing list of training and testing loss and accuracies. 
            In the form of;
                {"trainLoss": [...],
                "trainAcc": [...],
                "testLoss": [...],
                "testAcc": [...]}
    """
    
    # get training and testing losses
    trainLoss = results["trainLoss"]
    testLoss = results["testLoss"]

    # get training and testing accuracies
    trainAcc = results["trainAcc"]
    testAcc = results["testAcc"]

    # create array of all epochs
    epochs = range(len(results["trainLoss"]))

    plt.figure(figsize=(15, 7))

    # plot loss
    plt.subplot(121)
    plt.plot(epochs, trainLoss, label="Train")
    plt.plot(epochs, testLoss, label="Test")
    plt.grid()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.xlim([epochs[0], epochs[-1]])
    plt.legend()

    # plot accuracy
    plt.subplot(122)
    plt.plot(epochs, trainAcc, label="Train")
    plt.plot(epochs, testAcc, label="Test")
    plt.grid()
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.xlim([epochs[0], epochs[-1]])
    plt.legend()
    
    plt.show()
    
def plotRocCurves(dirPath):
    
    # create figure
    plt.figure()
    plotData = False

    # set the words that we want to remove from filename string
    wordToRemove = ["RESULTS", "date", "time", "PT"]

    # go through all directories in the list
    for currDir in dirPaths:
        
        # find all .npy data in the current directory
        files = glob.glob(f"{currDir}/*_PT.npy")
        print(files)
        
        # check if any files are found
        if files:
            plotData = True
            print("Found files")
            for file in files:
                print(f"   {os.path.basename(file)}")
        else:
            print(f"No files found in path {currDir}")
            
        # open file data and plot
        for file in files:
    
            # load the file data
            fileData = np.load(file, allow_pickle=True)
            
            # get the data
            probs = np.array(fileData.item().get('probs')).squeeze()
            targets = np.array(fileData.item().get('targets')).squeeze()
            
            # compute ROC
            fpr, tpr, thresholds = roc_curve(targets, probs)
            roc_auc = auc(fpr, tpr)
            
            # get the filename
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0]
            
            # remove the words from filename string
            plotLabel = "_".join([word for word in filename.split("_") if word not in wordToRemove])
            
            # plot the data
            plt.plot(fpr, tpr, label=f'{plotLabel} (AUC = {roc_auc:.3f})')
    
    if plotData:
        plt.xscale('log')
        plt.title('ROC for Deep Learning Detector on Pulse-Compressed Radar Signals')
        plt.xlabel("False Alarm Rate (Pfa)")
        plt.ylabel("Probability of Detection (Pd)")
        plt.legend()
        plt.grid(True, which='both')   
        plt.show()
    
if __name__ == "__main__":
    
    # create the list of directories from which to plot the data from
    dirPaths = ["../../cnn_design/data/syntheticData_new/noise_CN_snr_10_train_5000_test_1000/pc"]
    
    # run the compare data function
    compareResults(dirPaths)
    
    # plot the roc's
    plotRocCurves(dirPaths)