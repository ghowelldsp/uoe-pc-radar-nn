# add the top directory to the python system path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.runModel import trainModels

if __name__ == "__main__":
    
    # dataset to train on
    datasets = {"syntheticData_final": 
                    [
                        ["complex gaussian"], 
                        ["10"], 
                        ["pc"]
                    ],
                }
    
    # model parameters
    modelParams = {"nEpochs": 10,
                   "batchSize": 128,
                   "modelType": "dynamicCNN1",
                   "loss function": "Cross Entropy Loss",
                   "optimiser": "Adam",
                   "optimiser learn rate": 1e-4}
    
    # train model on all datasets
    trainModels(datasets=datasets,
                modelParams=modelParams,
                plotResults=False,
                verboseOutput=False)