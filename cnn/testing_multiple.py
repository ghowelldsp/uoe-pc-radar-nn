# add the top directory to the python system path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.runModel import trainModels, Datasets, ModelParams

if __name__ == "__main__":
    
    # dataset to train on
    datasets : Datasets =   {"testing": 
                                [
                                    ["complex gaussian", "complex weibull"], 
                                    ["10", "5"], 
                                    ["pc", "rx"]
                                ],
                            }
    
    # model parameters
    modelParams : ModelParams = {"nEpochs": 3,
                                 "batchSize": 2,
                                 "modelType": "basic1",
                                 "lossFn": "Cross Entropy Loss",
                                 "optimiser": "Adam",
                                 "optimiserLR": 1e-4}
    
    # train model on all datasets
    trainModels(datasets=datasets,
                modelParams=modelParams,
                evalModel=False,
                plotResults=False,
                verboseOutput=False)