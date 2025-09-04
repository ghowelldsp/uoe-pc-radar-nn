import torch
import torch.nn as nn

# TODO - tidy up all models

def loadModel(modelType: str,
              device: torch.device,
              signalLen: int,
              verboseOutput: bool
              ) -> nn.Module:
    """ Load an instance of a model

    Parameters
    ----------
    modelType : str
        Model to load
    device : torch.device
        Device the model is trained on.
    signalLen : int
        Length of the input signal.
    verboseOutput : bool
        Verbose output information.

    Returns
    -------
    _type_ # TODO
        Instance of the model.

    Raises
    ------
    ValueError
        If the model type is not recognised.
    """
    
    # create an instance of the model
    match modelType:
        case "basic":
            model = model_basic(in_channels=signalLen)
        case "dynamicCNN":
            model = model_dynamicCnn0(nInFeatures=signalLen, avePoolKernelSize=32)
        case "dynamicCNN1":
            model = model_dynamicCnn1()
        case "dynamicCNN2":
            model = model_dynamicCnn2()
        case "dynamicCnn1_1":
            model = model_dynamicCnn1_1()
        case _:
            raise ValueError(f"model {modelType} not recognised")
        
    # send model to device
    model = model.to(device)
    
    if verboseOutput:
        print("\nMODELS\n")
        print(f"Input Features (Signal Length): {signalLen}")
        print(f"Model: {modelType}")
        print(f"Device: {device}")
        # TODO - fix this
        # print("Model Summary:")
        # summary(testModel, input_size=[BATCH_SIZE, sigLen])
        
    return model

class model_basic(nn.Module):
    """ Basic Model
    """ 
    
    def __init__(self,
                 in_channels: int):
    
        # initialise nn.Module class
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=in_channels, out_features=10), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=10, out_features=2)
            )
        
        self.classifier = nn.Sequential(
            nn.Flatten()
            )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        x = self.classifier(x)
        
        return x
    
class model_dynamicCnn0(nn.Module):
    """ Dynamic CNN - Test Model 0
    """ 
    
    def __init__(self,
                 nInFeatures: int,
                 avePoolKernelSize: int,
                 ):
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the number of input channels for the initial linear layer
        linInChannels = nInFeatures - avePoolKernelSize + 1
    
        self.block_1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=avePoolKernelSize, stride=1),
            nn.Linear(in_features=linInChannels, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten()
        )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        # x = self.classifier(x)
        
        return x
    
class model_dynamicCnn1(nn.Module):
    """ Dynamic CNN - Test Model 0
    """ 
    
    def __init__(self
                 ):
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the number of input channels for the initial linear layer
        # linInChannels = nInFeatures - avePoolKernelSize + 1
    
        self.block_1 = nn.Sequential(
            # nn.AvgPool1d(kernel_size=32, stride=8),
            nn.AvgPool1d(kernel_size=16, stride=8),
            # nn.Linear(in_features=497, out_features=64),
            # nn.Linear(in_features=997, out_features=64),
            # nn.Linear(in_features=197, out_features=64),
            nn.Linear(in_features=399, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten()
        )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x
    
class model_dynamicCnn1_1(nn.Module):
    """ Dynamic CNN - Test Model 0
    """ 
    
    def __init__(self
                 ):
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the number of input channels for the initial linear layer
        # linInChannels = nInFeatures - avePoolKernelSize + 1
    
        self.block_1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=16, stride=8),
            nn.Linear(in_features=399, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten()
        )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x

class model_dynamicCnn2(nn.Module):
    """ Dynamic CNN - Test Model 0
    """ 
    
    def __init__(self
                 ):
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the number of input channels for the initial linear layer
        # linInChannels = nInFeatures - avePoolKernelSize + 1
    
        self.block_1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=16, stride=8),
            nn.Linear(in_features=399, out_features=266),
            nn.ReLU(),
            nn.Linear(in_features=266, out_features=177),
            nn.ReLU(),
            nn.Linear(in_features=177, out_features=118),
            nn.ReLU(),
            nn.Linear(in_features=118, out_features=79),
            nn.ReLU(),
            nn.Linear(in_features=79, out_features=53),
            nn.ReLU(),
            nn.Linear(in_features=53, out_features=35),
            nn.ReLU(),
            nn.Linear(in_features=35, out_features=23),
            nn.ReLU(),
            nn.Linear(in_features=23, out_features=15),
            nn.ReLU(),
            nn.Linear(in_features=15, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=7),
            nn.ReLU(),
            nn.Linear(in_features=7, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=3),
            nn.ReLU(),
            nn.Linear(in_features=3, out_features=2)
        )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x  